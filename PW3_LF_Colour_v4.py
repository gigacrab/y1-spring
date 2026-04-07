import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys

# ── Helper functions ──────────────────────────────────────────────────────────

def clamp(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    return value

def getSign(n):
    # Returns +1 for positive numbers, -1 for negative, 0 for zero.
    return (n > 0) - (n < 0)

# ── Command-line argument parsing ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Didn't input appropriate variables")

# ── Camera initialisation ─────────────────────────────────────────────────────

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

# ── PID persistent state ──────────────────────────────────────────────────────

total_error = 0
last_error  = 0
first       = True

# ── State machine setup ───────────────────────────────────────────────────────

STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"
STATE_TURN_90      = "TURN_90"

state = STATE_FOLLOW_BLACK

# ── Search / turn direction memory ────────────────────────────────────────────
# Instead of tracking which side the black line is on relative to the colour
# line (which requires both to be visible simultaneously and is often wrong),
# we capture the DIRECTION THE ROBOT WAS ALREADY STEERING at the moment the
# line is lost. This is "dead-reckoning by last intent":
#
#   last_error > 0  →  line was LEFT of centre  →  robot was turning LEFT
#   last_error < 0  →  line was RIGHT of centre →  robot was turning RIGHT
#
# search_turn_sign stores getSign(last_error) at the moment we enter SEARCH.
# In the motor block: turn_pwm = -SEARCH_SPEED * search_turn_sign
#   search_turn_sign = +1 → turn_pwm negative → left motor slows → turns LEFT  ✓
#   search_turn_sign = -1 → turn_pwm positive → left motor speeds → turns RIGHT ✓
#
# This works identically for colour line endings AND black line intersections
# because last_error is computed the same way regardless of which line is active.
search_turn_sign = 0    # Captured at the moment of entering SEARCH or TURN_90.
SEARCH_SPEED     = 0.35

# ── 90° turn detection ────────────────────────────────────────────────────────

TURN_90_SPEED   = 0.35
TURN_90_LOCKOUT = 0.5
turn_90_start   = None
turn_90_dir     = "right"  # Set by pixel-count geometry when a turn is detected.

# ── FOLLOW_BLACK dropout tolerance ───────────────────────────────────────────

black_miss_count = 0
BLACK_MISS_LIMIT = 20   # ~0.67 s at 30 fps before giving up and searching.

frame_count = 0

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    try:
        time_marker = time.perf_counter()

        # ── Image capture ─────────────────────────────────────────────────────
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Crop to the bottom half — the robot only needs to see the floor
        # immediately ahead. This also halves the work for every subsequent step.
        roi = frame[240:480, :]
        # HSV separates hue from brightness, making colour masks robust to
        # changes in ambient light that would ruin BGR-based thresholds.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ── Colour mask ───────────────────────────────────────────────────────
        red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        # ── Black mask — always runs, no ret gate ─────────────────────────────
        # Otsu's threshold value (ret) can legitimately exceed 180 on bright
        # floors even when a black line is clearly present. The area filter
        # below (7500–40000 px²) is the correct noise gate, not the ret value.
        imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── Find best colour contour ──────────────────────────────────────────
        color_cnts, _   = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_cnt = None
        color_cx        = None
        for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_color_cnt = cnt
                    color_cx        = int(M['m10'] / M['m00'])
            break   # Only the single largest valid contour is ever needed.

        # ── Find best black contour ───────────────────────────────────────────
        black_cnts, _   = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_black_cnt = None
        black_cx        = None
        for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_black_cnt = cnt
                    black_cx        = int(M['m10'] / M['m00'])
            break

        # ── Feature 2 – 90° horizontal contour detection ─────────────────────
        # A 90° turn makes the colour contour a wide horizontal bar.
        # boundingRect is sufficient: if width is >2.5× height, it's horizontal.
        color_is_horizontal = False
        bbox = None
        if valid_color_cnt is not None:
            bbox       = cv2.boundingRect(valid_color_cnt)
            x, y, w, h = bbox
            if h > 0 and w > (h * 2.5) and w > 150:
                color_is_horizontal = True

        # ── State machine transitions ─────────────────────────────────────────

        if state == STATE_TURN_90:
            elapsed_turn = time.perf_counter() - turn_90_start
            # Lockout prevents re-detecting the same horizontal bar that triggered
            # the turn and snapping back out of TURN_90 prematurely.
            if elapsed_turn > TURN_90_LOCKOUT:
                if valid_color_cnt is not None and not color_is_horizontal:
                    state = STATE_FOLLOW_COLOR
                    print("TURN_90 → FOLLOW_COLOR")
                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK
                    print("TURN_90 → FOLLOW_BLACK")

        elif state == STATE_SEARCH:
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
                print("SEARCH → FOLLOW_COLOR")
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK
                black_miss_count = 0
                print("SEARCH → FOLLOW_BLACK")

        else:
            # Normal FOLLOW transitions.
            if color_is_horizontal:
                # Pixel-count geometry tells us which way the colour line bends.
                # More coloured pixels on the left → the turn goes left, etc.
                left_px       = cv2.countNonZero(colour_mask[:, :320])
                right_px      = cv2.countNonZero(colour_mask[:, 320:])
                turn_90_dir   = "left" if left_px > right_px else "right"
                # Also capture directional memory in case the turn leads into a
                # SEARCH (i.e., colour line ends mid-turn).
                search_turn_sign = getSign(last_error)
                turn_90_start = time.perf_counter()
                state         = STATE_TURN_90
                print(f"90° TURN → {turn_90_dir}  (L={left_px} R={right_px})")

            elif valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

            elif valid_black_cnt is not None:
                state            = STATE_FOLLOW_BLACK
                black_miss_count = 0

            elif state == STATE_FOLLOW_COLOR:
                # Colour line gone, no black line visible either.
                # Capture the last steering direction before the line disappeared
                # and sweep in that same direction.
                search_turn_sign = getSign(last_error)
                state            = STATE_SEARCH
                print(f"FOLLOW_COLOR → SEARCH  "
                      f"(last_error={last_error:.3f}, sign={search_turn_sign})")

            elif state == STATE_FOLLOW_BLACK:
                # Black line dropout — tolerate brief gaps before giving up.
                black_miss_count += 1
                if black_miss_count >= BLACK_MISS_LIMIT:
                    search_turn_sign = getSign(last_error)
                    black_miss_count = 0
                    state            = STATE_SEARCH
                    print(f"FOLLOW_BLACK → SEARCH after {BLACK_MISS_LIMIT} misses  "
                          f"(last_error={last_error:.3f}, sign={search_turn_sign})")

        # ── Motor control ─────────────────────────────────────────────────────

        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                black_miss_count = 0  # Successful detection resets the dropout counter.

                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

                # Normalise error: +1 = line far left, -1 = line far right, 0 = centred.
                error        = (320 - cx) / 320
                total_error += error * elapsed_time

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first      = False
                    diff_error = 0

                pid        = kp * error + ki * total_error + kd * diff_error
                last_error = error

                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

            else:
                # Contour vanished between detection and here (noisy frame).
                pid       = getSign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_SEARCH:
            # search_turn_sign = +1 means robot was turning left → keep turning left.
            # Left turn requires: right motor > left motor → turn_pwm must be negative.
            # Therefore: turn_pwm = -SEARCH_SPEED * search_turn_sign
            #   sign=+1 → turn_pwm = -SEARCH_SPEED → left slows, right speeds → LEFT  ✓
            #   sign=-1 → turn_pwm = +SEARCH_SPEED → left speeds, right slows → RIGHT ✓
            turn_pwm  = -SEARCH_SPEED * search_turn_sign
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        elif state == STATE_TURN_90:
            # turn_90_dir comes from pixel-count geometry (left_px vs right_px above).
            # Same sign convention as SEARCH.
            turn_pwm  = -TURN_90_SPEED if turn_90_dir == "left" else TURN_90_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        else:
            left_pwm  = base_speed
            right_pwm = base_speed

        movement.move(clamp(left_pwm, -1, 1), clamp(right_pwm, -1, 1))

        # ── Debug display (every 5th frame to protect PID timing) ────────────
        frame_count += 1
        if frame_count % 5 == 0:
            if valid_color_cnt is not None and bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)
                ratio = w / h if h > 0 else 0
                cv2.putText(im2, f"W:{w} H:{h} R:{ratio:.1f}", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            sample = hsv[110:120, 310:320]
            print(f"HSV={sample.mean(axis=(0,1)).astype(int)}  "
                  f"state={state}  err={last_error:.3f}  sign={search_turn_sign}")

            cv2.putText(im2, f"STATE: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("1. Color Mask", colour_mask)
            cv2.imshow("2. Tracking & Math", im2)

            if cv2.waitKey(1) == 27:
                movement.move(0, 0)
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred – {e}")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
# cv2.destroyAllWindows()