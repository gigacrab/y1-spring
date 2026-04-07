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
    return (n > 0) - (n < 0)

# ── Command-line arguments ────────────────────────────────────────────────────

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

# ── PID state ─────────────────────────────────────────────────────────────────

total_error = 0
last_error  = 0
first       = True

# ── EMA direction memory ──────────────────────────────────────────────────────
# Instead of using last_error (a single noisy frame) to decide which way to
# search after losing a line, we maintain an Exponential Moving Average of the
# error. This smooths out frame-to-frame noise and gives a direction that
# reflects the robot's sustained trend over the past ~10–20 frames.
#
# Formula each frame: ema_error = EMA_ALPHA * error + (1-EMA_ALPHA) * ema_error
# EMA_ALPHA = 0.25 means each new frame contributes 25% weight; history 75%.
# Raising alpha makes it respond faster but retain less history.
ema_error  = 0.0
EMA_ALPHA  = 0.25

# search_turn_sign is set to getSign(ema_error) whenever we enter SEARCH.
# It tells the motor block which direction to sweep.
search_turn_sign = 0

# ── Adaptive corner speed ─────────────────────────────────────────────────────
# At large errors (sharp corners), the robot automatically slows down, giving
# the PID more time to pull it around the bend before it overshoots.
#
# effective_speed = base_speed * (1.0 - CORNER_BRAKE * abs(error))
#   abs(error) = 0   (centred, straight) → full base_speed
#   abs(error) = 1.0 (maximum corner)   → base_speed * (1 - CORNER_BRAKE)
#
# CORNER_BRAKE = 0.5 means the robot slows to 50% speed at the sharpest corner.
# Raise this if the robot still overshoots. Lower it if it's too sluggish.
CORNER_BRAKE = 0.5

# ── State machine ─────────────────────────────────────────────────────────────

STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"
STATE_TURN_90      = "TURN_90"

state = STATE_FOLLOW_BLACK

SEARCH_SPEED    = 0.35
TURN_90_SPEED   = 0.35
TURN_90_LOCKOUT = 0.5
turn_90_start   = None
turn_90_dir     = "right"

black_miss_count = 0
BLACK_MISS_LIMIT = 20

frame_count = 0

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    try:
        time_marker = time.perf_counter()

        # ── Image capture and crop ────────────────────────────────────────────
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi   = frame[240:480, :]   # Bottom half only — closer = more reliable.
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ── Near-zone ROI for centroid (bottom third of the roi) ─────────────
        # On sharp corners, the full contour becomes an L-shape and its centroid
        # is pulled toward the inside of the bend, under-reporting how far the
        # robot needs to turn. The near zone (what's directly under the wheels)
        # is more immediate and accurate for the centroid X calculation.
        # Detection still uses the full roi; only the centroid uses near_roi.
        near_roi     = roi[160:240, :]    # Bottom 80 px of the 240px roi.
        near_hsv     = cv2.cvtColor(near_roi, cv2.COLOR_BGR2HSV)
        near_imgray  = cv2.cvtColor(near_roi, cv2.COLOR_BGR2GRAY)

        # ── Colour masks ──────────────────────────────────────────────────────
        red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        # Near-zone colour mask for centroid refinement.
        nr_mask = cv2.bitwise_or(
            cv2.inRange(near_hsv, np.array([105, 30,  100]), np.array([140, 255, 255])),
            cv2.inRange(near_hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        )

        # ── Greyscale / Otsu for black detection ──────────────────────────────
        imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, near_thresh = cv2.threshold(
            near_imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # ── Helper: compute centroid X from a binary mask (near zone) ─────────
        # Returns the centroid X from the near-zone mask if enough pixels exist,
        # otherwise falls back to the full-contour centroid passed in as backup.
        def near_cx_or_fallback(near_mask, fallback_cx):
            M = cv2.moments(near_mask)
            if M['m00'] > 500:            # Enough pixels to trust the centroid.
                return int(M['m10'] / M['m00'])
            return fallback_cx            # Near zone is empty — use full centroid.

        # ── Find best colour contour (full roi for detection) ─────────────────
        color_cnts, _   = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_cnt = None
        color_cx        = None
        for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_color_cnt = cnt
                    full_cx         = int(M['m10'] / M['m00'])
                    # Refine with near-zone centroid if possible.
                    color_cx        = near_cx_or_fallback(nr_mask, full_cx)
            break

        # ── Find best black contour ───────────────────────────────────────────
        black_cnts, _   = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_black_cnt = None
        black_cx        = None
        for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_black_cnt = cnt
                    full_cx         = int(M['m10'] / M['m00'])
                    black_cx        = near_cx_or_fallback(near_thresh, full_cx)
            break

        # ── 90° horizontal detection ──────────────────────────────────────────
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
                state            = STATE_FOLLOW_BLACK
                black_miss_count = 0
                print("SEARCH → FOLLOW_BLACK")

        else:
            if color_is_horizontal:
                left_px          = cv2.countNonZero(colour_mask[:, :320])
                right_px         = cv2.countNonZero(colour_mask[:, 320:])
                turn_90_dir      = "left" if left_px > right_px else "right"
                search_turn_sign = getSign(ema_error)   # Save EMA direction too.
                turn_90_start    = time.perf_counter()
                state            = STATE_TURN_90
                print(f"90° TURN → {turn_90_dir}  (L={left_px} R={right_px})")

            elif valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

            elif valid_black_cnt is not None:
                state            = STATE_FOLLOW_BLACK
                black_miss_count = 0

            elif state == STATE_FOLLOW_COLOR:
                # Capture the EMA-smoothed direction before entering SEARCH.
                # ema_error reflects the past ~10–20 frames of steering, not just
                # the last frame, so it's a much more reliable direction signal.
                search_turn_sign = getSign(ema_error)
                state            = STATE_SEARCH
                print(f"FOLLOW_COLOR → SEARCH  "
                      f"(ema={ema_error:.3f}, sign={search_turn_sign})")

            elif state == STATE_FOLLOW_BLACK:
                black_miss_count += 1
                if black_miss_count >= BLACK_MISS_LIMIT:
                    search_turn_sign = getSign(ema_error)
                    black_miss_count = 0
                    state            = STATE_SEARCH
                    print(f"FOLLOW_BLACK → SEARCH after {BLACK_MISS_LIMIT} misses  "
                          f"(ema={ema_error:.3f}, sign={search_turn_sign})")

        # ── Motor control ─────────────────────────────────────────────────────

        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                black_miss_count = 0

                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

                error        = (320 - cx) / 320
                total_error += error * elapsed_time

                # Update the EMA every frame a line is visible.
                # This is the core of the direction memory improvement.
                ema_error = EMA_ALPHA * error + (1 - EMA_ALPHA) * ema_error

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first      = False
                    diff_error = 0

                pid        = kp * error + ki * total_error + kd * diff_error
                last_error = error

                # Adaptive speed: scale down proportionally to how far off-centre
                # the line is. On a straight (error≈0) → full speed.
                # On a sharp corner (error≈1) → speed * (1 - CORNER_BRAKE).
                effective_speed = base_speed * (1.0 - CORNER_BRAKE * abs(error))

                left_pwm  = effective_speed + pid
                right_pwm = effective_speed - pid

            else:
                pid       = getSign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_SEARCH:
            # search_turn_sign = +1 → was turning left → keep turning left.
            # Turning left = right motor faster = negative turn_pwm.
            turn_pwm  = -SEARCH_SPEED * search_turn_sign
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        elif state == STATE_TURN_90:
            turn_pwm  = -TURN_90_SPEED if turn_90_dir == "left" else TURN_90_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        else:
            left_pwm  = base_speed
            right_pwm = base_speed

        movement.move(clamp(left_pwm, -1, 1), clamp(right_pwm, -1, 1))

        # ── Debug display (throttled) ─────────────────────────────────────────
        frame_count += 1
        if frame_count % 5 == 0:
            if valid_color_cnt is not None and bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)
                ratio = w / h if h > 0 else 0
                cv2.putText(im2, f"W:{w} H:{h} R:{ratio:.1f}", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            sample = hsv[110:120, 310:320]
            print(f"HSV={sample.mean(axis=(0,1)).astype(int)}  state={state}  "
                  f"ema={ema_error:.3f}  sign={search_turn_sign}")

            cv2.putText(im2, f"STATE: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im2, f"EMA:{ema_error:.2f} spd:{base_speed*(1-CORNER_BRAKE*abs(last_error)):.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.imshow("1. Color Mask", colour_mask)
            cv2.imshow("2. Tracking & Math", im2)

            if cv2.waitKey(1) == 27:
                movement.move(0, 0)
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred – {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
# cv2.destroyAllWindows()