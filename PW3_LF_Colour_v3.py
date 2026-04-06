import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys

def clamp(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    return value

def getSign(n):
    return (n > 0) - (n < 0)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Didn't input appropriate variables")

# ── Picamera2 init ────────────────────────────────────────────────────────────
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

# ── PID state ─────────────────────────────────────────────────────────────────
error       = 0
total_error = 0
last_error  = 0
diff_error  = 0
first       = True

# ── State machine constants ───────────────────────────────────────────────────
STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"       # colour lost → sweep toward black_line_side
STATE_TURN_90      = "TURN_90"      # executing a 90-degree turn on the colour line

# ── State machine variables ───────────────────────────────────────────────────
state           = STATE_FOLLOW_BLACK

# Feature 1 – side memory
black_line_side = "right"   # which side the black line was on relative to the colour line
SEARCH_SPEED    = 0.35      # hard-turn PWM offset while searching (tune if needed)

# Feature 2 – 90° turn
ASPECT_THRESHOLD = 3.5      # width/height ratio above which a contour is "horizontal"
TURN_90_SPEED    = 0.35     # hard-turn PWM offset during the 90° manoeuvre
TURN_90_LOCKOUT  = 1      # seconds to ignore re-acquisition (prevents instant exit)
turn_90_start    = None
turn_90_dir      = "right"


while True:
    try:
        time_marker = time.perf_counter()

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi   = frame[240:480, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        sample = hsv[110:120, 310:320]
        print(f"HSV centre sample: {sample.mean(axis=(0,1)).astype(int)}  state={state}")

        # ── Colour mask (unchanged from original) ─────────────────────────────
        red_mask    = cv2.inRange(hsv, np.array([111, 30,  180]), np.array([131, 185, 230]))
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 205]), np.array([105, 255, 255]))
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        # ── Black mask (now always computed – needed for Feature 1 memory) ────
        imgray        = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh   = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── Find best colour contour ──────────────────────────────────────────
        color_cnts, _ = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_cnt = None
        color_cx        = None
        for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_color_cnt = cnt
                    color_cx        = int(M['m10'] / M['m00'])
                break

        # ── Find best black contour ───────────────────────────────────────────
        valid_black_cnt = None
        black_cx        = None
        if ret < 180:
            black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
                if 7500 <= cv2.contourArea(cnt) <= 40000:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        valid_black_cnt = cnt
                        black_cx        = int(M['m10'] / M['m00'])
                    break

        # ── Feature 1 – Update black-line-side memory ─────────────────────────
        # Whenever both lines are visible, compare their centroids and record
        # which side the black line sits on relative to the colour line.
        # This memory is used by SEARCH mode if the colour line vanishes suddenly.
        if valid_color_cnt is not None and valid_black_cnt is not None:
            if color_cx is not None and black_cx is not None:
                black_line_side = "right" if color_cx < black_cx else "left"

        # ── Feature 2 – Detect a horizontal colour contour (90° turn) ─────────
        # cv2.minAreaRect returns the tightest rotated rectangle around the contour.
        # If its longer dimension is > ASPECT_THRESHOLD times the shorter one the
        # contour is essentially a horizontal bar – the robot is looking straight
        # down the barrel of a 90° turn.
        color_is_horizontal = False
        if valid_color_cnt is not None:
            _, (w, h), _ = cv2.minAreaRect(valid_color_cnt)
            longer  = max(w, h)
            shorter = min(w, h)
            if shorter > 0 and (longer / shorter) > ASPECT_THRESHOLD:
                color_is_horizontal = True

        # ── State-machine transitions ──────────────────────────────────────────

        if state == STATE_TURN_90:
            # Wait out the lockout so we don't immediately exit on the same line
            # that triggered the turn.  After it expires, check for re-acquisition.
            elapsed_turn = time.perf_counter() - turn_90_start
            if elapsed_turn > TURN_90_LOCKOUT:
                if valid_color_cnt is not None and not color_is_horizontal:
                    state = STATE_FOLLOW_COLOR
                    print("TURN_90 → FOLLOW_COLOR (reacquired colour line)")
                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK
                    print("TURN_90 → FOLLOW_BLACK (reacquired black line)")
                # else: no line yet – keep turning

        elif state == STATE_SEARCH:
            # Exit search as soon as any line is re-acquired
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
                print("SEARCH → FOLLOW_COLOR")
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK
                print("SEARCH → FOLLOW_BLACK")
            # else: keep sweeping

        else:
            # ── Normal FOLLOW states ──────────────────────────────────────────
            if color_is_horizontal:
                # The colour line is making a 90° turn.
                # Determine left vs right by counting coloured pixels in each half
                # of the ROI.  More pixels on the left → the line goes left, etc.
                left_px  = cv2.countNonZero(colour_mask[:, :320])
                right_px = cv2.countNonZero(colour_mask[:, 320:])
                turn_90_dir   = "left" if left_px > right_px else "right"
                turn_90_start = time.perf_counter()
                state = STATE_TURN_90
                print(f"90° TURN detected → turning {turn_90_dir}  "
                      f"(left_px={left_px}, right_px={right_px})")

            elif valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

            elif state == STATE_FOLLOW_COLOR:
                # Colour line just disappeared and no black line in sight either.
                # Enter Search Mode using the remembered black_line_side.
                state = STATE_SEARCH
                print(f"FOLLOW_COLOR → SEARCH  (turning {black_line_side})")

            # If state was FOLLOW_BLACK and both contours are gone, stay in
            # FOLLOW_BLACK.  The motor section below handles the fallback with
            # getSign(last_error), preserving original behaviour.

        # ── Motor control ─────────────────────────────────────────────────────
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
            # Pick the correct contour/centroid for the current follow state
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                # ── PID (identical math to original) ─────────────────────────
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

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
                # State says follow, but contour vanished this frame before the
                # state machine could transition – fall back to last-error sign.
                print(f"Contour lost mid-frame ({state}) – using last_error sign")
                pid       = getSign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_SEARCH:
            # Hard-turn toward the side where the black line was last seen.
            # SEARCH_SPEED is positive → right turn; negative → left turn.
            turn_pwm  =  SEARCH_SPEED if black_line_side == "right" else -SEARCH_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        elif state == STATE_TURN_90:
            # Hard-turn in the direction the 90° geometry told us.
            turn_pwm  =  TURN_90_SPEED if turn_90_dir == "right" else -TURN_90_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        else:
            # Should never reach here, but safe fallback just in case.
            left_pwm  = base_speed
            right_pwm = base_speed

        clamped_left_pwm  = clamp(left_pwm,  -1, 1)
        clamped_right_pwm = clamp(right_pwm, -1, 1)
        movement.move(clamped_left_pwm, clamped_right_pwm)

        '''
        cv2.imshow("contours", im2)
        if cv2.waitKey(1) == 27:
            movement.move(0, 0)
            break
        '''

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred – {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
# cv2.destroyAllWindows()