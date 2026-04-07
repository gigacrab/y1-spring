import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys

# Helper functions

def clamp(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    return value

def getSign(n):
    # Returns +1 for positive numbers, -1 for negative, 0 for zero.
    return (n > 0) - (n < 0)

# Command-line argument parsing

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Didn't input appropriate variables")

# Camera initialisation

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)  # Allow the sensor's auto-exposure to settle before the loop begins.

# PID persistent state

total_error = 0   # Accumulates error over time (integral term).
last_error  = 0   # Stores previous frame's error (derivative term).
first       = True  # Guards against a nonsense derivative on the very first frame.

# State machine setup

STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"
STATE_TURN_90      = "TURN_90"

state = STATE_FOLLOW_BLACK  # Robot starts by looking for a black line.

# Feature 1 – Memory
black_line_side = "right"  # Last-known side of the black line relative to the colour line.
SEARCH_SPEED    = 0.35     # PWM offset applied as a hard-turn during SEARCH state.

# Feature 2 – 90° turn detection
TURN_90_SPEED   = 1     # PWM offset applied as a hard-turn during TURN_90 state.
TURN_90_LOCKOUT = 0.5      # Seconds to keep turning before checking for re-acquisition.
turn_90_start   = None     # Timestamp of when the current 90° turn began.
turn_90_dir     = "right"  # Direction to turn during TURN_90 state.

frame_count = 0  # Used to skip expensive display operations to every 5th frame.

# Main loop

while True:
    try:
        # Record the start of this frame to calculate dt for PID.
        time_marker = time.perf_counter()

        # Image capture and preprocessing
        
        # Capture a raw BGRA frame from the Picamera2 sensor.
        frame = picam2.capture_array()
        # Convert 4-channel BGRA → 3-channel BGR, which all OpenCV functions expect.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Crop to the bottom half only. The robot only cares about what is
        # immediately ahead of it; the top half of the frame is the far distance
        # and introduces noise. This also halves the pixel data for all subsequent steps.
        roi = frame[240:480, :]

        # Convert the ROI to HSV colour space. HSV separates colour (Hue) from
        # brightness (Value), making colour thresholds far more robust to lighting
        # changes than BGR-based thresholds would be.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Colour detection

        # Create a binary mask where pixels matching the red HSV range are white (255)
        # and everything else is black (0).
        red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
        # Same process for yellow.
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        # Merge both colour masks. A pixel is "coloured" if it matched either red OR yellow.
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        # ── Black line detection (always runs, needed for Feature 1 memory) ───

        # Convert the ROI to greyscale for Otsu thresholding.
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # THRESH_BINARY_INV + THRESH_OTSU: Otsu automatically picks the optimal
        # threshold value that separates the image into two classes (line vs background).
        # The INV flag makes the black line appear WHITE in the mask, which is what
        # findContours needs to work with.
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find best colour contour

        color_cnts, _ = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_cnt = None
        color_cx        = None
        # Sort all colour contours largest-first, then take the first one that
        # falls within the expected line area range (7500–40000 px²).
        # Too small = noise. Too large = the robot is sitting on the line / card border.
        for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:  # Guard: moments of an empty contour would cause division by zero.
                    valid_color_cnt = cnt
                    color_cx        = int(M['m10'] / M['m00'])  # Centroid X = m10 / m00
            break  # We only ever want the single largest valid contour.

        # Find best black contour

        valid_black_cnt = None
        black_cx        = None
        # ret is the Otsu threshold value. If it exceeds 180, the scene is likely
        # very bright/washed out with no real black line, so we skip black detection
        # to avoid picking up shadows or noise as a false line.
        if ret < 180:
            black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
                if 7500 <= cv2.contourArea(cnt) <= 40000:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        valid_black_cnt = cnt
                        black_cx        = int(M['m10'] / M['m00'])
                break

        # ── Feature 1 – Update black-line-side memory ──
        # This block runs passively every frame where BOTH lines are visible.
        # By the time the colour line ends and we need to search, this variable
        # will always hold the most recently confirmed answer.
        # Note: the inner None check is not needed — color_cx is guaranteed to be
        # set whenever valid_color_cnt is set (they are assigned in the same branch).
        if valid_color_cnt is not None and valid_black_cnt is not None:
            black_line_side = "right" if color_cx < black_cx else "left"

        # ── Feature 2 – Detect a 90° horizontal colour contour ──
        # cv2.boundingRect gives us an upright axis-aligned rectangle: (x, y, w, h).
        # When the robot approaches a 90° turn the coloured line becomes a wide
        # horizontal bar across the screen, making width >> height.
        # We cache x,y,w,h here so the debug block below can reuse it without
        # calling boundingRect a second time.
        color_is_horizontal = False
        bbox = None  # Cached bounding box for reuse in the debug display block.
        if valid_color_cnt is not None:
            bbox = cv2.boundingRect(valid_color_cnt)
            x, y, w, h = bbox
            # Trigger if: height is non-zero (safety), width is more than 2.5×
            # the height (horizontal bar shape), and width > 150 px (minimum size
            # so tiny noise blobs can't accidentally trigger a turn).
            if h > 0 and w > (h * 2.5) and w > 150:
                color_is_horizontal = True

        # State machine transitions
        # Priority order: TURN_90 and SEARCH are handled first (they are "active"
        # states that need to watch for an exit condition). The normal FOLLOW states
        # are handled last in the else branch.

        if state == STATE_TURN_90:
            elapsed_turn = time.perf_counter() - turn_90_start
            # The lockout period prevents the robot from instantly re-detecting the
            # same horizontal bar it just started turning away from and cancelling
            # the turn before it has physically rotated.
            if elapsed_turn > TURN_90_LOCKOUT:
                if valid_color_cnt is not None and not color_is_horizontal:
                    state = STATE_FOLLOW_COLOR
                    print("TURN_90 → FOLLOW_COLOR (reacquired colour line)")
                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK
                    print("TURN_90 → FOLLOW_BLACK (reacquired black line)")
                # If neither line is visible yet, we stay in TURN_90 and keep turning.

        elif state == STATE_SEARCH:
            # Exit the moment any valid line is reacquired.
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
                print("SEARCH → FOLLOW_COLOR")
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK
                print("SEARCH → FOLLOW_BLACK")

        else:
            # Normal FOLLOW state transition logic.
            if color_is_horizontal:
                # Count coloured pixels in the left vs right half of the ROI.
                # The half with more pixels is the direction the line continues into.
                left_px  = cv2.countNonZero(colour_mask[:, :320])
                right_px = cv2.countNonZero(colour_mask[:, 320:])
                turn_90_dir   = "left" if left_px > right_px else "right"
                turn_90_start = time.perf_counter()
                state = STATE_TURN_90
                print(f"90° TURN detected → turning {turn_90_dir} "
                      f"(left_px={left_px}, right_px={right_px})")

            elif valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

            elif state == STATE_FOLLOW_COLOR:
                # The colour line has vanished and there is no black line to fall
                # back on either. Use the remembered side to initiate a sweep.
                state = STATE_SEARCH
                print(f"FOLLOW_COLOR → SEARCH (turning {black_line_side})")

            # If state was FOLLOW_BLACK and all contours are gone, we do NOT enter
            # SEARCH — we stay in FOLLOW_BLACK. The motor block below handles this
            # gracefully using getSign(last_error) as a course-hold.

        # ── Motor control ─────────────────────────────────────────────────────

        im2 = np.zeros((240, 640, 3), dtype=np.uint8)  # Black canvas for debug drawing.

        if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                # dt: time elapsed this frame in seconds. Used to integrate and
                # differentiate error correctly regardless of loop speed.
                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001  # Prevent divide-by-zero on very fast loops.

                # Normalise error to the range [-1, +1].
                # cx=0 (far left) → error=+1. cx=640 (far right) → error≈-1.
                # cx=320 (centre) → error=0. A positive error means the line is
                # left of centre, and the robot must steer left to correct.
                error = (320 - cx) / 320

                # Integral: accumulates signed error over time. Corrects for
                # steady-state bias (e.g. robot drifting on a straight line).
                total_error += error * elapsed_time

                # Derivative: rate of change of error. Damps oscillation.
                # Skipped on the first frame because last_error is 0 by default,
                # which would produce a massive false spike.
                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first      = False
                    diff_error = 0

                # Combine all three PID terms.
                pid        = kp * error + ki * total_error + kd * diff_error
                last_error = error  # Store for next frame's derivative calculation.

                # PID output steers the robot: add to left, subtract from right.
                # A positive pid (line is left) → left motor speeds up, right slows
                # down → robot turns left to recentre.
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

            else:
                # Edge case: state machine says "follow" but contour vanished between
                # the detection block and here (can happen on noisy frames).
                # Use the sign of the last known error to coast in the right direction.
                print(f"Contour lost mid-frame ({state}) – using last_error sign")
                pid       = getSign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_SEARCH:
            # Apply a fixed hard-turn offset toward the side where the black line
            # was last confirmed to be. Negative offset → robot turns right.
            turn_pwm  = -SEARCH_SPEED if black_line_side == "right" else SEARCH_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        elif state == STATE_TURN_90:
            turn_pwm  = -TURN_90_SPEED if turn_90_dir == "right" else TURN_90_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        else:
            # Defensive fallback — should never be reached with the current state set.
            left_pwm  = base_speed
            right_pwm = base_speed

        # Clamp both PWM values to the [-1, 1] range the motor driver expects.
        clamped_left_pwm  = clamp(left_pwm,  -1, 1)
        clamped_right_pwm = clamp(right_pwm, -1, 1)
        movement.move(clamped_left_pwm, clamped_right_pwm)

        # ── Debug display (throttled to every 5th frame) ──────────────────────
        # imshow + waitKey are expensive. Running them every frame would cap the
        # loop speed and corrupt the PID's timing. Running every 5th frame gives
        # a ~15 fps preview with negligible impact on the control loop.
        frame_count += 1
        if frame_count % 5 == 0:

            if valid_color_cnt is not None and bbox is not None:
                x, y, w, h = bbox  # Reuse the cached result — no second call to boundingRect.
                cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)
                ratio = w / h if h > 0 else 0
                cv2.putText(im2, f"W:{w} H:{h} Ratio:{ratio:.1f}", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                # Also print the HSV sample here, throttled, so it doesn't spam stdout.
                sample = hsv[110:120, 310:320]
                print(f"HSV centre sample: {sample.mean(axis=(0,1)).astype(int)}  state={state}")

            cv2.putText(im2, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("1. Color Mask", colour_mask)
            cv2.imshow("2. Tracking & Math", im2)

            if cv2.waitKey(1) == 27:  # ESC to quit.
                movement.move(0, 0)
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred – {e}")
        break

# Cleanup
movement.move(0, 0)   # Stop motors immediately.
movement.pi.stop()    # Release the pigpio connection.
picam2.stop()
picam2.close()
# cv2.destroyAllWindows()