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

def get_sign(n):
    return (n > 0) - (n < 0)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Need: base_speed kp ki kd")

# ── Camera init ───────────────────────────────────────────────────────────────
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

# ── State machine ─────────────────────────────────────────────────────────────
STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_RECOVERY     = "RECOVERY"   # colour ended → arc back toward black

state      = STATE_FOLLOW_BLACK
last_state = STATE_FOLLOW_BLACK

# ── Settings ──────────────────────────────────────────────────────────────────
RECOVERY_SPEED = 0.5  # PWM offset applied during the recovery arc

# Remembers which direction the robot turned to get onto the colour line.
# When colour ends we turn the SAME direction to arc back to the black line.
# Initialised to "left" as a safe default; overwritten before first use.
turn_memory = "left"

frame_count = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    try:
        time_marker = time.perf_counter()

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)   # Picamera2 gives BGRA
        roi   = frame[240:480, :]

        # ── Colour mask ───────────────────────────────────────────────────────
        hsv         = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask    = cv2.inRange(hsv, np.array([105,  30, 100]), np.array([140, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        # ── Black mask ────────────────────────────────────────────────────────
        imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ── Find largest valid colour contour ─────────────────────────────────
        color_cnts, _ = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_color_cnt = None
        color_cx        = None
        for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_color_cnt = cnt
                    color_cx = int(M['m10'] / M['m00'])
                break

        # ── Find largest valid black contour ──────────────────────────────────
        valid_black_cnt = None
        black_cx        = None
        if ret < 180:
            black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
                if 7500 <= cv2.contourArea(cnt) <= 40000:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        valid_black_cnt = cnt
                        black_cx = int(M['m10'] / M['m00'])
                    break

        # ── Turn memory update ────────────────────────────────────────────────
        # While still on the black line, keep watching where the colour appears.
        # colour to the LEFT  (cx < 320) → robot will turn LEFT  to follow it
        # colour to the RIGHT (cx > 320) → robot will turn RIGHT to follow it
        # The last reading before the state switches is the one that matters.
        if state == STATE_FOLLOW_BLACK and valid_color_cnt is not None:
            turn_memory = "left" if color_cx < 320 else "right"

        # ── State transitions ─────────────────────────────────────────────────
        if state == STATE_FOLLOW_BLACK:
            if valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

        elif state == STATE_FOLLOW_COLOR:
            if valid_color_cnt is None:
                # Colour line ended. Switch to recovery arc.
                print(f"Colour lost — recovering with turn: {turn_memory}")
                state = STATE_RECOVERY

        elif state == STATE_RECOVERY:
            if valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

        # ── PID reset on state change ─────────────────────────────────────────
        if state != last_state:
            total_error = 0
            last_error  = 0
            first       = True
            last_state  = state

        # ── Motor control ─────────────────────────────────────────────────────
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if state in (STATE_FOLLOW_BLACK, STATE_FOLLOW_COLOR):
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = max(time.perf_counter() - time_marker, 0.0001)

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
                # Brief mid-frame loss: keep curving in the last known direction
                pid       = get_sign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_RECOVERY:
            # Turn the same way the robot turned to get onto the colour line.
            # Larger left_pwm  → turns LEFT  (matches their PID convention)
            # Larger right_pwm → turns RIGHT
            if turn_memory == "left":
                left_pwm  = base_speed + RECOVERY_SPEED
                right_pwm = base_speed - RECOVERY_SPEED
            else:  # "right"
                left_pwm  = base_speed - RECOVERY_SPEED
                right_pwm = base_speed + RECOVERY_SPEED

        else:
            left_pwm  = base_speed
            right_pwm = base_speed

        movement.move(
            clamp(left_pwm,  -1, 1),
            clamp(right_pwm, -1, 1)
        )

        # ── Debug display (every 5th frame to save CPU) ───────────────────────
        frame_count += 1
        if frame_count % 5 == 0:
            cv2.putText(im2, f"STATE: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im2, f"MEMORY: turn {turn_memory}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("1. Colour mask", colour_mask)
            cv2.imshow("2. Tracking",    im2)

            if cv2.waitKey(1) == 27:   # ESC to quit
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error: {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()