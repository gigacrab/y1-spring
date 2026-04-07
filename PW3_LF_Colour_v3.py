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
        raise Exception("Didn't input appropriate variables. Need: base_speed kp ki kd")

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

# ── State machine constants & variables ───────────────────────────────────────
STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"       
STATE_TURN_90      = "TURN_90"      

state      = STATE_FOLLOW_BLACK
last_state = STATE_FOLLOW_BLACK     

# ── Feature Settings ──────────────────────────────────────────────────────────
black_line_side = "right"  
SEARCH_SPEED    = 0.35     

TURN_90_SPEED   = 0.65     
TURN_90_LOCKOUT = 0.5      
turn_90_start   = 0
turn_90_dir     = "right"

blindfold_until = 0  # NEW: Timer to keep the blindfold on through the whole turn
frame_count = 0

# ── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    try:
        current_time = time.perf_counter()
        time_marker  = current_time

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi   = frame[240:480, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ── Masks ─────────────────────────────────────────────────────────────
        red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
        colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

        imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ✨ THE BLINDFOLD TIMER FIX ✨
        # Keeps the blindfold on during SEARCH *and* for 1.5 seconds after color ends
        if state == STATE_SEARCH or current_time < blindfold_until:
            if black_line_side == "left":
                thresh[:, :320] = 0  # Black out left, keep right
            elif black_line_side == "right":
                thresh[:, 320:] = 0  # Black out right, keep left

        # ── Contours & Centroids ──────────────────────────────────────────────
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

        # ── Geometry & Memory Updates ─────────────────────────────────────────
        if state == STATE_FOLLOW_BLACK and valid_color_cnt is not None:
            left_black_px  = cv2.countNonZero(thresh[:, :320])
            right_black_px = cv2.countNonZero(thresh[:, 320:])
            
            if right_black_px > left_black_px:
                black_line_side = "left"
            else:
                black_line_side = "right"

        color_is_horizontal = False
        if valid_color_cnt is not None:
            x, y, w, h = cv2.boundingRect(valid_color_cnt)
            if h > 0 and w > (h * 2.5) and w > 150:
                color_is_horizontal = True

        # ── State Machine Transitions ─────────────────────────────────────────
        if state == STATE_TURN_90:
            elapsed_turn = current_time - turn_90_start
            if elapsed_turn > TURN_90_LOCKOUT:
                if valid_color_cnt is not None and not color_is_horizontal:
                    state = STATE_FOLLOW_COLOR
                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK

        elif state == STATE_SEARCH:
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

        else: 
            if color_is_horizontal:
                left_px  = cv2.countNonZero(colour_mask[:, :320])
                right_px = cv2.countNonZero(colour_mask[:, 320:])
                turn_90_dir   = "left" if left_px > right_px else "right"
                turn_90_start = current_time
                state = STATE_TURN_90

            elif valid_color_cnt is not None:
                state = STATE_FOLLOW_COLOR

            elif state == STATE_FOLLOW_COLOR:
                # NEW: Color line ended! Trigger search and lock the blindfold for 1.5s
                state = STATE_SEARCH
                blindfold_until = current_time + 1.5 
                print(f"Color line ended. Memory: {black_line_side}. Blindfold locked for 1.5s!")

            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

        if state != last_state:
            total_error = 0
            last_error = 0
            first = True
            last_state = state

        # ── Motor Control ─────────────────────────────────────────────────────
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
            line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
            cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

            if line_contour is not None and cx is not None:
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = max(current_time - time_marker, 0.0001)

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
                pid       = getSign(last_error) * 2
                left_pwm  = base_speed + pid
                right_pwm = base_speed - pid

        elif state == STATE_SEARCH:
            # Sweeping turn (Adjust negative sign if physical spin is backwards)
            turn_pwm  = -SEARCH_SPEED if black_line_side == "left" else SEARCH_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        elif state == STATE_TURN_90:
            turn_pwm  = -TURN_90_SPEED if turn_90_dir == "left" else TURN_90_SPEED
            left_pwm  = base_speed + turn_pwm
            right_pwm = base_speed - turn_pwm

        else:
            left_pwm  = base_speed
            right_pwm = base_speed

        clamped_left_pwm  = clamp(left_pwm,  -1, 1)
        clamped_right_pwm = clamp(right_pwm, -1, 1)
        movement.move(clamped_left_pwm, clamped_right_pwm)

        # ── Debug Display ─────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 5 == 0: 
            if valid_color_cnt is not None:
                x, y, w, h = cv2.boundingRect(valid_color_cnt)
                cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # Displays if the blindfold is currently active
            blind_status = "ON" if current_time < blindfold_until else "OFF"
            
            cv2.putText(im2, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(im2, f"MEM: {black_line_side} | BLIND: {blind_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("1. Color Mask", colour_mask)
            cv2.imshow("2. Tracking & Math", im2)
            
            if cv2.waitKey(1) == 27: 
                break

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred – {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()