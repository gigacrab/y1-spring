import cv2
import numpy as np
from picamera2 import Picamera2
import time
import movement
import sys
import os
import collections # <--- NEW: Required for the Consensus Voting!

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================
def clamp(value, min_val, max_val):
    if value > max_val: return max_val
    elif value < min_val: return min_val
    return value

def getSign(n):
    return (n > 0) - (n < 0)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp = float(sys.argv[2])
        ki = float(sys.argv[3])
        kd = float(sys.argv[4])
    else:
        print("No variables provided! Defaulting to Day 1 Config...")
        base_speed = 0.4
        kp = 1.4
        ki = 0.01
        kd = 0.2

# --- LOAD SHAPE DNA ---
base_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Kite": "kite.npy",
    "Octagon": "octagon.npy", 
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
}

templates = {}
try:
    for name, filename in template_files.items():
        templates[name] = np.load(os.path.join(base_path, filename))
except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}. Check the Symbols_npy folder!")
    exit()

# --- START CAMERA ---
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)
print("Consensus Engine Online. Driving and Scanning...")

# PID Variables
error = 0
total_error = 0
last_error = 0
diff_error = 0
first = True
prev_frame_time = 0
last_pid_time = time.perf_counter()

# STATE MACHINE VARIABLES
scanning_mode = False
scan_frames = 0
scan_results = []
cooldown_until = 0

try: 
    while True: 
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # ==========================================
        # MODULE A: LIZARD BRAIN (LINE FOLLOWER)
        # ==========================================
        if not scanning_mode:
            roi_bottom = frame[240:480, :]
            imgray_line = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
            imgray_line = cv2.GaussianBlur(imgray_line, (5,5), 0)
            
            ret, thresh_line = cv2.threshold(imgray_line, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours_line, _ = cv2.findContours(thresh_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            im2 = np.zeros((240, 640, 3), dtype=np.uint8)
            
            if len(contours_line) > 0:
                contour_areas = [cv2.contourArea(cnt) for cnt in contours_line]
                filtered_contours = []
                filtered_contour_areas = []

                for i, cnt_a in enumerate(contour_areas):
                    if cnt_a >= 8500 and cnt_a <= 40000:
                        filtered_contours.append(contours_line[i])
                        filtered_contour_areas.append(contour_areas[i])
                
                if len(filtered_contours) > 0 and ret < 180:
                    if len(filtered_contours) > 1:
                        zipped_pairs = zip(filtered_contour_areas, filtered_contours)
                        sorted_pairs = sorted(zipped_pairs, reverse=True)
                        _, sorted_contours = zip(*sorted_pairs)
                        line_contour = sorted_contours[0]
                        cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
                    else:
                        line_contour = filtered_contours[0]
                    
                    cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

                    M = cv2.moments(line_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                        current_time = time.perf_counter()
                        elapsed_time = current_time - last_pid_time
                        if elapsed_time <= 0: elapsed_time = 0.0001
                        last_pid_time = current_time 
                        
                        error = (320 - cx) / 320    
                        total_error += error * elapsed_time

                        if not first: diff_error = (error - last_error) / elapsed_time
                        else: first = False
                            
                        pid = kp * error + ki * total_error + kd * diff_error
                        last_error = error
                    else:
                        pid = getSign(last_error)
                else:
                    pid = getSign(last_error)
            else:
                pid = getSign(last_error)

            left_pwm = base_speed + pid
            right_pwm = base_speed - pid

            clamped_left_pwm = clamp(left_pwm, -1, 1)
            clamped_right_pwm = clamp(right_pwm, -1, 1)

            movement.move(clamped_left_pwm, clamped_right_pwm)
        else:
            # WE ARE IN SCANNING MODE. HOLD BRAKES!
            movement.move(0, 0)
            im2 = np.zeros((240, 640, 3), dtype=np.uint8)

        # ==========================================
        # MODULE B: FRONTAL LOBE (SHAPE SCANNER)
        # ==========================================
        # Only look for shapes if we are NOT in cooldown!
        if time.perf_counter() > cooldown_until:
            gray_shape = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # FULL FRAME to prevent chopping!
            blurred_shape = cv2.GaussianBlur(gray_shape, (5, 5), 0)
            
            thresh_sym = cv2.adaptiveThreshold(blurred_shape, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
            kernel = np.ones((3, 3), np.uint8)
            thresh_sym = cv2.morphologyEx(thresh_sym, cv2.MORPH_CLOSE, kernel)
            
            cnts_shape, hierarchy = cv2.findContours(thresh_sym, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            best_match = None
            
            if hierarchy is not None:
                for i, c in enumerate(cnts_shape):
                    if cv2.contourArea(c) > 1500:
                        
                        holes = 0
                        for j, child_c in enumerate(cnts_shape):
                            if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 500: holes += 1
                                
                        if holes == 0:
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                            corners = len(approx)
                            
                            hull = cv2.convexHull(c)
                            hull_area = cv2.contourArea(hull)
                            area = cv2.contourArea(c)
                            solidity = area / float(hull_area) if hull_area > 0 else 0
                            
                            live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                            lowest_diff = 0.10
                            
                            for name, master_dna in templates.items():
                                diff = np.sum(np.abs(live_moments - master_dna))
                                if diff < lowest_diff:
                                    lowest_diff = diff
                                    best_match = name
                                    
                            if best_match in ["Plus", "Kite"]:
                                best_match = "Kite" if corners < 8 else "Plus"

                            if best_match:
                                x, y, w, h = cv2.boundingRect(c)
                                box_area = w * h
                                extent = area / float(box_area) if box_area > 0 else 0
                                
                                if best_match == "Star" and (solidity > 0.6 or corners < 8): best_match = None  
                                elif best_match == "Octagon" and (corners < 5 or solidity < 0.75): best_match = None
                                elif best_match == "Kite" and (extent > 0.75 or corners > 5): best_match = None 
                                elif best_match == "3/4 Circle" and solidity > 0.95: best_match = None

                            if best_match == "Arrow":
                            if not (6 <= corners <= 9) or solidity < 0.55:
                                best_match = None  
                            else:
                                bx, by = x + (w / 2.0), y + (h / 2.0)
                                M_shape = cv2.moments(c)
                                if M_shape["m00"] != 0:
                                    cx_s = M_shape["m10"] / M_shape["m00"]
                                    cy_s = M_shape["m01"] / M_shape["m00"]
                                    dx, dy = cx_s - bx, cy_s - by
                                    if abs(dx) > abs(dy): best_match = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                                    else: best_match = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"

                        # ==========================================
                        # THE BOX GATEKEEPER & PROXIMITY SENSOR
                        # ==========================================
                        if best_match:
                            x, y, w, h = cv2.boundingRect(c)
                            shape_cx = x + (w / 2.0)
                            shape_cy = y + (h / 2.0)
                            shape_footprint = w * h
                            
                            is_in_close_box = False
                            
                            # Scan all contours again to find the Box enclosing our shape
                            for b_cnt in cnts_shape:
                                bx, by, bw, bh = cv2.boundingRect(b_cnt)
                                box_footprint = bw * bh
                                
                                # 1. Is this contour physically larger than the shape?
                                if box_footprint > (shape_footprint * 1.5):
                                    
                                    # 2. Is the shape completely inside this contour?
                                    if bx < shape_cx < (bx + bw) and by < shape_cy < (by + bh):
                                        
                                        # 3. Is it a 4-cornered Box?
                                        b_peri = cv2.arcLength(b_cnt, True)
                                        b_approx = cv2.approxPolyDP(b_cnt, 0.04 * b_peri, True)
                                        
                                        if len(b_approx) == 4:
                                            # 4. PROXIMITY SENSOR: Is the box close enough to stop?
                                            # ---> TUNE THIS NUMBER: Higher = robot drives closer before stopping! <---
                                            if box_footprint > 25000: 
                                                is_in_close_box = True
                                                
                                                # Draw a BLUE box around the container to prove it locked on
                                                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 0), 3) 
                                                break
                                                
                            if not is_in_close_box:
                                # Shape is either not in a box, or the box is too far away!
                                # Throw the match in the trash and keep driving.
                                best_match = None 
                                
                            if best_match:
                                # If it survived the gatekeeper, draw the Green shape box and trigger the brakes!
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                break

            # --- THE CONSENSUS STATE MACHINE ---
            if not scanning_mode and best_match:
                # 1. First blurry glimpse detected! Hit brakes and start scanning.
                print(f"Motion trigger: {best_match}. Initiating 50-frame consensus scan...")
                scanning_mode = True
                scan_frames = 0
                scan_results = []
                movement.move(0, 0)

            elif scanning_mode:
                # 2. We are currently stopped and gathering votes!
                scan_frames += 1
                if best_match:
                    scan_results.append(best_match)
                    cv2.putText(frame, best_match, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Visual Indicator of scanning progress
                cv2.putText(frame, f"GATHERING VOTES: {scan_frames}/50", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if scan_frames >= 50:
                    # 3. 50 frames collected. Calculate the winner!
                    if len(scan_results) > 0:
                        counter = collections.Counter(scan_results)
                        final_answer = counter.most_common(1)[0][0]
                        confidence = counter.most_common(1)[0][1]
                        
                        print(f"==== FINAL DECISION: {final_answer} ({confidence}/50 votes) ====")
                        # TODO: Transmit 'final_answer' over UART/Serial to ESP32 here!

                    else:
                        print("==== FALSE ALARM: Shape lost during scan ====")
                        
                    # ==========================================
                    # THE COOLDOWN PROTOCOL (Safe PID Resume)
                    # ==========================================
                    print("Resuming PID Line Follower. Blinding shape scanner for 3 seconds...")
                    scanning_mode = False
                    cooldown_until = time.perf_counter() + 3.0 # Ignores shapes for 3 seconds!
                    first = True
                    last_pid_time = time.perf_counter()

        else:
            # We are driving away from a shape. Draw cooldown status.
            cv2.putText(frame, "COOLDOWN (IGNORING SHAPES)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # ==========================================
        # DISPLAY & FPS
        # ==========================================
        new_frame_time = time.perf_counter()
        time_diff = new_frame_time - prev_frame_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_frame_time = new_frame_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Robot View", frame)
        cv2.imshow("Line Follower Data", im2)
        
        if cv2.waitKey(1) == 27:
            movement.move(0, 0)
            break
            
except (KeyboardInterrupt, Exception) as e:
    print(f"Error has occurred - {e}")

finally:
    movement.move(0, 0)
    movement.pi.stop()
    picam2.stop()
    cv2.destroyAllWindows()