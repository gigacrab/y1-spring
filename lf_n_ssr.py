import cv2
import numpy as np
from picamera2 import Picamera2
import time
import movement
import sys
import os
import collections

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
                            # THE PROXIMITY SENSOR (SIMPLIFIED)
                            # ==========================================
                            if best_match:
                                x, y, w, h = cv2.boundingRect(c)
                                shape_footprint = w * h
                                
                                # ---> TUNE THIS NUMBER <---
                                # If the shape itself is smaller than 8000 pixels, it is too far away.
                                if shape_footprint < 15000:
                                    best_match = None 
                                
                                if best_match:
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    break 

            # --- THE CONSENSUS STATE MACHINE ---
            if not scanning_mode and best_match:
                print(f"Motion trigger: {best_match}. Initiating Active Braking!")
                
                # 1. ACTIVE BRAKING: Throw it in reverse to cancel momentum!
                # Tweak the speed (-0.3) and the sleep time (0.15) to make it reverse more or less.
                movement.move(-0.3, -0.3) 
                time.sleep(0.2) 
                
                # 2. Full stop.
                movement.move(0, 0)
                
                # 3. Wait a microsecond for the camera chassis to stop wobbling from the brake
                time.sleep(0.1) 
                
                print("Camera settled. Starting 50-frame consensus scan...")
                scanning_mode = True
                scan_frames = 0
                scan_results = []

            elif scanning_mode:
                scan_frames += 1
                if best_match:
                    scan_results.append(best_match)
                    cv2.putText(frame, best_match, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, f"GATHERING VOTES: {scan_frames}/50", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if scan_frames >= 50:
                    if len(scan_results) > 0:
                        counter = collections.Counter(scan_results)
                        final_answer = counter.most_common(1)[0][0]
                        confidence = counter.most_common(1)[0][1]
                        
                        print(f"==== FINAL DECISION: {final_answer} ({confidence}/50 votes) ====")
                        # TODO: Transmit 'final_answer' over UART/Serial to ESP32 here!

                    else:
                        print("==== FALSE ALARM: Shape lost during scan ====")
                        
                    print("Resuming PID Line Follower. Blinding shape scanner for 3 seconds...")
                    scanning_mode = False
                    cooldown_until = time.perf_counter() + 3.0 
                    first = True
                    last_pid_time = time.perf_counter()

        else:
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