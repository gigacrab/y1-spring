import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import sys
import movement

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================================
# 0. PID MOTOR CONFIGURATION
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
        print("Warning: Didn't input appropriate variables. Using defaults.")
        base_speed, kp, ki, kd = 0.4, 1.4, 0.01, 0.2

error = 0
total_error = 0
last_error = 0
diff_error = 0
first = True

# ==========================================
# 1. LOAD HU MOMENTS DNA (SHAPES ONLY)
# ==========================================
npy_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files_npy = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy",
    "Octagon": "octagon.npy",
    "Kite": "kite.npy"
}

templates_npy = {}
for name, filename in template_files_npy.items():
    try:
        templates_npy[name] = np.load(os.path.join(npy_path, filename))
    except FileNotFoundError:
        print(f"Warning: Missing DNA file {filename}")

# ==========================================
# 2. START CAMERA & MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Lean Split-Brain Ready! Driving and Scanning Shapes...")

prev_frame_time = 0

try:
    while True:
        time_marker = time.perf_counter()
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # ==========================================
        # BRAIN 1: LIZARD BRAIN (LINE FOLLOWER)
        # Uses Fast Otsu Thresholding on the bottom ROI
        # ==========================================
        roi = frame[240:480, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh_line = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        cnts_line, _ = cv2.findContours(thresh_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts_line) > 0:
            filtered_contours = []
            filtered_contour_areas = []

            for c in cnts_line:
                c_area = cv2.contourArea(c)
                if 11000 <= c_area <= 40000:
                    # Your brilliant smoothness filter
                    c_arc = cv2.arcLength(c, True)
                    epsilon = 0.2 * c_arc
                    c_approx = cv2.approxPolyDP(c, epsilon, True)
                    c_approx_arc = cv2.arcLength(c_approx, True)
                    
                    if c_approx_arc != 0: 
                        smoothness = c_arc / c_approx_arc 
                        if smoothness > 0.97:
                            filtered_contours.append(c)
                            filtered_contour_areas.append(c_area)
            
            if len(filtered_contours) > 0:
                if len(filtered_contours) > 1:
                    zipped_pairs = zip(filtered_contour_areas, filtered_contours)
                    sorted_pairs = sorted(zipped_pairs, reverse=True)
                    _, sorted_contours = zip(*sorted_pairs)
                    line_contour = sorted_contours[0]
                else:
                    line_contour = filtered_contours[0]
                
                M = cv2.moments(line_contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    
                    # Target line drawing (Offset by 240 for ROI visualization)
                    cv2.line(frame, (cx, 240), (cx, 480), (0, 255, 255), 3)

                    elapsed_time = time.perf_counter() - time_marker
                    if elapsed_time <= 0: elapsed_time = 0.0001
                    
                    error = (320 - cx) / 320    
                    total_error += error * elapsed_time

                    if not first:
                        diff_error = (error - last_error) / elapsed_time
                    else:
                        first = False
                        
                    pid = kp * error + ki * total_error + kd * diff_error
                    last_error = error
            else:
                pid = getSign(last_error) * 2
        else:
            pid = getSign(last_error) * 2

        left_pwm = base_speed + pid
        right_pwm = base_speed - pid

        clamped_left_pwm = clamp(left_pwm, -1, 1)
        clamped_right_pwm = clamp(right_pwm, -1, 1)

        # FIRE MOTORS NOW
        movement.move(clamped_left_pwm, clamped_right_pwm)

        # ==========================================
        # BRAIN 2: FRONTAL LOBE (SHAPE SCANNER)
        # Uses Advanced Adaptive Thresholding on the full frame
        # ==========================================
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_full, (5, 5), 0)
        
        # Donut-Hole Fix (Block Size 255) to protect against glare
        thresh_sym = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh_sym = cv2.morphologyEx(thresh_sym, cv2.MORPH_CLOSE, kernel)
        
        cnts_shape, hierarchy = cv2.findContours(thresh_sym, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_match = None
        
        if hierarchy is not None:
            for i, c in enumerate(cnts_shape):
                if cv2.contourArea(c) > 1500: 
                    
                    # Holes Gatekeeper (Protects against noise)
                    holes = 0
                    for j, child_c in enumerate(cnts_shape):
                        if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 500:
                            holes += 1
                            
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
                        geom_match = None
                        
                        for name, master_dna in templates_npy.items():
                            diff = np.sum(np.abs(live_moments - master_dna))
                            if diff < lowest_diff:
                                lowest_diff = diff
                                geom_match = name
                                
                        if geom_match in ["Plus", "Kite"]:
                            geom_match = "Kite" if corners < 8 else "Plus"

                        # FACT CHECKERS & ARROW PHYSICS
                        if geom_match:
                            if geom_match == "Star" and (solidity > 0.6 or corners < 8):
                                geom_match = None  
                            elif geom_match == "Octagon" and (corners < 5 or solidity < 0.75):
                                geom_match = None
                            elif geom_match == "3/4 Circle" and solidity > 0.95:
                                geom_match = None
                                
                        if geom_match == "Arrow":
                            if corners > 10: 
                                geom_match = None  
                            else:
                                x, y, w, h = cv2.boundingRect(c)
                                bx = x + (w / 2.0)
                                by = y + (h / 2.0)
                                
                                M = cv2.moments(c)
                                if M["m00"] != 0:
                                    cx_shape = M["m10"] / M["m00"]
                                    cy_shape = M["m01"] / M["m00"]
                                    dx = cx_shape - bx
                                    dy = cy_shape - by
                                    
                                    if abs(dx) > abs(dy):
                                        geom_match = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                                    else:
                                        geom_match = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"

                        if geom_match:
                            best_match = geom_match
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            break 

        # ==========================================
        # DISPLAY RESULTS & FPS
        # ==========================================
        if best_match:
            cv2.putText(frame, f"MATCH: {best_match}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        new_frame_time = time.perf_counter()
        time_diff = new_frame_time - prev_frame_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_frame_time = new_frame_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Robot View", frame)
        
        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to stop
            break

except (KeyboardInterrupt, Exception) as e:
    print(f"Error has occurred - {e}")

finally:
    movement.move(0, 0)
    movement.pi.stop()
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()