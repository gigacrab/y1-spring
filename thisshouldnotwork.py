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
# 1. LOAD ORB TEMPLATES (Phase 2)
# ==========================================
png_path = '/home/jaydenbryan/Project/Symbols_png/'
template_files_png = {
    "Danger": "danger.png",
    "Fingerprint": "fingerprint.png",
    "Press Button": "pressbutton.png",
    "Recycle": "recycle.png",
    "QR Code": "qrcode.png"
}

orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params, search_params)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

template_features = {}
for label, filename in template_files_png.items():
    img = cv2.imread(os.path.join(png_path, filename), 0)
    if img is not None:
        img_clahe = clahe.apply(img) 
        kp, des = orb.detectAndCompute(img_clahe, None)
        if des is not None:
            template_features[label] = (kp, des)
    else:
        print(f"Warning: Missing ORB photo {filename}")

# ==========================================
# 2. LOAD HU MOMENTS DNA (Phase 1)
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
# 3. START CAMERA & MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Autonomous Hybrid Brain Ready! Driving and Scanning...")

try:
    while True:
        time_marker = time.perf_counter()
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # MODULE A: LINE FOLLOWING (Executed First for Motor Priority)
        # ==========================================
        roi = frame[240:480, :]
        imgray_line = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh_line = cv2.threshold(imgray_line, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours_line, hierarchy_line = cv2.findContours(thresh_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_line) > 0:
            contour_areas = [cv2.contourArea(cnt) for cnt in contours_line]
            filtered_contours = []
            filtered_contour_areas = []

            for i, cnt_a in enumerate(contour_areas):
                if cnt_a >= 7500 and cnt_a <= 40000:
                    filtered_contours.append(contours_line[i])
                    filtered_contour_areas.append(contour_areas[i])
            
            if len(filtered_contours) > 0 and ret < 180:
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
                    
                    # Draw yellow target line on the main frame for debugging (Offset by 240 for ROI)
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
        # MODULE B: SYMBOL RECOGNITION 
        # ==========================================
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        best_match = None

        # THE DONUT FIX: 255 Block Size prevents hollowing out shapes in dark corners
        thresh_sym = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 4)
        kernel_sym = np.ones((3, 3), np.uint8)
        thresh_sym = cv2.morphologyEx(thresh_sym, cv2.MORPH_CLOSE, kernel_sym)
        
        cnts, hierarchy = cv2.findContours(thresh_sym, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i, c in enumerate(cnts):
                if cv2.contourArea(c) > 1500: 
                    
                    holes = 0
                    for j, child_c in enumerate(cnts):
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

                        if geom_match:
                            x, y, w, h = cv2.boundingRect(c)
                            box_area = w * h
                            extent = area / float(box_area) if box_area > 0 else 0
                            
                            if geom_match == "Star":
                                if solidity > 0.6 or corners < 8:
                                    geom_match = None  
                            elif geom_match == "Octagon":
                                if corners < 5 or solidity < 0.75:
                                    geom_match = None
                            elif geom_match == "Kite":
                                if extent > 0.75:
                                    geom_match = None 

                        if geom_match == "Arrow":
                            # --- THE BOOMERANG SHIELD (Protects the Recycle Sign) ---
                            # A real block arrow has 7 corners and is thick (Solidity > 0.55).
                            # A Recycle arrow is a curved boomerang with low solidity!
                            if not (6 <= corners <= 9) or solidity < 0.55:
                                geom_match = None  
                            else:
                                x, y, w, h = cv2.boundingRect(c)
                                bx = x + (w / 2.0)
                                by = y + (h / 2.0)
                                
                                M = cv2.moments(c)
                                if M["m00"] != 0:
                                    cx = M["m10"] / M["m00"]
                                    cy = M["m01"] / M["m00"]
                                    dx = cx - bx
                                    dy = cy - by
                                    
                                    if abs(dx) > abs(dy):
                                        geom_match = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                                    else:
                                        geom_match = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"

                        if geom_match:
                            best_match = geom_match
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            break 

        # Phase 2 ORB SCANNER
        if best_match is None:
            gray_processed = clahe.apply(gray)
            kp_frame, des_frame = orb.detectAndCompute(gray_processed, None)
            max_good_matches = 0
            
            if des_frame is not None and len(des_frame) >= 2:
                for label, (kp_template, des_template) in template_features.items():
                    if des_template is not None:
                        matches = flann.knnMatch(des_template, des_frame, k=2)
                        
                        good_matches = []
                        for m_n in matches:
                            if len(m_n) == 2:
                                m, n = m_n
                                if m.distance < 0.75 * n.distance:
                                    good_matches.append(m)
                        
                        if label == "Danger": required_matches = 8
                        elif label == "Fingerprint": required_matches = 15
                        else: required_matches = 12
                        
                        if len(good_matches) >= required_matches and len(good_matches) > max_good_matches:
                            max_good_matches = len(good_matches)
                            best_match = label

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        if best_match:
            cv2.putText(frame, f"MATCH: {best_match}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # TODO: Add your UART/Serial broadcast here!
            # Example: serial_port.write(f"{best_match}\n".encode())

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