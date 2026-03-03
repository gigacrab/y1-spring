import cv2
import numpy as np
from picamera2 import Picamera2
import math
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================================
# 1. LOAD ORB TEMPLATES (For Art/Textures)
# ==========================================
img_path = '/home/jaydenbryan/Project/Symbols_png/'
template_files = {
    "Danger": "symbol_danger.png",
    "Fingerprint": "symbol_fingerprint.png",
    "Press Button": "symbol_pressbutton.png",
    "Recycle": "symbol_recycle.png",
    "QR Code": "symbol_qrcode.png"
}

# Senior's ORB + FLANN Setup
orb = cv2.ORB_create(nfeatures=1000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

template_features = {}
for label, filename in template_files.items():
    img = cv2.imread(os.path.join(img_path, filename), 0)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            template_features[label] = (kp, des)
    else:
        print(f"Warning: Could not load {filename}")

# ==========================================
# 2. SHAPE DETECTOR (For Geometry)
# ==========================================
def get_shape_name(contour):
    perimeter = cv2.arcLength(contour, True)
    # The multiplier determines how strict the corner counting is
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    num_sides = len(approx)
    area = cv2.contourArea(contour)
    
    if perimeter == 0: 
        return "Unknown"
        
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    
    # Recognize your specific geometric symbols based on corner counts
    if num_sides == 4:
        return "Kite / Trapezium"
    elif num_sides == 7:
        return "Arrow"
    elif num_sides == 8:
        return "Octagon"
    elif num_sides == 10:
        return "Star"
    elif num_sides == 12:
        return "Plus"
    elif circularity > 0.75 and area > 300:
        return "Circle"
    elif 0.5 < circularity <= 0.75:
        return "Major Segment"
        
    return "Unknown"

# ==========================================
# 3. CAMERA & MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Senior's Vision Logic Ready! Waiting for symbols...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        symbol_detected = False
        
        # --- TOOL A: ORB TEXTURE SCANNER ---
        gray_processed = cv2.equalizeHist(blurred)
        kp_frame, des_frame = orb.detectAndCompute(gray_processed, None)
        
        best_label = None
        max_good_matches = 0
        
        if des_frame is not None and len(des_frame) > 1:
            for label, (kp_template, des_template) in template_features.items():
                if des_template is not None:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    # Lowe's ratio test to filter out bad matches
                    good_matches = [m for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
                    
                    if len(good_matches) > 10 and len(good_matches) > max_good_matches:
                        max_good_matches = len(good_matches)
                        best_label = label
                        symbol_detected = True

        if symbol_detected:
            cv2.putText(frame, f"ORB: {best_label}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        # --- TOOL B: CANNY EDGE CORNER COUNTER ---
        else:
            # If ORB didn't find art, look for sharp geometric edges
            edges = cv2.Canny(blurred, 50, 150)
            shape_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort the contours so it only analyzes the biggest shapes in the room
            shape_contours = sorted(shape_contours, key=cv2.contourArea, reverse=True)[:5]
            
            for contour in shape_contours:
                # Ignore background noise and tiny dots
                if cv2.contourArea(contour) > 1500:
                    shape_name = get_shape_name(contour)
                    
                    if shape_name != "Unknown":
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                        cv2.putText(frame, shape_name, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        break # Once it finds the biggest valid shape, stop looking!

        # --- LIVE DISPLAY ---
        cv2.imshow("Robot View", frame)
        
        # NOTE: Uncomment the line below to see what the Canny Edge detector is seeing!
        # cv2.imshow("Canny Edges", edges) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()