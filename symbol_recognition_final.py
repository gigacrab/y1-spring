import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================================
# 1. LOAD ORB TEMPLATES (Phase 1: Art)
# ==========================================
png_path = '/home/jaydenbryan/Project/Symbols_png/'
template_files_png = {
    "Danger": "danger.png",
    "Fingerprint": "fingerprint.png",
    "Press Button": "pressbutton.png",
    "Recycle": "recycle.png",
    "QR Code": "qrcode.png"
}

# The hyper-sensitive ORB Brain
orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

template_features = {}
for label, filename in template_files_png.items():
    img = cv2.imread(os.path.join(png_path, filename), 0)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            template_features[label] = (kp, des)
    else:
        print(f"Warning: Missing ORB photo {filename}")

# ==========================================
# 2. LOAD HU MOMENTS DNA (Phase 2: Geometry)
# ==========================================
npy_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files_npy = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Kite": "kite.npy",
    "Octagon": "octagon.npy", 
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
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
print("Hybrid Master Brain Ready! Scanning for all 13 symbols...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ==========================================
        # THE STATIC SCANNER BOX (300x300 in the center)
        # ==========================================
        x1, y1 = 170, 90
        x2, y2 = 470, 390
        
        # Crop everything down to just the scanner box!
        roi_gray = gray[y1:y2, x1:x2]
        roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        best_match = None

        # ==========================================
        # METHOD 1: ORB SCANNER (Inside the box)
        # ==========================================
        gray_processed = cv2.equalizeHist(roi_blurred)
        kp_roi, des_roi = orb.detectAndCompute(gray_processed, None)
        max_good_matches = 0
        
        if des_roi is not None and len(des_roi) >= 2:
            for label, (kp_template, des_template) in template_features.items():
                if des_template is not None:
                    matches = flann.knnMatch(des_template, des_roi, k=2)
                    
                    good_matches = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                    
                    # Danger is tight (8), others are looser (10)
                    required_matches = 8 if label == "Danger" else 10
                    
                    if len(good_matches) >= required_matches and len(good_matches) > max_good_matches:
                        max_good_matches = len(good_matches)
                        best_match = label 

        # ==========================================
        # METHOD 2: GEOMETRY SCANNER (If ORB found nothing)
        # ==========================================
        if best_match is None:
            # We use the crisp, un-glued image so your Hu Moments stay perfect!
            thresh = cv2.adaptiveThreshold(roi_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 10)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                # Lowered the area requirement slightly since the box is smaller
                if cv2.contourArea(c) > 800: 
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 0.05 
                    geom_match = None
                    
                    for name, master_dna in templates_npy.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            geom_match = name
                            
                    # Your Kite vs Plus Tie-Breaker
                    if geom_match in ["Plus", "Kite"]:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        corners = len(approx)
                        geom_match = "Kite" if corners < 8 else "Plus"
                        
                    if geom_match:
                        best_match = geom_match
                        
                        # Draw a mini box tightly around the shape inside the big box
                        shape_x, shape_y, shape_w, shape_h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x1+shape_x, y1+shape_y), (x1+shape_x+shape_w, y1+shape_y+shape_h), (0, 255, 255), 2)
                        break 

        # ==========================================
        # DISPLAY HUD AND RESULTS
        # ==========================================
        # Draw the permanent Scanner Box on the screen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "SCAN ZONE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if best_match:
            cv2.putText(frame, f"MATCH: {best_match}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Robot View", frame)
        
        # --- SAFE QUIT ---
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1: break

finally:
    picam2.stop()
    cv2.destroyAllWindows()