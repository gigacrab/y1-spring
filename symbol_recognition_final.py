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
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_processed = cv2.equalizeHist(blurred)
        
        best_match = None

        # ==========================================
        # METHOD 1: LIGHTWEIGHT GEOMETRY (Find the shape first!)
        # ==========================================
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 10)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) > 1500:
                # We found a big shape! Get its coordinates.
                x, y, w, h = cv2.boundingRect(c)
                
                # Check Hu Moments DNA
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                lowest_diff = 0.05 
                
                for name, master_dna in templates_npy.items():
                    diff = np.sum(np.abs(live_moments - master_dna))
                    if diff < lowest_diff:
                        lowest_diff = diff
                        best_match = name
                        
                # Kite vs Plus Tie-Breaker
                if best_match in ["Plus", "Kite"]:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    corners = len(approx)
                    best_match = "Kite" if corners < 8 else "Plus"

                # ==========================================
                # METHOD 2: HEAVY ORB OVERRIDE (Inside the box only)
                # ==========================================
                # Crop the camera feed down to just the shape to save the Pi's CPU!
                roi_image = gray_processed[max(0, y-10):min(frame.shape[0], y+h+10), 
                                           max(0, x-10):min(frame.shape[1], x+w+10)]
                
                kp_roi, des_roi = orb.detectAndCompute(roi_image, None)
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
                            
                            required_matches = 6 if label == "Danger" else 12
                            
                            # IF ORB SEES ART, IT OVERWRITES THE GEOMETRY GUESS!
                            if len(good_matches) >= required_matches and len(good_matches) > max_good_matches:
                                max_good_matches = len(good_matches)
                                best_match = label 
                
                # Draw the box and stop looking at background noise!
                if best_match:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break 

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
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