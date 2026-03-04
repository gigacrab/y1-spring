import cv2
import numpy as np
from picamera2 import Picamera2
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

# ORB + FLANN Setup
orb = cv2.ORB_create(nfeatures=1200)
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
        print(f"Warning: Could not load {filename}. Check spelling!")

# ==========================================
# 2. CAMERA & MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Dedicated ORB Scanner Ready! Waiting for the 5 complex symbols...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        symbol_detected = False
        best_label = None

        # ==========================================
        # TOOL: ORB TEXTURE SCANNER
        # ==========================================
        gray_processed = cv2.equalizeHist(blurred)
        kp_frame, des_frame = orb.detectAndCompute(gray_processed, None)
        
        max_good_matches = 0
        
        if des_frame is not None and len(des_frame) >= 2:
            for label, (kp_template, des_template) in template_features.items():
                if des_template is not None:
                    matches = flann.knnMatch(des_template, des_frame, k=2)
                    
                    # --- CUSTOM TUNING FOR EACH SYMBOL ---
                    if label == "QR Code":
                        ratio_strictness = 0.85 # Relaxed for repetitive squares
                        required_matches = 20   # High count to prevent false positives
                    elif label == "Danger":
                        ratio_strictness = 0.75 # Standard strictness
                        required_matches = 5    # Low count because it's a smooth shape
                    else:
                        ratio_strictness = 0.75 # Standard strictness
                        required_matches = 12   # Standard count
                    # -------------------------------------
                    
                    good_matches = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            # Apply the custom ratio test
                            if m.distance < ratio_strictness * n.distance:
                                good_matches.append(m)
                    
                    # Check if it passes the required matches!
                    if len(good_matches) >= required_matches and len(good_matches) > max_good_matches:
                        max_good_matches = len(good_matches)
                        best_label = label
                        symbol_detected = True

        # --- DRAW THE RESULTS ---
        if symbol_detected:
            print(f"MATCH: {best_label} (Strength: {max_good_matches})")
            cv2.putText(frame, f"MATCH: {best_label}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Robot View", frame)
        
        # --- SAFE QUIT LOGIC ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()