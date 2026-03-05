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
    "QR Code": "qrcode.png",
    "Octagon": "octagon.png",  # ADDED TO PHASE 2!
    "Kite": "kite.png"         # ADDED TO PHASE 2!
}

# --- RESTORED ORB SETTINGS (Full power!) ---
orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Create the exact same Anti-Glare filter used in the live video
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

template_features = {}
for label, filename in template_files_png.items():
    img = cv2.imread(os.path.join(png_path, filename), 0)
    if img is not None:
        # MAGIC FIX: Apply the filter to the PNGs so they perfectly match the live camera!
        img_clahe = clahe.apply(img) 
        kp, des = orb.detectAndCompute(img_clahe, None)
        
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
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
    # OCTAGON AND KITE HAVE BEEN DELETED FROM HERE!
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
print("Hybrid Master Brain Ready! Scanning the whole room...")
try:
    while True:
        frame = picam2.capture_array()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Only keep this if colors look backwards on your screen!
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        best_match = None

        # ==========================================
        # PHASE 1: GEOMETRY FIRST
        # ==========================================
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 10)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i, c in enumerate(cnts):
                # UPGRADED TO 4000: This blinds the robot to the tiny nested QR squares!
                if cv2.contourArea(c) > 4000:
                    
                    holes = 0
                    for j, child_c in enumerate(cnts):
                        if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 500:
                            holes += 1
                            
                    # IF IT IS SOLID INK (0 HOLES)...
                    if holes == 0:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        corners = len(approx)
                        
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        area = cv2.contourArea(c)
                        solidity = area / float(hull_area) if hull_area > 0 else 0
                        
                        # RUN DNA MATH (Octagon and Kite are gone! Only solid black shapes remain)
                        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                        lowest_diff = 0.05 
                        geom_match = None
                        
                        for name, master_dna in templates_npy.items():
                            diff = np.sum(np.abs(live_moments - master_dna))
                            if diff < lowest_diff:
                                lowest_diff = diff
                                geom_match = name

                        # THE FACT-CHECKERS
                        if geom_match:
                            
                            # --- The QR Block vs Star Defense ---
                            if geom_match == "Star":
                                # Stars are spiky. If solidity is high, it's a blurry square!
                                if solidity > 0.6 or corners < 8:
                                    geom_match = None  

                        # THE ARROW DIRECTION FINDER
                        if geom_match == "Arrow":
                            if corners > 10: 
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

                        # DRAW THE BOX
                        if geom_match:
                            best_match = geom_match
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            break

        # ==========================================
        # PHASE 2: ORB SCANNER (Complex Symbols)
        # ==========================================
        if best_match is None:
            # --- THE ANTI-GLARE UPGRADE ---
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_processed = clahe.apply(blurred)
            
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
                        
                        # Relaxed Danger threshold!
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

        cv2.imshow("Robot View", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1: break

finally:
    picam2.stop()
    cv2.destroyAllWindows()