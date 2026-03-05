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

# --- THE CPU FIX: Optimized so the Pi doesn't stutter! ---
orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=20) 
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
print("Hybrid Master Brain Ready! Scanning the whole room...")
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        best_match = None

        # ==========================================
        # PHASE 1: GEOMETRY FIRST (With the "Holes" Gatekeeper)
        # ==========================================
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 10)
        
        # CRITICAL FIX: We changed RETR_EXTERNAL to RETR_TREE so the robot can look INSIDE the shapes!
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None:
            for i, c in enumerate(cnts):
                if cv2.contourArea(c) > 1500:
                    
                    # --- THE "HOLES" FILTER ---
                    # Count how many inner details (children) are inside this main shape
                    holes = 0
                    for j, child_c in enumerate(cnts):
                        # hierarchy[0][j][3] tells us who the "parent" of the contour is
                        if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 200:
                            holes += 1
                    
                    # IF THE SHAPE IS COMPLETELY SOLID (0 HOLES), IT'S A BASIC GEOMETRY SHAPE!
                    if holes == 0:
                        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                        lowest_diff = 0.05 
                        geom_match = None
                        
                        for name, master_dna in templates_npy.items():
                            diff = np.sum(np.abs(live_moments - master_dna))
                            if diff < lowest_diff:
                                lowest_diff = diff
                                geom_match = name
                                
                        if geom_match in ["Plus", "Kite"]:
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                            corners = len(approx)
                            geom_match = "Kite" if corners < 8 else "Plus"
                        
                        if geom_match == "Arrow":
                            peri = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                            if len(approx) > 9: 
                                geom_match = None
                            if geom_match == "Arrow":
                                x, y, w, h = cv2.boundingRect(c)
                                
                                # 1. Create a perfect digital silhouette of the arrow
                                mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
                                
                                # 2. Grab the extreme 15% edges of all four sides
                                margin_x = max(1, int(w * 0.15))
                                margin_y = max(1, int(h * 0.15))
                                
                                top_edge = mask[:margin_y, :]
                                bottom_edge = mask[-margin_y:, :]
                                left_edge = mask[:, :margin_x]
                                right_edge = mask[:, -margin_x:]
                                
                                # 3. Weigh the ink on all four edges
                                # An arrow has 3 sharp points (low mass) and 1 flat tail (high mass)!
                                masses = {
                                    "Arrow (UP)": cv2.countNonZero(top_edge),   # If Top is the heavy tail, it points DOWN
                                    "Arrow (DOWN)": cv2.countNonZero(bottom_edge),  # If Bottom is the heavy tail, it points UP
                                    "Arrow (LEFT)": cv2.countNonZero(left_edge), # If Left is the heavy tail, it points RIGHT
                                    "Arrow (RIGHT)": cv2.countNonZero(right_edge)  # If Right is the heavy tail, it points LEFT
                                }
                                
                                # 4. The arrow points in the opposite direction of the heaviest edge!
                                geom_match = max(masses, key=masses.get)
                        # --------------------------------
                        if geom_match == "Star":
                            # Draw a rotating box that perfectly hugs the shape
                            rect = cv2.minAreaRect(c)
                            w_rect, h_rect = rect[1]
                            
                            if w_rect != 0 and h_rect != 0:
                                # Divide the long side by the short side
                                aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                                
                                # A QR Code block is a perfect square (ratio ~ 1.0)
                                # A Kite is stretched out (ratio usually > 1.3)
                                if aspect_ratio < 1.15:
                                    geom_match = None  # It's a square! Reject it and let ORB scan!
                        # --------------------------------

                        if geom_match:
                            best_match = geom_match
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            break 

        # ==========================================
        # PHASE 2: ORB SCANNER (Catches all complex/hollow symbols)
        # ==========================================
        if best_match is None:
            gray_processed = cv2.equalizeHist(blurred)
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
                        
                        # Danger is strict, Fingerprint is loose
                        if label == "Danger": required_matches = 5
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
        
        # --- SAFE QUIT ---
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1: break

finally:
    picam2.stop()
    cv2.destroyAllWindows()