import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================================
# 1. TIER 1: TEXTURES (ORB Scanner)
# ==========================================
img_path = '/home/jaydenbryan/Project/Symbols_img/'
orb_files = {
    "Danger": "symbol_danger.png", 
    "Fingerprint": "symbol_fingerprint.png"
}

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb_templates = {}

for name, f in orb_files.items():
    img = cv2.imread(os.path.join(img_path, f), 0)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            orb_templates[name] = des

# ==========================================
# 2. TIER 2 & 3: GEOMETRY (Hu Moments DNA)
# ==========================================
npy_path = '/home/jaydenbryan/Project/Symbols_npy/'

# TIER 2: Solid Shapes (Needs Sharp Edges)
sharp_files = {
    "Arrow": "arrow.npy", "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy", "Kite": "kite.npy",
    "Octagon": "octagon.npy", "Plus": "plus.npy",
    "Star": "star.npy", "Trapezium": "trapezium.npy"
}

# TIER 3: Fragmented Shapes (Needs the "Glue")
glued_files = {
    "Press Button": "pressbutton.npy", 
    "Recycle": "recycle.npy",
    "QR Code": "qrcode.npy" 
}

hu_sharp_templates = {}
for name, f in sharp_files.items():
    try: hu_sharp_templates[name] = np.load(os.path.join(npy_path, f))
    except: pass

hu_glued_templates = {}
for name, f in glued_files.items():
    try: hu_glued_templates[name] = np.load(os.path.join(npy_path, f))
    except: pass

# --- START CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Three-Tier System Ready!")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # FIX 1: Raised block size to 151 to stop "hollowing out" the Star and Plus!
        thresh_sharp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 151, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh_glued = cv2.morphologyEx(thresh_sharp, cv2.MORPH_CLOSE, kernel)

        detected_label = None
        box_coords = None

        # --- PROCESS 1: CHECK THE SHARP IMAGE ---
        cnts_sharp, _ = cv2.findContours(thresh_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_sharp:
            if detected_label: break
            area = cv2.contourArea(c)
            
            if 1000 < area < 60000:
                x, y, w, h = cv2.boundingRect(c)
                if 0.3 <= (w/h) <= 3.0: 
                    
                    # Tool 1: Check ORB first (Danger, Fingerprint)
                    roi = gray[y:y+h, x:x+w]
                    kp_live, des_live = orb.detectAndCompute(roi, None)
                    
                    if des_live is not None:
                        max_matches = 0
                        best_orb = None
                        for name, master_des in orb_templates.items():
                            matches = bf.match(master_des, des_live)
                            if len(matches) > max_matches:
                                max_matches = len(matches)
                                best_orb = name
                                
                        if max_matches > 12:
                            detected_label = f"{best_orb} (ORB: {max_matches})"
                            box_coords = (x, y, w, h)
                            break

                    # Tool 2: Check Sharp Hu Moments (Star, Arrow, Plus, etc.)
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 4.0
                    for name, master_dna in hu_sharp_templates.items():
                        live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                        master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                        diff = np.sum(np.abs(live_log - master_log))
                        
                        if diff < lowest_diff:
                            lowest_diff = diff
                            detected_label = f"{name} (Hu: {lowest_diff:.2f})"
                            box_coords = (x, y, w, h)

        # --- PROCESS 2: CHECK THE GLUED IMAGE ---
        if not detected_label:
            cnts_glued, _ = cv2.findContours(thresh_glued, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_glued:
                if detected_label: break
                area = cv2.contourArea(c)
                
                if 1000 < area < 60000:
                    x, y, w, h = cv2.boundingRect(c)
                    if 0.3 <= (w/h) <= 3.0: 
                        
                        # Tool 3: Check Glued Hu Moments (QR Code, Recycle, Press Button)
                        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                        lowest_diff = 4.0
                        for name, master_dna in hu_glued_templates.items():
                            live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                            master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                            diff = np.sum(np.abs(live_log - master_log))
                            
                            if diff < lowest_diff:
                                lowest_diff = diff
                                detected_label = f"{name} (Hu: {lowest_diff:.2f})"
                                box_coords = (x, y, w, h)

        # --- DRAW FINAL OUTPUT ---
        if detected_label and box_coords:
            x, y, w, h = box_coords
            print(f">>> {detected_label}")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_label.split(" ")[0], (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Robot View", frame)
        cv2.imshow("Brain View (Sharp)", thresh_sharp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    picam2.stop()
    cv2.destroyAllWindows()