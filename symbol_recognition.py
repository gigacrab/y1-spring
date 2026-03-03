import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

# --- 1. LOAD YOUR SAVED DNA (Now only used as a fallback!) ---
base_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files = {
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Danger": "danger.npy",
    "Fingerprint": "fingerprint.npy",
    "Kite": "kite.npy",          # 4 sides
    "Trapezium": "trapezium.npy",# 4 sides
    "Press Button": "pressbutton.npy",
    "Recycle": "recycle.npy"
}

templates = {}
try:
    for name, filename in template_files.items():
        templates[name] = np.load(os.path.join(base_path, filename))
except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}.")
    exit()

qr_decoder = cv2.QRCodeDetector()

# --- 2. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Hybrid System Ready! Scanning...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- LAYER 1: QR CODE SHIELD ---
        try:
            data, bbox, _ = qr_decoder.detectAndDecode(frame)
            if data:
                print(f"QR Code Detected! Message: {data}")
                continue # Skip the rest of the loop to save CPU!
        except Exception:
            pass

        # --- PRE-PROCESSING: Shadow Killer & Shrink-Wrap ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 5)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area > 1000:
                
                # --- LAYER 2: THE CORNER COUNTER (Fast Math) ---
                # Calculate the perimeter of the shape
                peri = cv2.arcLength(c, True)
                
                # 'approx' mathematically smooths the shape. 
                # 0.03 is the magic "strictness" multiplier for a Pi camera.
                approx = cv2.approxPolyDP(c, 0.03 * peri, True)
                vertices = len(approx)
                
                label = None
                
                if vertices == 7:
                    label = "Arrow"
                elif vertices == 8:
                    label = "Octagon"
                elif vertices == 10:
                    label = "Star"
                elif vertices == 12:
                    label = "Plus"
                
                # --- LAYER 3: HU MOMENTS FALLBACK (Heavy Math) ---
                # If the shape has curves (Circle) or crazy edges (Fingerprint), 
                # vertices will be a random number like 15+. We send it to Hu Moments.
                if label is None:
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 15.0 # Keep floodgates slightly open
                    
                    for name, master_dna in templates.items():
                        live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                        master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                        
                        diff = np.sum(np.abs(live_log - master_log))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            label = f"{name} (Diff: {lowest_diff:.1f})"
                
                # --- FINAL OUTPUT ---
                if label:
                    print(f"Detected: {label} | Vertices: {vertices} | Area: {area:.0f}")
                    
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label.split(" ")[0], (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

finally:
    picam2.stop()
    cv2.destroyAllWindows()