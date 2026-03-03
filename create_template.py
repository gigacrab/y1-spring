import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

# --- WE REMOVED THE "OFFSCREEN" RULE SO VNC CAN OPEN WINDOWS! ---
os.environ["DISPLAY"] = ":0"           # Tells it to use the main monitor
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# --- 1. LOAD SAVED DNA ---
base_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Danger": "danger.npy",
    "Fingerprint": "fingerprint.npy",
    "Kite": "kite.npy",
    "Octagon": "octagon.npy",
    "Plus": "plus.npy",
    "Press Button": "pressbutton.npy",
    "Recycle": "recycle.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
}

templates = {}
try:
    for name, filename in template_files.items():
        templates[name] = np.load(os.path.join(base_path, filename))
except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}.")
    exit()

qr_decoder = cv2.QRCodeDetector()

# --- 2. START CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("VNC Debug Mode Ready! Look for the two video windows...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- 3. QR CODE SCANNER (Shielded) ---
        try:
            data, bbox, _ = qr_decoder.detectAndDecode(frame)
            if data:
                print(f"QR Code Detected: {data}")
                if bbox is not None:
                    for i in range(len(bbox[0])):
                        pt1 = tuple(map(int, bbox[0][i]))
                        pt2 = tuple(map(int, bbox[0][(i+1) % 4]))
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
                    cv2.putText(frame, "QR Code", tuple(map(int, bbox[0][0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        except Exception:
            pass

        # --- 4. PRE-PROCESSING (Shadow Killer & Shrink-Wrap) ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 5)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 5. THE SNIPER FILTER ---
        for c in cnts:
            area = cv2.contourArea(c)
            
            # FILTER 1: Strict Size (Ignore background noise and giant shadows)
            if 4000 < area < 35000:
                
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                
                # FILTER 2: Shape Proportions (Is it roughly square?)
                if 0.6 <= aspect_ratio <= 1.4:
                    
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    best_match = None
                    lowest_diff = 3.0  # Strict Log-Scale Threshold

                    for name, master_dna in templates.items():
                        live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                        master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                        
                        diff = np.sum(np.abs(live_log - master_log))
                        
                        if diff < lowest_diff:
                            lowest_diff = diff
                            best_match = name

                    if best_match:
                        print(f"LOCKED ON: {best_match} (Diff: {lowest_diff:.2f})")
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, best_match, (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 6. LIVE VNC DISPLAY ---
        # Show what the robot sees in color
        cv2.imshow("Robot View", frame)
        
        # Show the X-Ray "Brain" view (CRITICAL FOR DEBUGGING LIGHTING)
        cv2.imshow("Black & White Brain", thresh)
        
        # Press 'q' on your keyboard to safely exit the script
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Camera...")
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()