import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# --- 1. LOAD YOUR SAVED DNA INTO A DICTIONARY ---
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
    "QR Code": "qrcode.npy",
    "Recycle": "recycle.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
}

templates = {}
try:
    for name, filename in template_files.items():
        templates[name] = np.load(os.path.join(base_path, filename))
except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}. Check the Symbols_npy folder!")
    exit()

# --- 2. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("System Ready! Scanning for symbols...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's thresholding to isolate shapes automatically
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 15)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 1500:
                # Calculate Hu Moments for the live shape
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                
                best_match = None
                lowest_diff = 4.0 # Lower this if it's too sensitive

                # AUTOMATICALLY check against ALL templates in the dictionary
                for name, master_dna in templates.items():
                    live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                    master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                    diff = np.sum(np.abs(live_log - master_log))
                    if diff < lowest_diff:
                        lowest_diff = diff
                        best_match = name

                if best_match:
                    print(f"Match Found: {best_match} (Diff: {lowest_diff:.4f})")
                    
                    # Optional visual markers
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the live feed and the Otsu brain
        cv2.imshow("Robot View", frame)
        cv2.imshow("Otsu Brain", thresh)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()