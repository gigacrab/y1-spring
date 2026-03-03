import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

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
    # Notice: "QR Code" is removed! We use the smart scanner for this now.
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

# --- 2. INITIALIZE THE QR SCANNER ---
qr_decoder = cv2.QRCodeDetector()

# --- 3. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("System Ready! Scanning for shapes and QR codes...")

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- 4. CHECK FOR QR CODE FIRST ---
        # This completely bypasses Hu Moments and uses OpenCV's dedicated QR reader
        data, bbox, _ = qr_decoder.detectAndDecode(frame)
        if data:
            print(f"QR Code Detected! Hidden Message: {data}")
            # Optional: Draw a blue box around the QR code
            if bbox is not None:
                for i in range(len(bbox[0])):
                    pt1 = tuple(map(int, bbox[0][i]))
                    pt2 = tuple(map(int, bbox[0][(i+1) % 4]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
                cv2.putText(frame, "QR Code", tuple(map(int, bbox[0][0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # --- 5. SHAPE DETECTION (For everything else) ---
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 1500:
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                
                best_match = None
                # Note: Log-scale differences are naturally larger numbers, so we start at 0.5
                lowest_diff = 0.5 

                for name, master_dna in templates.items():
                    # --- THE PRO MATH: Logarithmic Scale ---
                    live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                    master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                    
                    diff = np.sum(np.abs(live_log - master_log))
                    
                    if diff < lowest_diff:
                        lowest_diff = diff
                        best_match = name

                if best_match:
                    print(f"Match Found: {best_match} (Diff: {lowest_diff:.4f})")
                    
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, best_match, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

finally:
    picam2.stop()
    cv2.destroyAllWindows()