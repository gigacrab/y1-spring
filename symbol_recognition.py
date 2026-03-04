import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# --- 1. LOAD SAVED DNA ---
base_path = '/home/jaydenbryan/Project/Symbols_npy/'

# Brain 1: These shapes NEED their sharp corners
sharp_files = {
    "Arrow": "arrow.npy", "3/4 Circle": "circle34.npy", 
    "Major Segment": "circlemajorsegment.npy", "Danger": "danger.npy",
    "Kite": "kite.npy", "Octagon": "octagon.npy", "Plus": "plus.npy", 
    "Press Button": "pressbutton.npy", "Star": "star.npy", "Trapezium": "trapezium.npy"
}

# Brain 2: These shapes are fragmented and NEED the glue
glued_files = {
    "Fingerprint": "fingerprint.npy", 
    "QR Code": "qrcode.npy", 
    "Recycle": "recycle.npy"
}

templates_sharp = {}
templates_glued = {}

for name, filename in sharp_files.items():
    try: templates_sharp[name] = np.load(os.path.join(base_path, filename))
    except: pass

for name, filename in glued_files.items():
    try: templates_glued[name] = np.load(os.path.join(base_path, filename))
    except: pass

# --- 2. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("System Ready! Scanning for symbols...")

# (Make sure to comment out os.environ["QT_QPA_PLATFORM"] = "offscreen" at the top so VNC works!)

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Erase the paper edge shadow!
        thresh_sharp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 15)
        
        # --- THE GLUE ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        thresh_glued = cv2.morphologyEx(thresh_sharp, cv2.MORPH_CLOSE, kernel)

        best_label = None
        best_box = None
        best_moments = None # <--- NEW: We will store the actual winner's DNA here!
        largest_valid_area = 0

        # --- CHECK GLUED SHAPES FIRST ---
        cnts_glued, _ = cv2.findContours(thresh_glued, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts_glued:
            area = cv2.contourArea(c)
            if 1500 < area < 60000 and area > largest_valid_area:
                x, y, w, h = cv2.boundingRect(c)
                
                if 0.4 <= (w/h) <= 2.5: 
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 0.15 
                    
                    for name, master_dna in templates_glued.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            best_label = f"{name} (Diff: {lowest_diff:.4f})"
                            best_box = (x, y, w, h)
                            best_moments = live_moments # <--- SAVE THE WINNER'S DNA!
                            largest_valid_area = area

        # --- CHECK SHARP SHAPES ---
        cnts_sharp, _ = cv2.findContours(thresh_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts_sharp:
            area = cv2.contourArea(c)
            if 1500 < area < 60000 and area > largest_valid_area:
                x, y, w, h = cv2.boundingRect(c)
                
                if 0.4 <= (w/h) <= 2.5:
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 0.1  
                    
                    for name, master_dna in templates_sharp.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            best_label = f"{name} (Diff: {lowest_diff:.4f})"
                            best_box = (x, y, w, h)
                            best_moments = live_moments # <--- SAVE THE WINNER'S DNA!
                            largest_valid_area = area

        # --- DRAW THE WINNER ---
        if best_label and best_box:
            print(f">>> Match Found: {best_label}")
            x, y, w, h = best_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, best_label.split(" ")[0], (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Robot View", frame)
        cv2.imshow("Brain View (Sharp)", thresh_sharp)
        cv2.imshow("Brain View (Glued)", thresh_glued)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
            
        # --- THE SMART DNA SCANNER ---
        elif key == ord('s'):
            if best_box is not None and best_moments is not None:
                save_path = '/home/jaydenbryan/Project/Symbols_npy/recycle.npy'
                # 1. Save it to the hard drive
                np.save(save_path, best_moments)
                
                # 2. Instantly update the robot's active memory so you don't have to restart!
                templates_glued["Recycle"] = best_moments
                
                print(f"\n[SUCCESS] DNA SAVED AND BRAIN UPDATED INSTANTLY!\n")
            else:
                print("No shape detected to save!")

finally:
    picam2.stop()
    cv2.destroyAllWindows()