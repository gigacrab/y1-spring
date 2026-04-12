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
    "Kite": "kite.npy",
    "Octagon": "octagon.npy", 
    "Plus": "plus.npy",
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
print("System Ready! Scanning for shapes with advanced shields...")

prev_frame_time = 0

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # --- THE DONUT FIX ---
        # Increased block size to 255 to stop glare from hollowing out shapes in dark corners
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # --- RETR_TREE FIX ---
        # We must use RETR_TREE to see inside the shapes for the "Holes" gatekeeper
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is not None:
            for i, c in enumerate(cnts):
                if cv2.contourArea(c) > 1500:
                    
                    # --- THE HOLES GATEKEEPER ---
                    holes = 0
                    for j, child_c in enumerate(cnts):
                        if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 500:
                            holes += 1
                            
                    if holes == 0:
                        # Calculate geometry metrics
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        corners = len(approx)
                        
                        hull = cv2.convexHull(c)
                        hull_area = cv2.contourArea(hull)
                        area = cv2.contourArea(c)
                        solidity = area / float(hull_area) if hull_area > 0 else 0
                        
                        # Calculate Hu Moments
                        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                        
                        best_match = None
                        lowest_diff = 0.10  # Relaxed for camera lens distortion

                        for name, master_dna in templates.items():
                            diff = np.sum(np.abs(live_moments - master_dna))
                            if diff < lowest_diff:
                                lowest_diff = diff
                                best_match = name
                                
                        # --- TIE-BREAKER ---
                        if best_match in ["Plus", "Kite"]:
                            best_match = "Kite" if corners < 8 else "Plus"

                        # --- THE FACT CHECKERS (BOUNCERS) ---
                        if best_match:
                            x, y, w, h = cv2.boundingRect(c)
                            box_area = w * h
                            extent = area / float(box_area) if box_area > 0 else 0
                            
                            if best_match == "Star":
                                if solidity > 0.6 or corners < 8:
                                    best_match = None  
                            elif best_match == "Octagon":
                                if corners < 5 or solidity < 0.75:
                                    best_match = None
                            elif best_match == "Kite":
                                if extent > 0.75: # Blocks the QR Square
                                    best_match = None 
                            elif best_match == "3/4 Circle":
                                if solidity > 0.95:
                                    best_match = None

                        # --- ARROW PHYSICS & BOOMERANG SHIELD ---
                        if best_match == "Arrow":
                            if not (6 <= corners <= 9) or solidity < 0.55:
                                best_match = None  # Rejects the curvy Recycle arrow!
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
                                        best_match = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                                    else:
                                        best_match = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"

                        # --- DRAW RESULTS ---
                        if best_match:
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, f"{best_match} ({lowest_diff:.3f})", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break # Once we find a match, stop looking at other contours

        # --- CALCULATE FPS ---
        new_frame_time = time.perf_counter()
        time_diff = new_frame_time - prev_frame_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_frame_time = new_frame_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the live feed and the Threshold Brain
        cv2.imshow("Robot View", frame)
        cv2.imshow("Threshold Brain", thresh)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()