import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["QT_QPA_PLATFORM"] = "xcb" 
'''
# --- 1. LOAD YOUR SAVED DNA INTO A DICTIONARY ---
# This is much cleaner than creating 13 different variables!
base_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Danger": "danger.npy",
    "Fingerprint": "fingerprint.npy",
    "Kite": "kite.npy",
    "Octagon": "octagon.npy", # Fixed spelling
    "Plus": "plus.npy",
    "Press Button": "pressbutton.npy",
    "QR Code": "qrcode.npy",
    "Recycle": "recycle.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
}
'''
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
'''
templates = {}
try:
    for name, filename in template_files.items():
        templates[name] = np.load(os.path.join(base_path, filename))
except FileNotFoundError as e:
    print(f"Error: Could not find {e.filename}. Check the Symbols_npy folder!")
    exit()
'''

# --- 2. START THE CAMERA ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("System Ready! Scanning for symbols...")
'''
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's thresholding to isolate shapes automatically
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 1500:
                # Calculate Hu Moments for the live shape
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                
                best_match = None
                lowest_diff = 0.1 # Lower this if it's too sensitive

                # AUTOMATICALLY check against ALL templates in the dictionary
                for name, master_dna in templates.items():
                    diff = np.sum(np.abs(live_moments - master_dna))
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

finally:
    picam2.stop()
    cv2.destroyAllWindows()
'''
'''
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- PRE-PROCESSING ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. The Sharp Image (No Glue)
        thresh_sharp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 31, 5)
        
        # 2. The Glued Image (Adds the 19x19 brush on top of the sharp image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        thresh_glued = cv2.morphologyEx(thresh_sharp, cv2.MORPH_CLOSE, kernel)

        detected_label = None
        box_coords = None
        lowest_diff = 4.0 # Strict Log-Scale Threshold

        # --- BRAIN 1: CHECK SHARP SHAPES (Star, Plus, Arrow, etc.) ---
        cnts_sharp, _ = cv2.findContours(thresh_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_sharp:
            area = cv2.contourArea(c)
            if 2000 < area < 40000:
                x, y, w, h = cv2.boundingRect(c)
                if 0.6 <= (w/h) <= 1.4:
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    for name, master_dna in templates_sharp.items():
                        # Log-Scale Math
                        live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                        master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                        diff = np.sum(np.abs(live_log - master_log))
                        
                        if diff < lowest_diff:
                            lowest_diff = diff
                            detected_label = f"{name} (Diff: {lowest_diff:.2f})"
                            box_coords = (x, y, w, h)

        # --- BRAIN 2: CHECK GLUED SHAPES (QR Code, Fingerprint, Recycle) ---
        # Only run this if Brain 1 didn't already find a perfect match
        if not detected_label:
            cnts_glued, _ = cv2.findContours(thresh_glued, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_glued:
                area = cv2.contourArea(c)
                if 2000 < area < 40000:
                    x, y, w, h = cv2.boundingRect(c)
                    if 0.6 <= (w/h) <= 1.4:
                        live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                        for name, master_dna in templates_glued.items():
                            # Log-Scale Math
                            live_log = -np.sign(live_moments) * np.log10(np.abs(live_moments) + 1e-20)
                            master_log = -np.sign(master_dna) * np.log10(np.abs(master_dna) + 1e-20)
                            diff = np.sum(np.abs(live_log - master_log))
                            
                            if diff < lowest_diff:
                                lowest_diff = diff
                                detected_label = f"{name} (Diff: {lowest_diff:.2f})"
                                box_coords = (x, y, w, h)

        # --- DRAW THE WINNER ---
        if detected_label and box_coords:
            print(f">>> {detected_label}")
            x, y, w, h = box_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_label.split(" ")[0], (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # (Optional) Remove the "offscreen" rule at the top of your file to see this in VNC
        # cv2.imshow("Robot View", frame)
        # cv2.imshow("Glued Brain", thresh_glued)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
'''
# (Make sure to comment out os.environ["QT_QPA_PLATFORM"] = "offscreen" at the top so VNC works!)

try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Thresholding and pre-processing for both brains
        thresh_sharp = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 151, 6)
        # --- THE GLUE: For the broken shapes ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        thresh_glued = cv2.morphologyEx(thresh_sharp, cv2.MORPH_CLOSE, kernel)

        detected_label = None
        box_coords = None

        # --- BRAIN 1: CHECK SHARP SHAPES (Using your exact working math) ---
        cnts_sharp, _ = cv2.findContours(thresh_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts_sharp:
            # Your exact working filter
            if cv2.contourArea(c) > 1500:
                live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                lowest_diff = 0.1  # Your exact working threshold
                
                for name, master_dna in templates_sharp.items():
                    # Your exact working simple math (No log scale!)
                    diff = np.sum(np.abs(live_moments - master_dna))
                    
                    if diff < lowest_diff:
                        lowest_diff = diff
                        detected_label = f"{name} (Diff: {lowest_diff:.4f})"
                        box_coords = cv2.boundingRect(c)

        # --- BRAIN 2: CHECK GLUED SHAPES (Only if Brain 1 found nothing) ---
        if not detected_label:
            cnts_glued, _ = cv2.findContours(thresh_glued, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts_glued:
                # Same exact working filter and math, just applied to the glued shapes
                if cv2.contourArea(c) > 1500:
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 0.1  
                    
                    for name, master_dna in templates_glued.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        
                        if diff < lowest_diff:
                            lowest_diff = diff
                            detected_label = f"{name} (Diff: {lowest_diff:.4f})"
                            box_coords = cv2.boundingRect(c)

        # --- DRAW THE WINNER ---
        if detected_label and box_coords:
            print(f">>> Match Found: {detected_label}")
            x, y, w, h = box_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_label.split(" ")[0], (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the windows in VNC so you can debug!
        cv2.imshow("Robot View", frame)
        cv2.imshow("Brain View (Sharp - Otsu)", thresh_sharp)
        cv2.imshow("Brain View (Glued)", thresh_glued)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()