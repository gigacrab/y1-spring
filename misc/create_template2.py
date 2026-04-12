import cv2
import numpy as np

# Change this to whichever symbol you are testing
image_path = r"C:\Users\WINDOWS 11\Documents\Documents\UNM\Year_1\Applied Electrical and Electronic Engineering\y1-spring\Symbols_png\symbol_danger.png"
save_path = r"C:\Users\WINDOWS 11\Documents\Documents\UNM\Year_1\Applied Electrical and Electronic Engineering\y1-spring\Symbols_npy\danger.npy"

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not find {image_path}. Check the file name!")
    exit()

# --- 1. THE PI SIMULATOR (Force laptop to see like the robot) ---
# Resize the perfect high-res image down to the Pi's camera resolution
image = cv2.resize(image, (640, 480))

# Convert to grayscale and blur exactly like the Pi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply the exact same Adaptive Threshold from your Pi script
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 151, 15)

# Apply the exact same Glue brush from your Pi script
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
dilated_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# --- 2. FIND THE SHAPE ---
# (We search the glued image so the QR Code and Fingerprint stay connected)
cnts, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) > 0:
    # Filter out tiny static, get the largest valid shape
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 1500]
    
    if valid_cnts:
        c = max(valid_cnts, key=cv2.contourArea)
        
        # Extract the DNA
        moments = cv2.HuMoments(cv2.moments(c)).flatten()
        np.save(save_path, moments)
        
        print("SUCCESS! Master Template DNA Saved:")
        print(moments)
        
        # Show what the robot's brain actually sees
        cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        cv2.imshow("What the Robot Sees (Glued DNA)", dilated_thresh)
        cv2.imshow("Extracted Shape", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed: Contours found, but they were too small.")
else:
    print("Failed to find any shapes in the image.")