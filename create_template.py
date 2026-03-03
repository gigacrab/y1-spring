import cv2
import numpy as np

image_path = r"C:\Users\WINDOWS 11\Documents\Documents\UNM\Year_1\Applied Electrical and Electronic Engineering\y1-spring\Symbols_png\symbol_fingerprint1.png"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not find {image_path}. Check the file name!")
    exit()

# Convert to grayscale and threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5), np.uint8)
dilated_thresh = cv2.dilate(thresh, kernel, iterations=0)

# Find the shape
cnts, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(cnts) > 0:
    # Get the largest contour of the shape
    c = max(cnts, key=cv2.contourArea)
    
    # Extract the Math value of 7 Hu Moments
    moments = cv2.HuMoments(cv2.moments(c)).flatten()
    
    # Save it permanently
    np.save(r"C:\Users\WINDOWS 11\Documents\Documents\UNM\Year_1\Applied Electrical and Electronic Engineering\y1-spring\Symbols_npy\fingerprint.npy", moments)
    
    print("SUCCESS! Master Template DNA Saved:")
    print(moments)
    
    # Show what the computer sees just to be safe
    cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
    cv2.imshow("Extracted Shape", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to find a shape in the image.")