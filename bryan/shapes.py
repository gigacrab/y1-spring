import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# loading ORB pngs
png_path = '/home/jaydenbryan/Project/Symbols_png/'
template_files_png = {
    "Danger": "danger.png",
    "Fingerprint": "fingerprint.png",
    "Press Button": "pressbutton.png",
    "Recycle": "recycle.png",
    "QR Code": "qrcode.png"
}

# ORB configs
orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=20) 
flann = cv2.FlannBasedMatcher(index_params, search_params)

# loading ORB features into a dictionary
template_features = {}
for label, filename in template_files_png.items():
    # 0 - reads grayscale image, uses BGR to gray
    img = cv2.imread(os.path.join(png_path, filename), 0) 
    if img is not None:
        # keypoints and descs
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            # creates key value pairs for the label
            template_features[label] = (kp, des)
    else:
        print(f"Warning: Missing ORB photo {filename}")

# loading hu moments files
npy_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files_npy = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Kite": "kite.npy",
    "Octagon": "octagon.npy", 
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy"
}

# loading npy files into the dictionary
templates_npy = {}
for name, filename in template_files_npy.items():
    try:
        templates_npy[name] = np.load(os.path.join(npy_path, filename))
    except FileNotFoundError:
        print(f"Warning: Missing DNA file {filename}")

# camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)
print("Camera set up!")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        best_match = None

        # uses adaptive thresholding, 151 - size of neighbourhood area, 8 - constant subtracted from weighted mean
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
        
        cnts, hrchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # just prints all contours
        im2 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.drawContours(im2, cnts, -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imshow("contours", im2)

        parents = []
        targets = []

        # why is it only one hole?
        # make sure that they follow the boundaries to check
        # try increasing neighbourhood area!
        
        # [next, previous, first_child, parent]
        if hrchy is not None:
            for i, c in enumerate(cnts): 
                if hrchy[0][i][3] == -1:
                    # we now know this is a parent
                    parents.append([i, c])
            cv2.drawContours(im2, [row[1] for row in parents], -1, (0, 0, 255), thickness=cv2.FILLED)
            for i, c in parents:
                # cv2.contourArea gives us closed area by external contour, so we can check area
                if cv2.contourArea(c) > 3000: #MODIFY THIS NUMBER LATER!
                    epsilon = 0.01 * cv2.arcLength(c, closed=True)
                    approx = cv2.approxPolyDP(c, epsilon, closed=True)
                    if len(approx) == 4:
                        # keep the hierarchy in the first element
                        targets.append([hrchy[0][i], c])    
            # now we take a look at the targets, hopefully it's right
            cv2.drawContours(im2, [row[1] for row in targets], -1, (0, 255, 0), thickness=cv2.FILLED)
            
            for hrc, c in targets:
                holes = 0
                # now we check the children, by using the parent's first child and continuing on
                # if there are holes, we go to ORB
                # if there aren't any, we go to shapes!
                
                # first child of the parent
                curr_i = hrc[2]
                # first child of the child, because it's its internal contour
                if cv2.contourArea(cnts[curr_i]) - cv2.contourArea(cnts[hrchy[0][curr_i][2]]) < 50:
                    curr_i = hrchy[0][curr_i][2]
                while curr_i != -1:
                    # now we check this child
                    if cv2.contourArea(cnts[curr_i]) > 200:
                        holes += 1
                        # remember that this accepts an array of contours!
                        cv2.drawContours(im2, [cnts[curr_i]], -1, (255, 0, 0), thickness=cv2.FILLED)
                    # set to the next child
                    curr_i = hrchy[0][curr_i][0]
                print(f'there are {holes} holes!')

                # see blue holes
                cv2.imshow("holes", im2)
                if holes == 0:
                    # check for shapes!
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff = 0.05 
                    geom_match = None
                    
                    for name, master_dna in templates_npy.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            geom_match = name
                            
                    if geom_match in ["Plus", "Kite"]:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        corners = len(approx)
                        geom_match = "Kite" if corners < 8 else "Plus"
                    
                    if geom_match == "Arrow":
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) > 9: 
                            geom_match = None
                        if geom_match == "Arrow":
                            x, y, w, h = cv2.boundingRect(c)
                            
                            # 1. Create a perfect digital silhouette of the arrow
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
                            
                            # 2. Grab the extreme 15% edges of all four sides
                            margin_x = max(1, int(w * 0.15))
                            margin_y = max(1, int(h * 0.15))
                            
                            top_edge = mask[:margin_y, :]
                            bottom_edge = mask[-margin_y:, :]
                            left_edge = mask[:, :margin_x]
                            right_edge = mask[:, -margin_x:]
                            
                            # 3. Weigh the ink on all four edges
                            # An arrow has 3 sharp points (low mass) and 1 flat tail (high mass)!
                            masses = {
                                "Arrow (UP)": cv2.countNonZero(top_edge),   # If Top is the heavy tail, it points DOWN
                                "Arrow (DOWN)": cv2.countNonZero(bottom_edge),  # If Bottom is the heavy tail, it points UP
                                "Arrow (LEFT)": cv2.countNonZero(left_edge), # If Left is the heavy tail, it points RIGHT
                                "Arrow (RIGHT)": cv2.countNonZero(right_edge)  # If Right is the heavy tail, it points LEFT
                            }
                            
                            # 4. The arrow points in the opposite direction of the heaviest edge!
                            geom_match = max(masses, key=masses.get)
                    # --------------------------------
                    if geom_match in ["Star", "Kite"]:
                        # Draw a rotating box that perfectly hugs the shape
                        rect = cv2.minAreaRect(c)
                        w_rect, h_rect = rect[1]
                        
                        if w_rect != 0 and h_rect != 0:
                            # Divide the long side by the short side
                            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                            
                            # A QR Code block is a perfect square (ratio ~ 1.0)
                            # A Kite is stretched out (ratio usually > 1.3)
                            if aspect_ratio < 1.15:
                                geom_match = None  # It's a square! Reject it and let ORB scan!
                    # --------------------------------

                    if geom_match:
                        best_match = geom_match
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        break
                else:
                    gray_processed = cv2.equalizeHist(blurred)
                    kp_frame, des_frame = orb.detectAndCompute(gray_processed, None)
                    max_good_matches = 0
                    
                    if des_frame is not None and len(des_frame) >= 2:
                        for label, (kp_template, des_template) in template_features.items():
                            if des_template is not None:
                                matches = flann.knnMatch(des_template, des_frame, k=2)
                                
                                good_matches = []
                                for m_n in matches:
                                    if len(m_n) == 2:
                                        m, n = m_n
                                        if m.distance < 0.75 * n.distance:
                                            good_matches.append(m)
                                
                                # Danger is strict, Fingerprint is loose
                                if label == "Danger": required_matches = 8
                                elif label == "Fingerprint": required_matches = 15
                                else: required_matches = 12
                                
                                if len(good_matches) >= required_matches and len(good_matches) > max_good_matches:
                                    max_good_matches = len(good_matches)
                                    best_match = label
        if best_match:
            cv2.putText(frame, f"MATCH: {best_match}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Robot View", frame)
        
        # --- SAFE QUIT ---
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1: break

finally:
    picam2.stop()
    cv2.destroyAllWindows()