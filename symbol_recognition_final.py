import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os

os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==========================================
# 1. LOAD ORB TEMPLATES (Phase 1: Art)
# ==========================================
png_path = '/home/jaydenbryan/Project/Symbols_png/'
template_files_png = {
    "Danger": "danger.png",
    "Fingerprint": "fingerprint.png",
    "Press Button": "pressbutton.png",
    "Recycle": "recycle.png",
    "QR Code": "qrcode.png"
}

# --- THE CPU FIX: Optimized so the Pi doesn't stutter! ---
orb = cv2.ORB_create(nfeatures=800, fastThreshold=12)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=20) 
flann = cv2.FlannBasedMatcher(index_params, search_params)

template_features = {}
for label, filename in template_files_png.items():
    img = cv2.imread(os.path.join(png_path, filename), 0)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            template_features[label] = (kp, des)
    else:
        print(f"Warning: Missing ORB photo {filename}")

# ==========================================
# 2. LOAD HU MOMENTS DNA (Phase 2: Geometry)
# ==========================================
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

templates_npy = {}
for name, filename in template_files_npy.items():
    try:
        templates_npy[name] = np.load(os.path.join(npy_path, filename))
    except FileNotFoundError:
        print(f"Warning: Missing DNA file {filename}")

# ==========================================
# 3. START CAMERA & MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

# --- DEBUG FLAGS: turn these on/off as needed ---
DEBUG_PRINT_CONTOURS = True   # prints hierarchy table to terminal
DEBUG_PRINT_HU       = True   # prints Hu Moments scores to terminal
DEBUG_PRINT_ORB      = True   # prints ORB match counts to terminal
DEBUG_WINDOW         = True   # shows the 2x2 debug window

print("Hybrid Master Brain Ready!")

try:
    while True:
        frame = picam2.capture_array()
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        best_match = None
        frame_area = frame.shape[0] * frame.shape[1]  # calculate ONCE per frame

        # ==========================================
        # PHASE 1: GEOMETRY
        # ==========================================
        thresh = cv2.adaptiveThreshold(blurred, 255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY_INV, 151, 10)

        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is not None:

            # --- DEBUG: Print contour hierarchy table ONCE per frame ---
            if DEBUG_PRINT_CONTOURS:
                print(f"\n{'='*60}")
                print(f"{'ID':>4} | {'Area':>7} | {'Parent':>6} | {'Child':>5} | Note")
                print(f"{'-'*60}")
                for i, c in enumerate(cnts):
                    area = cv2.contourArea(c)
                    if area < 500:
                        continue
                    parent = hierarchy[0][i][3]
                    child  = hierarchy[0][i][2]
                    
                    # Annotate what each contour likely is
                    if area > frame_area * 0.8:
                        note = "SKIP (too large)"
                    elif area < 1500:
                        note = "SKIP (too small)"
                    elif parent == -1 and child != -1:
                        note = ">>> IS BOX (skipped)"
                    elif parent != -1:
                        note = "*** SYMBOL CANDIDATE"
                    else:
                        note = ""
                    
                    print(f"{i:>4} | {area:>7.0f} | {parent:>6} | {child:>5} | {note}")

            for i, c in enumerate(cnts):
                area = cv2.contourArea(c)

                # Size filter
                if area < 1500 or area > frame_area * 0.8:
                    continue

                parent_index = hierarchy[0][i][3]
                first_child  = hierarchy[0][i][2]

                # Box filter
                is_box = (parent_index == -1 and first_child != -1)
                if is_box:
                    continue

                # Holes filter
                holes = 0
                for j, child_c in enumerate(cnts):
                    if hierarchy[0][j][3] == i and cv2.contourArea(child_c) > 200:
                        holes += 1

                if holes == 0:
                    live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
                    lowest_diff  = 0.05
                    geom_match   = None

                    # --- DEBUG: Print ALL Hu Moment scores ---
                    if DEBUG_PRINT_HU:
                        print(f"\n  [HU] Contour {i} | area={area:.0f} | holes={holes}")
                        for name, master_dna in templates_npy.items():
                            diff = np.sum(np.abs(live_moments - master_dna))
                            bar  = '█' * int(min(diff * 200, 40))  # visual bar
                            tick = " ✓" if diff < lowest_diff else ""
                            print(f"    {name:20s}: {diff:.4f} {bar}{tick}")

                    for name, master_dna in templates_npy.items():
                        diff = np.sum(np.abs(live_moments - master_dna))
                        if diff < lowest_diff:
                            lowest_diff = diff
                            geom_match  = name

                    # ... your existing Plus/Kite/Arrow/Star logic unchanged ...
                    if geom_match in ["Plus", "Kite"]:
                        peri   = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        corners = len(approx)
                        geom_match = "Kite" if corners < 8 else "Plus"

                    if geom_match == "Arrow":
                        peri   = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                        if len(approx) > 9:
                            geom_match = None
                        if geom_match == "Arrow":
                            x, y, w, h = cv2.boundingRect(c)
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
                            margin_x = max(1, int(w * 0.15))
                            margin_y = max(1, int(h * 0.15))
                            masses = {
                                "Arrow (UP)":    cv2.countNonZero(mask[:margin_y, :]),
                                "Arrow (DOWN)":  cv2.countNonZero(mask[-margin_y:, :]),
                                "Arrow (LEFT)":  cv2.countNonZero(mask[:, :margin_x]),
                                "Arrow (RIGHT)": cv2.countNonZero(mask[:, -margin_x:])
                            }
                            if DEBUG_PRINT_HU:
                                print(f"    [ARROW] masses: {masses}")
                            geom_match = max(masses, key=masses.get)

                    if geom_match == "Star":
                        rect = cv2.minAreaRect(c)
                        w_rect, h_rect = rect[1]
                        if w_rect != 0 and h_rect != 0:
                            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
                            if DEBUG_PRINT_HU:
                                print(f"    [STAR] aspect_ratio={aspect_ratio:.3f} ({'REJECT' if aspect_ratio < 1.15 else 'KEEP'})")
                            if aspect_ratio < 1.15:
                                geom_match = None

                    if geom_match:
                        best_match = geom_match
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        break

        # ==========================================
        # PHASE 2: ORB
        # ==========================================
        if best_match is None:
            gray_processed = cv2.equalizeHist(blurred)
            kp_frame, des_frame = orb.detectAndCompute(gray_processed, None)
            max_good_matches = 0

            if des_frame is not None and len(des_frame) >= 2:

                # --- DEBUG: Print ORB scores ---
                if DEBUG_PRINT_ORB:
                    print(f"\n  [ORB] Keypoints in frame: {len(des_frame)}")

                for label, (kp_template, des_template) in template_features.items():
                    if des_template is not None:
                        matches = flann.knnMatch(des_template, des_frame, k=2)
                        good_matches = [m for m_n in matches
                                        if len(m_n) == 2
                                        for m, n in [m_n]
                                        if m.distance < 0.75 * n.distance]

                        required = 8 if label == "Danger" else 15 if label == "Fingerprint" else 12
                        status   = "✓ PASS" if len(good_matches) >= required else "✗"

                        if DEBUG_PRINT_ORB:
                            filled = int((len(good_matches) / required) * 20)
                            bar    = '█' * min(filled, 20) + '░' * (20 - min(filled, 20))
                            print(f"    {label:15s}: [{bar}] {len(good_matches):3d}/{required} {status}")

                        if len(good_matches) >= required and len(good_matches) > max_good_matches:
                            max_good_matches = len(good_matches)
                            best_match = label

        # ==========================================
        # DISPLAY
        # ==========================================
        if best_match:
            cv2.putText(frame, f"MATCH: {best_match}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Robot View", frame)

        # --- DEBUG WINDOW ---
        if DEBUG_WINDOW:
            debug_contours = frame.copy()
            debug_thresh   = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # All contours red, valid-sized ones green with area label
            cv2.drawContours(debug_contours, cnts, -1, (0, 0, 255), 1)
            for i, c in enumerate(cnts):
                area = cv2.contourArea(c)
                if area > 1500:
                    # Colour code by type
                    parent = hierarchy[0][i][3] if hierarchy is not None else -1
                    child  = hierarchy[0][i][2] if hierarchy is not None else -1
                    if parent == -1 and child != -1:
                        colour = (0, 165, 255)   # orange = box
                    elif parent != -1:
                        colour = (0, 255, 0)     # green = symbol candidate
                    else:
                        colour = (255, 255, 0)   # yellow = other

                    cv2.drawContours(debug_contours, [c], -1, colour, 2)
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.putText(debug_contours, f"{area:.0f}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, colour, 1)

            top      = np.hstack([frame, debug_thresh])
            bottom   = np.hstack([debug_contours, debug_thresh.copy()])
            combined = np.vstack([top, bottom])
            combined = cv2.resize(combined, (1280, 720))
            cv2.imshow("DEBUG", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if cv2.getWindowProperty("Robot View", cv2.WND_PROP_VISIBLE) < 1: break

finally:
    picam2.stop()
    cv2.destroyAllWindows()