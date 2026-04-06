# it should always show output, while showing final verdict at the end of each frame

import cv2
import numpy as np 
from picamera2 import Picamera2
import time

MIN_AREA = 3000
MAX_ASPECT_RATIO = 1.6

def check_special_in_group(i, cnts, hrchy):
    arc_count    = 0
    square_count = 0
    arrow_count  = 0

    child_idx = hrchy[0][i][2] # first child of inner border

    if child_idx == -1:
        return "No idea"

    # keep on checking next child
    while True:
        c    = cnts[child_idx]
        area = cv2.contourArea(c)
        print(f"area: {area}")
        if area > MIN_AREA * 0.3:
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w != 0 and h != 0:
                ar        = max(w, h) / min(w, h)
                rect_area = w_rot * h_rot
                extent = area / rect_area if rect_area > 0 else 0
                corners   = len(cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True))

                if ar > 4 or ar > 1.8 and extent < 0.5:
                    arc_count += 1
                if corners == 4 and 0.9 < extent < 1.1 and ar < 1.15:
                    square_count += 1
                if 12 <= corners <= 18 and 1.5 <= ar <= 2.0 and 0.45 <= extent <= 0.65:
                    arrow_count += 1

        child_idx = hrchy[0][child_idx][0] # next sibling
        if child_idx == -1:
            break

    if arrow_count >= 3:
        return "Recycle"
    if arc_count >= 3:
        return "Fingerprint"
    if square_count >= 3:
        return "QR Code"
    return "No idea"

def shape_detect(i, c, cnts, hrchy):
    area = cv2.contourArea(c)
    if area < MIN_AREA:
        return None

    sel_c = None
    w_rot, h_rot = 0, 0

    child_idx = hrchy[0][i][2]

    # check if it has a child, and is at hierarchy 0
    if hrchy[0][i][3] == -1:
        if child_idx != -1:
            child_area = cv2.contourArea(cnts[child_idx])
            hollow_ratio = child_area / area if area > 0 else 0

            if hollow_ratio > 0.85:
                print(f"Hollow container at: {i}")

                min_rect = cv2.minAreaRect(c)
                w_rot, h_rot = min_rect[1]
                if w_rot == 0 or h_rot == 0:
                    return None
                rect_area = w_rot * h_rot
                extent = area / rect_area if rect_area > 0 else 0

                if extent > 0.85: # parent is rectangle-like, confirms it's a container
                    # Find largest valid grandchild instead of assuming first
                    largest_gc = None
                    largest_gc_area = 0
                    gchild_idx = hrchy[0][child_idx][2] # first grandchild
                    sel_i = i
                    total_area = 0

                    while gchild_idx != -1:
                        gc_curr = cnts[gchild_idx]
                        gc_area = cv2.contourArea(gc_curr)
                        total_area += gc_area
                        if gc_area > MIN_AREA and gc_area > largest_gc_area:
                            largest_gc = gc_curr
                            largest_gc_area = gc_area
                            sel_i = gchild_idx
                        gchild_idx = hrchy[0][gchild_idx][0] # next sibling

                    if largest_gc is not None:
                        sel_c = largest_gc
                        min_rect = cv2.minAreaRect(sel_c)
                        w_rot, h_rot = min_rect[1]
                        if w_rot == 0 or h_rot == 0:
                            return None
                        print(f"Contained shape for: {sel_i}")
                    # to not recognize fingerprint, QR and recycling
                    elif total_area > MIN_AREA:
                        return child_idx
        
        if sel_c is None:
            return None
        else:
            return sel_c, w_rot, h_rot, min_rect, area

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

prev_frame_time = 0
fps = 0

try:
    while True:
        frame = picam2.capture_array()
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 8
        )

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        cnts, hrchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        total_c = len(cnts)
        print(f"Number of contours: {total_c}")

        answer = []
        containers = []

        # should already have hierarchy if a contour exists
        for i, c in enumerate(cnts):
            pred = "No idea"
            result = shape_detect(i, c, cnts, hrchy)

            if result is not None: # found container
                if isinstance(result, (int, np.integer)): # no valid contours
                    containers.append(result)
                else: # has valid contours
                    sel_c, w_rot, h_rot, min_rect, area = result

                    peri = cv2.arcLength(sel_c, True)
                    approx = cv2.approxPolyDP(sel_c, 0.01 * peri, True)
                    corners = len(approx)
                    aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot) if min(w_rot, h_rot) > 0 else 0

                    # Convex hull
                    hull = cv2.convexHull(sel_c)
                    hull_area = cv2.contourArea(hull)
                    sel_area = cv2.contourArea(sel_c)
                    solidity = sel_area / hull_area if hull_area > 0 else 0

                    # Rotated extent
                    extent = sel_area / (w_rot * h_rot) if w_rot*h_rot > 0 else 0

                    ellipse_area_ratio = 0

                    if solidity < 0.70:
                        if solidity < 0.55 and extent < 0.40:
                            pred = "Star"
                        else:
                            pred = "Arrow"
                    elif corners == 4:
                        if extent < 0.80:
                            pred = "Trapezium"
                        else:
                            pred = "Kite"
                    elif aspect_ratio > 1.35:
                        pred = "Major Segment"
                    elif corners == 8:
                        pred = "Octagon"
                    elif corners == 12 and aspect_ratio < 1.1:
                        pred = "Plus"
                    else:
                        # Ellipse area ratio
                        (xc, yc), radius = cv2.minEnclosingCircle(sel_c)
                        circle_area = np.pi * radius * radius
                        ellipse_area_ratio = sel_area / circle_area if circle_area > 0 else 0

                        if ellipse_area_ratio < 0.65 and extent > 0.75:
                            pred = "Press Button"
                        elif ellipse_area_ratio < 0.8:
                            pred = "3/4 Circle"
                        elif ellipse_area_ratio < 1.05:
                            pred = "Danger"
                        else:
                            pred = "No idea"

                    if pred != "No idea":
                        answer.append(pred)
                
                    box = cv2.boxPoints(min_rect)
                    box = np.intp(box)
                    cv2.drawContours(output, [sel_c], -1, (0, 255, 0), 2)
                    cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
                    cv2.putText(output, f"{pred}", (int(min_rect[0][0]-min_rect[1][0]/2), int(min_rect[0][1]-10-min_rect[1][1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                    child_area_debug = cv2.contourArea(cnts[hrchy[0][i][2]]) if hrchy[0][i][2] != -1 else -1
                    print(f"(Single) P:{hrchy[0][i]} C:{corners} AR:{aspect_ratio:.2f} S:{solidity:.2f} E:{extent:.2f} R:{ellipse_area_ratio:.2f} A:{area:.2f} AC:{child_area_debug}")        

            # jumps to here if no containers, then continues for loop

        for container in containers:
            pred = check_special_in_group(container, cnts, hrchy)
            if pred != "No idea":
                answer.append(pred)
            
            
        cv2.imshow("Threshold", closed)
        cv2.imshow("Geometry Debug", output)
            
        '''
        for i, c in enumerate(cnts):
            print(f"{i} hr:{hrchy[0][i]}, a:{cv2.contourArea(c)}")
        '''

        if len(answer) == 0:
            print("Final verdict: No shapes")
        else:
            print(f"Final verdict: {answer}")
            
        if cv2.waitKey(1) == ord('q'):
            break
            
        new_frame_time = time.perf_counter()
        
        time_diff = new_frame_time - prev_frame_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        else:
            fps = 0.0
        prev_frame_time = new_frame_time
        print(f"FPS: {fps}")

        print("------\n")

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
    raise

cv2.destroyAllWindows()
picam2.stop()