# it should always show output, while showing final verdict at the end of each frame
# i changed 8 to 5 for C and may be able to add more tolerance for container area if needed

import cv2
import numpy as np 

MIN_AREA = 3500
MAX_ASPECT_RATIO = 1.6

def stop():
    cv2.destroyAllWindows()

def detect_symbols_in_container(i, cnts, hrchy):
    arc_count = 0
    square_count = 0
    arrow_count = 0

    child_idx = hrchy[0][i][2] # first child of inner border

    if child_idx == -1:
        return "No idea"

    # keep on checking next child
    while True:
        c = cnts[child_idx]
        area = cv2.contourArea(c)
        if area > MIN_AREA * 0.075:
            rect = cv2.minAreaRect(c)
            w_rot, h_rot = rect[1]
            if w_rot != 0 and h_rot != 0:
                ar = max(w_rot, h_rot) / min(w_rot, h_rot)
                rect_area = w_rot * h_rot
                extent = area / rect_area if rect_area > 0 else 0
                corners = len(cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True))

                if ar > 4 or ar > 1.8 and extent < 0.5:
                    arc_count += 1
                if 4 <= corners <= 5 and 0.8 < extent < 1.1 and ar < 1.25:
                    square_count += 1
                if 12 <= corners <= 18 and 1.5 <= ar <= 2.0 and 0.45 <= extent <= 0.65:
                    arrow_count += 1

            #print(f"A:{area}, AR:{ar} E:{extent} corner:{corners}")

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

def detect_container(i, c, cnts, hrchy):
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
                    qr = 0

                    while gchild_idx != -1:
                        gc_curr = cnts[gchild_idx]
                        gc_area = cv2.contourArea(gc_curr)
                        total_area += gc_area
                        if gc_area > MIN_AREA and gc_area > largest_gc_area:
                            largest_gc = gc_curr
                            largest_gc_area = gc_area
                            sel_i = gchild_idx
                        if gc_area > MIN_AREA * 0.7:
                            qr += 1
                        gchild_idx = hrchy[0][gchild_idx][0] # next sibling

                    if qr >= 3:
                        return child_idx

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
            return sel_c, w_rot, h_rot, min_rect, child_area, child_idx

def detect_object(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, hrchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    answer = []
    containers = []

    cv2.imshow("frame", frame)
    cv2.imshow("close", closed)
    output = frame
    cv2.waitKey(1)

    # should already have hierarchy if a contour exists
    for i, c in enumerate(cnts):
        pred = "No idea"
        result = detect_container(i, c, cnts, hrchy)

        if result is not None: # found container
            if isinstance(result, (int, np.integer)): # no valid contours
                containers.append(result)
            else: # has valid contours
                sel_c, w_rot, h_rot, min_rect, inner_area, child_idx = result

                peri = cv2.arcLength(sel_c, True)
                approx = cv2.approxPolyDP(sel_c, 0.01 * peri, True)
                corners = len(approx)
                aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot) if min(w_rot, h_rot) > 0 else 0

                # Convex hull
                hull = cv2.convexHull(sel_c)
                hull_area = cv2.contourArea(hull)
                sel_area = cv2.contourArea(sel_c)
                solidity = sel_area / hull_area if hull_area > 0 else 0

                inner_area_ratio = sel_area / inner_area

                # Rotated extent
                extent = sel_area / (w_rot * h_rot) if w_rot*h_rot > 0 else 0

                ellipse_area_ratio = 0

                if solidity < 0.70:
                    if solidity < 0.55 and extent < 0.40:
                        pred = "Star"
                    else:
                        # pred = "Arrow"
                        x, y, w, h = cv2.boundingRect(sel_c)
                        bx = x + (w / 2.0)
                        by = y + (h / 2.0)
                        
                        M = cv2.moments(sel_c)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            dx = cx - bx
                            dy = cy - by
                            
                            #pred = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                            if abs(dx) > abs(dy):
                                pred = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                            else:
                                pred = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"
                elif corners == 4 and inner_area_ratio < 0.45:
                    if extent < 0.80:
                        pred = "Trapezium"
                    else:
                        pred = "Kite"
                elif aspect_ratio > 1.3 and solidity > 0.95:
                    pred = "Major Segment"
                elif corners == 8:
                    pred = "Octagon"
                else:
                    if sel_area > 6000:
                        # Ellipse area ratio
                        _, radius = cv2.minEnclosingCircle(sel_c)
                        circle_area = np.pi * radius * radius
                        ellipse_area_ratio = sel_area / circle_area if circle_area > 0 else 0
                        
                        if corners == 12 and aspect_ratio < 1.13: 
                            pred = "Plus"
                        elif ellipse_area_ratio < 0.55 and inner_area_ratio < 0.6:
                            pred = "Press Button"
                        elif ellipse_area_ratio < 0.8 and extent < 0.75 and solidity < 0.9:
                            pred = "3/4 Circle"
                        elif ellipse_area_ratio < 1.05:
                            pred = "Danger"
                    else:
                        pred = "No idea"

                if pred == "No idea":
                    containers.append(child_idx)
                else:
                    output = frame
                    box = cv2.boxPoints(min_rect)
                    box = np.intp(box)
                    cv2.drawContours(output, [sel_c], -1, (0, 255, 0), 2)
                    cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
                    cv2.putText(output, f"{pred}",#f"C:{corners} AR:{aspect_ratio:.2f} S:{solidity:.2f} E:{extent:.2f} R:{ellipse_area_ratio:.2f} A:{area:.2f}",
                                (int(min_rect[0][0]-min_rect[1][0]/2), int(min_rect[0][1]-10-min_rect[1][1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
                    answer.append(pred)
                    #print(sel_area)

        # jumps to here if no containers, then continues for loop
    cv2.imshow("Debug", output)

    for container in containers:
        pred = detect_symbols_in_container(container, cnts, hrchy)
        if pred != "No idea":
            answer.append(pred)

    if len(answer) == 0:
        #print("Final verdict: No shapes")
        return None
    else:
        #print(f"Final verdict: {answer}")
        return answer
   