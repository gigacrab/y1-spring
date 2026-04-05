import cv2
import numpy as np 
from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

MIN_AREA = 3000
MAX_ASPECT_RATIO = 1.6

try:
    while True:
        frame = picam2.capture_array()
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 8 # blocksize 51, must be odd
            # C 5 - value to minus from obtained threshold
        )

        # morph close
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cnts, hrchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pred = ""
        # should already have hierarchy if a contour exists
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            sel_c = c
            selected = False
            w_rot, h_rot = 0, 0 
            aspect_ratio = 0
            ellipse_area_ratio = 0

            '''
            if hrchy[0][i][3] == -1:
                hello = hrchy[0][i][2]
                if hello != -1:
                    while hrchy[0][hello][0] != -1:
                        print(f"hr:{hrchy[0][hello]}{cv2.contourArea(cnts[hrchy[0][i][0]])}")
                        hello = hrchy[0][hello][0]
            '''

            # ===== Container check =====
            child_idx = hrchy[0][i][2]
            if child_idx != -1 and hrchy[0][i][3] == -1:
                child_area = cv2.contourArea(cnts[child_idx])
                hollow_ratio = child_area / area if area > 0 else 0

                if hollow_ratio > 0.9:

                    print(f"hollow {i}")
                    # Also verify the parent looks like a rectangle via extent
                    rect = cv2.minAreaRect(c)
                    w_rot, h_rot = rect[1]
                    if w_rot == 0 or h_rot == 0:
                        continue
                    rect_area = w_rot * h_rot
                    extent = area / rect_area if rect_area > 0 else 0
                    aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot) if min(w_rot, h_rot) > 0 else 0

                    if extent > 0.85:  # parent is rectangle-like, confirms it's a container
                        # Find largest valid grandchild instead of assuming first
                        best_gc = None
                        best_gc_area = 0
                        gchild_idx = hrchy[0][child_idx][2]  # first grandchild
                        index = i
                        total_area = 0

                        while gchild_idx != -1:
                            gc_candidate = cnts[gchild_idx]
                            gc_area = cv2.contourArea(gc_candidate)
                            total_area += gc_area
                            if gc_area > MIN_AREA and gc_area > best_gc_area:
                                best_gc = gc_candidate
                                best_gc_area = gc_area
                                index = gchild_idx
                            gchild_idx = hrchy[0][gchild_idx][0]  # next sibling

                        if best_gc is not None:
                            rect = cv2.minAreaRect(best_gc)
                            w_rot, h_rot = rect[1]
                            if w_rot == 0 or h_rot == 0:
                                continue
                            sel_c = best_gc
                            selected = True
                            print(f"container for {index}")

                        if total_area > MIN_AREA:
                            continue
                        '''
                        gchild_idx = hrchy[0][child_idx][2]
                        if gchild_idx != -1:
                            gc = cnts[gchild_idx]
                            if cv2.contourArea(gc) > MIN_AREA: # already pretty confirmed ngl, this might filter some out, also considering we are only using the first child
                                # Recompute rect for the grandchild (this is what we'll classify)
                                rect = cv2.minAreaRect(gc)
                                w_rot, h_rot = rect[1]
                                if w_rot == 0 or h_rot == 0:
                                    continue
                                sel_c = gc
                                selected = True
                        '''

            # ===== Aspect ratio check (fallback) =====
            if not selected and hrchy[0][i][3] == -1:
                rect = cv2.minAreaRect(sel_c)
                w_rot, h_rot = rect[1]
                if w_rot == 0 or h_rot == 0:
                    continue
                aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                selected = True
                print(f"hello for {i}")

            if not selected:
                continue
            
            # ===== Now process selected_contour normally =====
            peri = cv2.arcLength(sel_c, True)
            approx = cv2.approxPolyDP(sel_c, 0.01 * peri, True)
            corners = len(approx)
            aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot) if min(w_rot, h_rot) > 0 and aspect_ratio != 0 else 0

            # Convex hull
            hull = cv2.convexHull(sel_c)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(sel_c) / hull_area if hull_area > 0 else 0

            # Rotated extent
            extent = cv2.contourArea(sel_c) / (w_rot * h_rot) if w_rot*h_rot > 0 else 0

            # Determine concave ones first - arrow and star via solidity (arrow has larger extent and solidity)
            # 4 corners - kite and trapezium (kite has larger extent)
            # 1.44 aspect ratio - major segment
            # 8 corners - octagon
            # 12 corners and aspect ratio around 1.006 (1.176 is ¾ circle) - plus
            # [calc ellipse area ratio]
            # Ellipse area ratio - 0.75 for ¾ circle and 1.00 for danger

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
            elif aspect_ratio > 1.3:
                pred = "Major Segment"
            elif corners == 8:
                pred = "Octagon"
            elif corners == 12 and aspect_ratio < 1.05:
                pred = "Plus"
            else:
                # Ellipse area ratio
                (xc, yc), radius = cv2.minEnclosingCircle(sel_c)
                circle_area = np.pi * radius * radius
                ellipse_area_ratio = cv2.contourArea(sel_c) / circle_area if circle_area > 0 else 0

                if ellipse_area_ratio < 0.8:
                    pred = "3/4 Circle"
                elif ellipse_area_ratio < 1.05:
                    pred = "Danger"
                else:
                    pred = "No Idea"
            
            print(f"Prediction: {pred}")
            
            # ===== DRAW / DEBUG =====
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(output, [sel_c], -1, (0, 255, 0), 2)
            cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
            cv2.putText(output, f"{pred}",#f"C:{corners} AR:{aspect_ratio:.2f} S:{solidity:.2f} E:{extent:.2f} R:{ellipse_area_ratio:.2f} A:{area:.2f}",
                        (int(rect[0][0]-rect[1][0]/2), int(rect[0][1]-10-rect[1][1]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            

            
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Geometry Debug", output)

            print(f"P:{hrchy[0][i]} C:{corners} AR:{aspect_ratio:.2f} S:{solidity:.2f} E:{extent:.2f} R:{ellipse_area_ratio:.2f} A:{area:.2f} AC:{cv2.contourArea(cnts[hrchy[0][i][2]])}")

        '''
        for i, c in enumerate(cnts):
            print(f"{i} hr:{hrchy[0][i]}, a:{cv2.contourArea(c)}")
        '''
        

        if cv2.waitKey(1) == ord('q'):
            break
            
        print("\n")

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
    raise  # re-raise so you see the full traceback
finally:
    cv2.destroyAllWindows()
    picam2.stop()