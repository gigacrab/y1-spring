import os
# Suppress Qt font warnings (cosmetic only, doesn't affect detection)
os.environ['QT_LOGGING_RULES'] = '*=false'

import cv2
import numpy as np
from picamera2 import Picamera2
import time


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL SYMBOL DETECTION
#
# The key insight: these three symbols can't be identified from a single
# contour's geometry. They're recognised by the *pattern of multiple contours
# appearing together* — arcs for fingerprint, finder squares for QR, bent
# arrows for recycle.
#
# The previous version only checked top-level contours, which worked on the
# raw reference images. In real life the symbols sit on a card, making them
# grandchildren in the hierarchy — so we need to check whatever level we're
# currently examining, not just the top level.
#
# Solution: split into two functions.
#   check_special_in_group() — pure geometry check on any list of indices
#   detect_special_symbols() — calls the above with top-level indices
#                              (used as a pre-scan before the per-contour loop)
# The container logic inside the main loop calls check_special_in_group()
# directly with its grandchild list, catching symbols that are on cards.
# ─────────────────────────────────────────────────────────────────────────────

def check_special_in_group(cnts, hrchy, indices, min_area):
    """
    Given any list of contour indices, count characteristic features and
    return a label if a special symbol pattern is detected, else None.
    This works at any hierarchy depth — top-level, grandchildren, etc.
    """
    arc_count    = 0   # fingerprint: elongated open strokes
    square_count = 0   # QR icon:     near-perfect solid squares
    arrow_count  = 0   # recycle:     medium-polygon bent arrows

    for i in indices:
        c    = cnts[i]
        area = cv2.contourArea(c)
        if area < min_area * 0.3:
            continue

        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        ar       = max(w, h) / min(w, h)
        hull_a   = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_a if hull_a > 0 else 0
        corners  = len(cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True))
        has_child = hrchy[0][i][2] != -1

        # Fingerprint arc: elongated (AR > 1.5), simple shape (<=12 corners to
        # exclude recycle's 14-corner arrows), no hollow interior (no child).
        if ar > 1.5 and corners <= 12 and solidity > 0.3 and not has_child:
            arc_count += 1

        # QR finder square: 4 corners, almost perfectly square, fully solid.
        # cv2.contourArea doesn't subtract child holes, so the outer dark
        # square's area is the full square area, giving solidity ~1.0.
        if corners == 4 and ar < 1.15 and solidity > 0.9:
            square_count += 1

        # Recycle arrow: the bent body + arrowhead creates ~14 corners,
        # the curve gives moderate AR, and partial fill gives medium solidity.
        if 10 <= corners <= 18 and 1.4 <= ar <= 2.2 and 0.55 <= solidity <= 0.80:
            arrow_count += 1

    # Recycle must be checked before fingerprint: recycle arrows (AR ~1.75)
    # also satisfy the arc condition (AR > 1.5), so order prevents false hits.
    if arrow_count >= 3:
        return "Recycle"
    if arc_count >= 4:
        return "Fingerprint"
    if square_count >= 3:
        return "QR Code"
    return None


def detect_special_symbols(cnts, hrchy, min_area):
    """
    Frame-level pre-scan using only top-level contours.
    Catches special symbols shown directly without a surrounding card border.
    """
    top_level = [i for i in range(len(cnts)) if hrchy[0][i][3] == -1]
    return check_special_in_group(cnts, hrchy, top_level, min_area)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────────────────────────────────────
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

MIN_AREA         = 3000
MAX_ASPECT_RATIO = 1.6
prev_frame_time  = 0

try:
    while True:
        frame  = picam2.capture_array()
        output = frame.copy()
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51, 8
        )

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, hrchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pred = ""

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1 — Frame-level special symbol pre-scan (no card border)
        # If the symbol is held up directly without a surrounding card, the
        # components are top-level and this catches them immediately.
        # ─────────────────────────────────────────────────────────────────────
        if cnts:
            special = detect_special_symbols(cnts, hrchy, MIN_AREA)
            if special:
                pred = special
                cv2.putText(output, pred, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("Threshold", thresh)
                cv2.imshow("Geometry Debug", output)
                print(f"Special (no card): {pred}\n")
                if cv2.waitKey(1) == ord('q'):
                    break
                continue

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2 — Per-contour geometric classification
        # ─────────────────────────────────────────────────────────────────────
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue

            # Only process top-level contours here — children are handled via
            # the container logic below, not as independent classification targets.
            if hrchy[0][i][3] != -1:
                continue

            sel_c              = None
            w_rot, h_rot       = 0, 0
            aspect_ratio       = 0
            ellipse_area_ratio = 0
            min_rect           = None
            child_idx          = hrchy[0][i][2]

            # ── Container check ───────────────────────────────────────────────
            if child_idx != -1:
                child_area   = cv2.contourArea(cnts[child_idx])
                hollow_ratio = child_area / area if area > 0 else 0

                if hollow_ratio > 0.85:
                    print(f"Hollow container at: {i}")

                    min_rect     = cv2.minAreaRect(c)
                    w_rot, h_rot = min_rect[1]
                    if w_rot == 0 or h_rot == 0:
                        continue
                    extent       = area / (w_rot * h_rot)
                    aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)

                    if extent > 0.85:
                        # Collect all grandchild indices first
                        grandchild_indices = []
                        gchild_idx = hrchy[0][child_idx][2]
                        while gchild_idx != -1:
                            grandchild_indices.append(gchild_idx)
                            gchild_idx = hrchy[0][gchild_idx][0]

                        # ── KEY FIX ───────────────────────────────────────────
                        # Check grandchildren for special symbols BEFORE trying
                        # to find a single geometric shape among them.
                        # The old code skipped this entirely — it only checked
                        # top-level contours in Stage 1, so symbols on cards
                        # (whose components are grandchildren) were never caught.
                        special = check_special_in_group(
                            cnts, hrchy, grandchild_indices, MIN_AREA
                        )
                        if special:
                            pred = special
                            box  = np.intp(cv2.boxPoints(min_rect))
                            cv2.drawContours(output, [box], 0, (0, 200, 255), 2)
                            cv2.putText(
                                output, pred,
                                (int(min_rect[0][0] - min_rect[1][0] / 2),
                                 int(min_rect[0][1] - 10 - min_rect[1][1] / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2
                            )
                            print(f"Special (on card): {pred}")
                            break  # found our symbol, stop checking other contours

                        # No special symbol — find the largest single grandchild
                        # to classify as a geometric shape
                        largest_gc      = None
                        largest_gc_area = 0
                        total_area      = 0
                        sel_i           = i

                        for gc_idx in grandchild_indices:
                            gc_area = cv2.contourArea(cnts[gc_idx])
                            total_area += gc_area
                            if gc_area > MIN_AREA and gc_area > largest_gc_area:
                                largest_gc      = cnts[gc_idx]
                                largest_gc_area = gc_area
                                sel_i           = gc_idx

                        if largest_gc is not None:
                            sel_c    = largest_gc
                            min_rect = cv2.minAreaRect(sel_c)
                            w_rot, h_rot = min_rect[1]
                            if w_rot == 0 or h_rot == 0:
                                continue
                            print(f"Contained shape for: {sel_i}")
                        elif total_area > MIN_AREA:
                            # Complex interior, no single classifiable shape
                            continue

            # ── Fallback: no container, classify the contour itself ───────────
            if sel_c is None:
                min_rect     = cv2.minAreaRect(c)
                w_rot, h_rot = min_rect[1]
                if w_rot == 0 or h_rot == 0:
                    continue
                aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                sel_c = c   # was missing in original — caused NoneType crash
                print(f"No container, took: {i} instead")

            if sel_c is None:
                continue

            # ── Feature extraction ────────────────────────────────────────────
            peri         = cv2.arcLength(sel_c, True)
            approx       = cv2.approxPolyDP(sel_c, 0.01 * peri, True)
            corners      = len(approx)
            aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot) \
                           if min(w_rot, h_rot) > 0 and aspect_ratio != 0 else 0

            hull      = cv2.convexHull(sel_c)
            hull_area = cv2.contourArea(hull)
            solidity  = cv2.contourArea(sel_c) / hull_area if hull_area > 0 else 0
            extent    = cv2.contourArea(sel_c) / (w_rot * h_rot) \
                        if w_rot * h_rot > 0 else 0

            # ── Classification ────────────────────────────────────────────────
            if solidity < 0.70:
                pred = "Star" if (solidity < 0.55 and extent < 0.40) else "Arrow"
            elif corners == 4:
                pred = "Trapezium" if extent < 0.80 else "Kite"
            elif aspect_ratio > 1.3:
                pred = "Major Segment"
            elif corners == 8:
                pred = "Octagon"
            elif corners == 12 and aspect_ratio < 1.05:
                pred = "Plus"
            else:
                (xc, yc), radius   = cv2.minEnclosingCircle(sel_c)
                circle_area        = np.pi * radius * radius
                ellipse_area_ratio = cv2.contourArea(sel_c) / circle_area \
                                     if circle_area > 0 else 0
                if ellipse_area_ratio < 0.65:
                    pred = "Press Button"
                elif ellipse_area_ratio < 0.8:
                    pred = "3/4 Circle"
                elif ellipse_area_ratio < 1.05:
                    pred = "Danger"
                else:
                    pred = "No Idea"

            print(f"Prediction: {pred}")

            # ── Draw results ──────────────────────────────────────────────────
            box = np.intp(cv2.boxPoints(min_rect))
            cv2.drawContours(output, [sel_c], -1, (0, 255, 0), 2)
            cv2.drawContours(output, [box], 0, (255, 0, 0), 2)
            cv2.putText(
                output, pred,
                (int(min_rect[0][0] - min_rect[1][0] / 2),
                 int(min_rect[0][1] - 10 - min_rect[1][1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

            new_frame_time  = time.perf_counter()
            time_diff       = new_frame_time - prev_frame_time
            fps             = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_frame_time = new_frame_time

            child_area_dbg = cv2.contourArea(cnts[child_idx]) if child_idx != -1 else 0
            print(f"FPS:{fps:.1f} P:{hrchy[0][i]} C:{corners} AR:{aspect_ratio:.2f} "
                  f"S:{solidity:.2f} E:{extent:.2f} R:{ellipse_area_ratio:.2f} "
                  f"A:{area:.2f} AC:{child_area_dbg:.0f}")

        cv2.imshow("Threshold", thresh)
        cv2.imshow("Geometry Debug", output)

        if cv2.waitKey(1) == ord('q'):
            break

        print("\n")

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"Error: {e}")
    raise

cv2.destroyAllWindows()
picam2.stop()