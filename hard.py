import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys
import os

def shape_rec(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_processed = clahe.apply(gray)  # FIX: Apply lighting fix BEFORE Phase 1!
    blurred = cv2.GaussianBlur(gray_processed, (5, 5), 0)
    
    best_match = None

    # ==========================================
    # PHASE 1: GEOMETRY FIRST
    # ==========================================
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(cnts):
        if cv2.contourArea(c) > 1500: 
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            corners = len(approx)
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            area = cv2.contourArea(c)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            
            live_moments = cv2.HuMoments(cv2.moments(c)).flatten()
            lowest_diff = 0.1
            geom_match = None
            
            for name, master_dna in templates_npy.items():
                diff = np.sum(np.abs(live_moments - master_dna))
                if diff < lowest_diff:
                    lowest_diff = diff
                    geom_match = name
            if geom_match in ["Plus", "Kite"]:
                geom_match = "Kite" if corners < 8 else "Plus"
            if geom_match:
                # Calculate 'extent' to defeat the QR Square
                x, y, w, h = cv2.boundingRect(c)
                box_area = w * h
                extent = area / float(box_area) if box_area > 0 else 0

                if geom_match == "Star":
                    # Rejects chunky squares!
                    if solidity > 0.6 or corners < 8:
                        geom_match = None
                elif geom_match == "Octagon":
                    # FLICKER FIX: Relaxed corners to 5 to forgive camera blur!
                    if corners < 5 or solidity < 0.75:
                        geom_match = None
                elif geom_match == "Kite":
                    # THE EXTENT SHIELD: A QR square fills the box (Extent > 0.75).
                    if extent > 0.75:
                        geom_match = None # It's a QR block! Reject!
                elif geom_match == "3/4 Circle":
                    # FLICKER FIX: Accept 5-7 corners to forgive camera blur!
                    if solidity > 0.95:
                        geom_match = None
            if geom_match == "Arrow":
                if not (6 <= corners <= 9) or solidity < 0.55:
                    geom_match = None  
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
                            geom_match = "Arrow (RIGHT)" if dx > 0 else "Arrow (LEFT)"
                        else:
                            geom_match = "Arrow (DOWN)" if dy > 0 else "Arrow (UP)"

            if geom_match:
                best_match = geom_match
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break 
    if best_match is not None:
        return best_match
    else:
        return "unknown"


def clamp(value, min, max):
    if value > max:
        return max
    elif value < min:
        return min
    return value

# returns 1 if positive, -1 if negative
def getSign(n):
    return (n > 0) - (n < 0)

# __main__ is the script that was passed to execute
# config from day 1 - 0.4 1.4 0.01 0.2
if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp = float(sys.argv[2])
        ki = float(sys.argv[3])
        kd = float(sys.argv[4])
    else:
        raise Exception("Didn't input appropriate variables")

# Picamera2 init
picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

# PID
# error and diff_error are assigned
error = 0
total_error = 0
last_error = 0
diff_error = 0
first = True
#kp = 1.4
#ki = 0.01
#kd = 0.2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

npy_path = '/home/jaydenbryan/Project/Symbols_npy/'
template_files_npy = {
    "Arrow": "arrow.npy",
    "3/4 Circle": "circle34.npy",
    "Major Segment": "circlemajorsegment.npy",
    "Plus": "plus.npy",
    "Star": "star.npy",
    "Trapezium": "trapezium.npy",
    "Octagon": "octagon.npy",
    "Kite": "kite.npy"
}

templates_npy = {}
for name, filename in template_files_npy.items():
    try:
        templates_npy[name] = np.load(os.path.join(npy_path, filename))
    except FileNotFoundError:
        print(f"Warning: Missing DNA file {filename}")

time_marker = time.perf_counter()
time_cool = 0
flag = False
while True: 
    try: 
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # line following
        roi = frame[240:480, :]

        #cv2.imshow("raw", im)
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # we now try gaussian blur
        imgray = cv2.GaussianBlur(imgray, (5,5), 0)
        #cv2.imshow("gray", imgray)
        # 0 - values above this, assigned 255, the Otsu method adjusts according to lighting
        # however the Otsu method wasn't that good because it'd always find a region of threshold
        # also idc about the ret
        ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #_, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow("thresh", thresh)

        # hierarchy -> [next, previous, first_child, parent]
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.drawContours(im2, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        count = 0
        if len(contours) > 0:
            contour_areas = [cv2.contourArea(cnt) for cnt in contours]
            filtered_contours = []
            filtered_contour_areas = []

            # areas between 7500 to 40000 are accepted
            for i, cnt_a in enumerate(contour_areas):
                if cnt_a > 2500:
                    count += 1
                if cnt_a >= 8500 and cnt_a <= 40000:
                    filtered_contours.append(contours[i])
                    filtered_contour_areas.append(contour_areas[i])

            if not flag and count >= 2 and count <= 4:
                movement.move(0, 0)
                movement.move(-0.5, -0.5)
                time.sleep(0.3)
                movement.move(0, 0)
                time.sleep(1)
                hello = picam2.capture_array()
                cv2.imshow("hello", hello)
                #cv2.imshow("real", frame)
                for _ in range(100):
                    cv2.waitKey(10)
                # do checking
                #time.sleep(2)
                print(f"The shape is {shape_rec(hello)}")
                time_cool = time.perf_counter()
                first = True
                flag = True
            current_time = time.perf_counter()
            if (current_time - time_cool) > 2:
                flag = False
                
            # here we have the ACTUAL contours, if none, maximum error
            if len(filtered_contours) > 0 and ret < 180:
                if len(filtered_contours) > 1:
                    zipped_pairs = zip(filtered_contour_areas, filtered_contours)
                    # this sorts by the first element
                    sorted_pairs = sorted(zipped_pairs, reverse=True)

                    _, sorted_contours = zip(*sorted_pairs)
                    line_contour = sorted_contours[0]
                    cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
                else:
                    line_contour = filtered_contours[0]
                
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

                M = cv2.moments(line_contour)

                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                #cy = int(M['m01']/M['m00'])

                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                # pwm - 80 for left, 78 for right 
                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

                time_marker = time.perf_counter()
                
                # error is normalized
                error = (320 - cx) / 320    

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                    total_error += error * elapsed_time
                else:
                    first = False
                    
                pid = kp * error + ki * total_error + kd * diff_error
                
                last_error = error

            else:
                #print(f"we cannot find contours {getSign(last_error)}")
                pid = getSign(last_error)

            left_pwm = base_speed + pid
            right_pwm = base_speed - pid

            clamped_left_pwm = clamp(left_pwm, -1, 1)
            clamped_right_pwm = clamp(right_pwm, -1, 1)

            movement.move(clamped_left_pwm, clamped_right_pwm)

            cv2.imshow("contours", im2)
            
            if cv2.waitKey(1) == 27:
                movement.move(0, 0)
                break
        else:
            # this is unlikely but
            continue     
    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occured - {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()