import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys
import os

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
                if cnt_a > 1000:
                    count += 1
                if cnt_a >= 8500 and cnt_a <= 40000:
                    filtered_contours.append(contours[i])
                    filtered_contour_areas.append(contour_areas[i])
            print(count)

            if not flag and count >= 2 and count <= 4:
                movement.move(0, 0)
                movement.move(-0.5, -0.5)
                time.sleep(0.01)
                hello = picam2.capture_array()
                cv2.imshow("hello", hello)
                cv2.imshow("real", frame)
                for _ in range(100):
                    cv2.waitKey(10)
                # do checking
                time.sleep(10)
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
                total_error += error * elapsed_time

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first = False
                    
                pid = kp * error + ki * total_error + kd * diff_error
                
                last_error = error

            else:
                print(f"we cannot find contours {getSign(last_error)}")
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