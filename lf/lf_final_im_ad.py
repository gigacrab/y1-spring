import cv2
from picamera2 import Picamera2
import time
import numpy as np
import movement
import sys

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

# others init
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

while True: 
    try: 
        time_marker = time.perf_counter()

        frame = picam2.capture_array()

        roi = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_processed = clahe.apply(gray)  # FIX: Apply lighting fix BEFORE Phase 1!
        blurred = cv2.GaussianBlur(gray_processed, (5, 5), 0)
        
        best_match = None

        # ==========================================
        # PHASE 1: GEOMETRY FIRST
        # ==========================================
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 8)
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        cnts_shape, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        thresh_roi = thresh[240:480, :]
        cnts_line, hierarchy = cv2.findContours(thresh_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)
        cv2.drawContours(im2, cnts_line, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        if len(cnts_line) > 0:
            filtered_contours = []
            filtered_contour_areas = []

            for i, c in enumerate(cnts_line):
                c_area = cv2.contourArea(c)
                if c_area >= 11000 and c_area <= 40000:
                    c_arc = cv2.arcLength(c, True)
                    epsilon = 0.2 * c_arc
                    c_approx = cv2.approxPolyDP(c, epsilon, True)
                    c_approx_arc = cv2.arcLength(c_approx, True)
                    
                    if c_approx_arc != 0: smoothness = c_arc / c_approx_arc 
                    if smoothness > 0.95:
                        filtered_contours.append(c)
                        #print(c_area)
                        filtered_contour_areas.append(c_area)
                    
                    '''
                    c_hull = cv2.convexHull(c)
                    c_hull_area = cv2.contourArea(c_hull)
                    solidity = c_area / c_hull_area

                    if solidity > 0.85:
                        filtered_contours.append(c)
                        filtered_contour_areas.append(c_area)'''
            #print(len(filtered_contours))
            
            # here we have the ACTUAL contours, if none, maximum error
            if len(filtered_contours) > 0:
                
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


            print(error)
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