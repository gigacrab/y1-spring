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

# __main__ is the script that was passed to execute
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
last_error = 0
total_error = 0
flag = False
flag2 = False
#kp = 0.8
#ki = 0
#kd = 0

while True: 
    try: 
        time_marker = time.perf_counter()

        frame = picam2.capture_array()

        roi = frame[240:480, :]

        #cv2.imshow("raw", im)
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", imgray)
        # 0 - values above this, assigned 255, the Otsu method adjusts according to lighting 
        # also idc about the ret
        #_, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow("thresh", thresh)

        # hierarchy -> [next, previous, first_child, parent]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)
        
        if len(contours) > 0:
            
            if len(contours) > 1:
                # descending sorting using contourArea function
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                cnt = sorted_contours[0]
                cv2.drawContours(im2, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
                cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
                
            else:
                cnt = contours[0]
                cv2.drawContours(im2, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
                
            M = cv2.moments(cnt)

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
            diff_error = (error - last_error) / elapsed_time

            if not flag:
                diff_error = 0
                flag = True
            pid = kp * error + ki * total_error + kd * diff_error
            
            last_error = error

            #left_pwm = 0.8 + pid
            #right_pwm = 0.78 - pid
            left_pwm = base_speed + pid
            right_pwm = base_speed - pid

            clamped_left_pwm = clamp(left_pwm, -1, 1)
            clamped_right_pwm = clamp(right_pwm, -1, 1)

            movement.move(clamped_left_pwm, clamped_right_pwm)
        
        else:
            #movement.move(0, 0)
            #break
            pid = error * 1000
            left_pwm = base_speed + pid
            right_pwm = base_speed - pid
            print("we cannot find contours")

            clamped_left_pwm = clamp(left_pwm, -1, 1)
            clamped_right_pwm = clamp(right_pwm, -1, 1)

            movement.move(clamped_left_pwm, clamped_right_pwm)

        #cv2.imshow("contours", im2)

        '''
        if cv2.waitKey(1) == 27:
            movement.move(0, 0)
            break
        '''
    except Exception as e:
        print(f"Error has occured - {e}")
        break
        
movement.pi.stop()
picam2.stop()
picam2.close()
cv2.destroyAllWindows()