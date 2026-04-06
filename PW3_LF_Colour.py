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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi = frame[240:480, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Sample a 10x10 patch from the centre of the ROI
        sample = hsv[110:120, 310:320]
        print(f"HSV centre sample: {sample.mean(axis=(0,1)).astype(int)}")
        #red:
        red_lower = np.array([111, 49, 8])
        red_upper = np.array([131, 255, 220])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        #yellow:
        yellow_lower = np.array([85, 14, 22])
        yellow_upper = np.array([105, 170, 178])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

        combined_colour_mask = cv2.bitwise_or(red_mask, yellow_mask)
        colour_contour, _ = cv2.findContours(combined_colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid_colour_contour = None
        im2 = np.zeros((240, 640, 3), dtype=np.uint8)
        
        if len(colour_contour) > 0:
            colour_contour = sorted(colour_contour, key=cv2.contourArea, reverse=True)
            for cnt in colour_contour:
                cnt_area = cv2.contourArea(cnt)
                if 7500 <= cnt_area <= 40000:
                    valid_colour_contour = cnt
                    break
        
        line_contour = None
            
        if valid_colour_contour is not None:
            line_contour = valid_colour_contour
            cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
        else:
            imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            black_contour, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = []
            filtered_contour_areas = []

            if len(black_contour) > 0:
                contour_areas = [cv2.contourArea(cnt) for cnt in black_contour]
                for i, cnt_a in enumerate(contour_areas):
                    if 7500 <= cnt_a <= 40000:
                        filtered_contours.append(black_contour[i])
                        filtered_contour_areas.append(contour_areas[i])
            
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


        if line_contour is not None:
            M = cv2.moments(line_contour)

            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
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
                    diff_error = 0
                
                pid = kp * error + ki * total_error + kd * diff_error
                
                last_error = error

        else:
            print(f"we cannot find contours {getSign(last_error)}")
            pid = getSign(last_error) * 2

        left_pwm = base_speed + pid
        right_pwm = base_speed - pid

        clamped_left_pwm = clamp(left_pwm, -1, 1)
        clamped_right_pwm = clamp(right_pwm, -1, 1)

        movement.move(clamped_left_pwm, clamped_right_pwm)

        '''
        cv2.imshow("contours", im2)
        
        if cv2.waitKey(1) == 27:
            movement.move(0, 0)
            break
        '''
    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occured - {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
#cv2.destroyAllWindows()