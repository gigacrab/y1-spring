import cv2
from picamera2 import Picamera2
import time
import numpy as np

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

while True: 
    frame = picam2.capture_array()
    cv2.imshow("frame", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array([13, 205, 169]), np.array([33, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([246, 205, 173]), np.array([255, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([0, 205, 173]), np.array([10, 255, 255]))
    color_mask = cv2.bitwise_or(red_mask1, red_mask2, yellow_mask)

    roi = frame
    im = frame

    #cv2.imshow("raw", im)
    imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", imgray)
    # 0 - values above this, assigned 255, the Otsu method adjusts according to lighting 
    # also idc about the ret
    _, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("thresh", thresh)

    # hierarchy -> [next, previous, first_child, parent]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((120, 720, 3), dtype=np.uint8)
    
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

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv2.line(im2, (cx, 0), (cx, 120), (0, 255, 255), 3)

        
            
    cv2.imshow("contours", im)
    if cv2.waitKey(1) == 27:
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()