import cv2
from picamera2 import Picamera2
import time
import numpy as np

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

while True: 
    im = picam2.capture_array()

    roi = im[360:480, :]

    #cv2.imshow("raw", im)
    imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", imgray)
    # 0 - values above this, assigned 255, the Otsu method adjusts according to lighting 
    # also idc about the ret
    _, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("thresh", thresh)

    # hierarchy -> [next, previous, first_child, parent]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((480, 720, 3), dtype=np.uint8)
    
    if len(contours) > 0:
           
        if len(contours) > 1:
            # descending sorting using contourArea function
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cnt = sorted_contours[0]
            cv2.drawContours(im2, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
            
        else:
            cnt = contours
            cv2.drawContours(im2, [cnt], -1, (0, 255, 0), thickness=cv2.FILLED)
            
        M = cv2.moments(cnt)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv2.line(im2, (cx, 360), (cx, 480), (255, 0, 0), 1)
            
    cv2.imshow("contours", im2)
    if cv2.waitKey(1) == 27:
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()