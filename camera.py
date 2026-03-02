import cv2
from picamera2 import Picamera2
import time
import numpy as np

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

while True: 
    im = picam2.capture_array()

    #cv2.imshow("raw", im)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", imgray)
    # 127 - values above this, assigned 255
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("thresh", thresh)

    # hierarchy -> [next, previous, first_child, parent]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # descending sorting using contourArea function
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)


    cv2.drawContours(im2, sorted_contours[0], -1, (0, 255, 0), thickness=cv2.FILLED)
    cv2.drawContours(im2, sorted_contours[1:], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.imshow("contours", im2)
    
    if cv2.waitKey(1) == 27:
        break

picam2.stop()
picam2.close()