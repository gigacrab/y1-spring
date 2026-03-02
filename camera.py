import cv2 as cv
from picamera2 import Picamera2
import time
import numpy as np

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

im = picam2.capture_array()

cv.imshow("raw", im)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
cv.imshow("gray", imgray)
ret, thresh = cv.threshold(imgray, 0, 127, 0)
cv.imshow("thresh", thresh)

# hierarchy -> [next, previous, first_child, parent]
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
im2 = np.zeros((1280, 720, 3), dtype=np.uint8)
cv.drawContours(im2, contours, -1, (255, 255, 255), 1)
cv.imshow("contours", im2)
while cv.waitKey(1) != 27:
    pass

picam2.stop()
picam2.close()