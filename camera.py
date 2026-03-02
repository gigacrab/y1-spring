import cv2 as cv
from picamera2 import Picamera2
import time

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

im = picam2.capture_array()

#cv2.imshow("output", img)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.imshow(im2)
cv.waitKey(0)

picam2.stop()
picam2.close()