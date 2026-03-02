import cv2
from picamera2 import Picamera2
import time

picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

img = picam2.capture_array()

cv2.imshow("output", img)
cv2.waitKey(0)

picam2.stop()
picam2.close()