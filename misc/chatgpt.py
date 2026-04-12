import cv2
from picamera2 import Picamera2
import numpy as np
import time

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(
    main={"size": (640, 480)}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold (auto lighting adjustment)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours by area (largest first)
    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    # Blank image for visualization
    contour_img = np.zeros_like(frame)

    for i, contour in enumerate(sorted_contours):

        # Assign colors based on rank
        if i == 0:
            color = (0, 255, 0)      # Green (largest)
        elif i == 1:
            color = (255, 0, 0)      # Blue
        elif i == 2:
            color = (0, 0, 255)      # Red
        elif i == 3:
            color = (0, 255, 255)    # Yellow
        else:
            color = tuple(np.random.randint(0, 255, 3).tolist())

        cv2.drawContours(contour_img, [contour], -1, color, thickness=cv2.FILLED)

    # ---- Find centroid of largest contour (the line) ----
    if len(sorted_contours) > 0:
        largest = sorted_contours[0]
        M = cv2.moments(largest)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw centroid
            cv2.circle(contour_img, (cx, cy), 8, (255, 255, 255), -1)
            print("Line centroid:", cx, cy)

    cv2.imshow("Colored Contours", contour_img)

    if cv2.waitKey(1) == 27:
        break

picam2.stop()
picam2.close()
cv2.destroyAllWindows()