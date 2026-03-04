import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()

if ret:
    cv2.imwrite('recycle.png', frame)
    print(r"C:\Users\WINDOWS 11\Documents\Documents\UNM\Year_1\Applied Electrical and Electronic Engineering\y1-spring\Symbols_png\recycle.png")

cap.release()
