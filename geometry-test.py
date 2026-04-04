import cv2
import numpy as np

# ===== SETTINGS =====
IMAGE_PATH = "./pictures-bryan/springprj-shapes.png"   # change this
AREA_THRESHOLD = 1500

# ===== LOAD IMAGE =====
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("Error: Image not found")
    exit()

resized = cv2.resize(img, None, fx=0.5, fy=0.5)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold (adaptive for robustness)
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    51, 5
)

# Morph close (clean noise)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# ===== FIND CONTOURS =====
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = resized.copy()

for c in contours:
    area = cv2.contourArea(c)

    if area < AREA_THRESHOLD:
        continue

    # ===== POLYGON APPROX =====
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    corners = len(approx)

    # ===== AXIS-ALIGNED BOUNDING BOX (for display only) =====
    x, y, w_box, h_box = cv2.boundingRect(c)

    # ===== ROTATED BOUNDING BOX (for features) =====
    rect = cv2.minAreaRect(c)
    (w_rot, h_rot) = rect[1]

    if w_rot > 0 and h_rot > 0:
        extent_rot = area / (w_rot * h_rot)
        aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)
    else:
        extent_rot = 0
        aspect_ratio = 0

    # ===== CONVEX HULL =====
    hull = cv2.convexHull(c)
    cv2.drawContours(output, [hull], -1, (0, 0, 255), 2)

    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # ===== DRAW =====
    cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

    # Axis-aligned box (blue)
    cv2.rectangle(output, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

    # Rotated box (optional but VERY useful)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(output, [box], 0, (255, 255, 0), 2)

    # ===== TEXT INFO =====
    text = f"C:{corners} E:{extent_rot:.2f} S:{solidity:.2f} AR:{aspect_ratio:.2f}"
    cv2.putText(output, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)

# ===== SHOW =====
cv2.imshow("Threshold", thresh)
cv2.imshow("Geometry Debug", output)
cv2.waitKey(0)
cv2.destroyAllWindows()