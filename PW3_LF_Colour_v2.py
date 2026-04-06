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

def getSign(n):
    return (n > 0) - (n < 0)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Didn't input appropriate variables")

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

# PID state
error       = 0
total_error = 0
last_error  = 0
diff_error  = 0
first       = True
pid         = 0

# Memory / state tracking
# "following_colour" — robot is on the colour line
# "search"          — colour line lost, turning to recover black line
# "following_black" — normal black line following
robot_state     = "following_black"
black_line_side = None   # "left" or "right", set whenever both lines visible
SEARCH_SPEED    = 0.4    # fixed turning magnitude during search mode
TURN_90_SPEED   = 0.5

while True:
    try:
        time_marker = time.perf_counter()

        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        roi   = frame[240:480, :]
        im2   = np.zeros((240, 640, 3), dtype=np.uint8)

        # ── HSV sample (debug) ────────────────────────────────────────────────
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sample = hsv[110:120, 310:320]
        print(f"HSV centre sample: {sample.mean(axis=(0,1)).astype(int)}")

        # ── Colour detection ──────────────────────────────────────────────────
        red_lower    = np.array([111, 50, 180])
        red_upper    = np.array([131, 185, 230])
        yellow_lower = np.array([ 85, 100, 205])
        yellow_upper = np.array([105, 255, 255])

        combined_colour_mask = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower,    red_upper),
            cv2.inRange(hsv, yellow_lower, yellow_upper)
        )

        colour_cnts, _ = cv2.findContours(combined_colour_mask, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
        valid_colour_contour = None
        colour_cx            = None
        colour_is_perpendicular = False

        if colour_cnts:
            for cnt in sorted(colour_cnts, key=cv2.contourArea, reverse=True):
                if 7500 <= cv2.contourArea(cnt) <= 40000:
                    valid_colour_contour = cnt
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        colour_cx = int(M['m10'] / M['m00'])
                    rect = cv2.minAreaRect(cnt)
                    rw, rh = rect[1]                 # rotated rect dimensions
                    if rw > 0 and rh > 0:
                        longer  = max(rw, rh)
                        shorter = min(rw, rh)
                        # ratio > 2.5 means the contour is at least 2.5x
                        # longer in one axis — a clear perpendicular line.
                        # Tune this threshold if needed.
                        if longer / shorter > 2.5 and rw > rh:
                            colour_is_perpendicular = True
                    # ─────────────────────────────────────────────────────────
                    break

        # ── Black line detection (always computed for memory update) ──────────
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_SIMPLE)

        valid_black_contour = None
        black_cx            = None
        filtered = []

        if black_cnts and ret < 180:
            filtered = [(cv2.contourArea(c), c) for c in black_cnts
                        if 7500 <= cv2.contourArea(c) <= 40000]
            if filtered:
                filtered.sort(reverse=True)
                valid_black_contour = filtered[0][1]
                M = cv2.moments(valid_black_contour)
                if M['m00'] != 0:
                    black_cx = int(M['m10'] / M['m00'])

        # ── Memory update: save black line side whenever both are visible ─────
        if colour_cx is not None and black_cx is not None:
            black_line_side = "right" if colour_cx < black_cx else "left"
            print(f"Memory updated: black line is to the {black_line_side}")

        # ── State transitions ─────────────────────────────────────────────────
        if robot_state == "following_black":
            if valid_colour_contour is not None:
                if colour_is_perpendicular:
                    robot_state = "turning_90"     # ← ADD branch
                    print("State → turning_90 (perpendicular colour line)")
                else:
                    robot_state = "following_colour"
                    print("State → following_colour")

        elif robot_state == "following_colour":
            if valid_colour_contour is None:
                # Colour line just ended — enter search only if we have a memory
                if black_line_side is not None:
                    robot_state = "search"
                    print(f"State → search (will turn {black_line_side})")
                else:
                    # No memory yet, fall back to normal black following
                    robot_state = "following_black"
                    print("State → following_black (no memory)")

        elif robot_state == "search":
            if valid_black_contour is not None:
                # Found the black line — resume normal following
                robot_state = "following_black"
                first       = True   # reset derivative term to avoid spike
                total_error = 0
                print("State → following_black (black line recovered)")
        elif robot_state == "turning_90":
            if valid_colour_contour is not None and not colour_is_perpendicular:
                # The perpendicular line is now running along our direction —
                # we've turned enough, follow it normally
                robot_state = "following_colour"
                print("State → following_colour (turn complete)")
            elif valid_colour_contour is None and valid_black_contour is None:
                # Both lines lost mid-turn — keep turning, enter search
                robot_state = "search"
                print("State → search (lost both lines during turn)")

        # ── Motor commands based on current state ─────────────────────────────
        if robot_state == "search":
            # Hard-turn toward the remembered side, ignore PID entirely
            if black_line_side == "right":
                movement.move( SEARCH_SPEED, -SEARCH_SPEED)
            else:
                movement.move(-SEARCH_SPEED,  SEARCH_SPEED)
        elif robot_state == "turning_90":
            # colour_cx < 320 means the line is to the left, turn left
            # colour_cx > 320 means the line is to the right, turn right
            if colour_cx is not None:
                if colour_cx < 320:
                    movement.move(-TURN_90_SPEED, TURN_90_SPEED)
                else:
                    movement.move( TURN_90_SPEED, -TURN_90_SPEED)
            else:
                # colour already gone — use black_line_side as fallback
                if black_line_side == "right":
                    movement.move( TURN_90_SPEED, -TURN_90_SPEED)
                else:
                    movement.move(-TURN_90_SPEED,  TURN_90_SPEED)

        else:
            # Normal PID — use colour cx if following colour, else black cx
            if robot_state == "following_colour" and colour_cx is not None:
                line_contour = valid_colour_contour
                cx           = colour_cx
                cv2.drawContours(im2, [line_contour], -1, (0, 0, 255),
                                 thickness=cv2.FILLED)
            elif robot_state == "following_black" and black_cx is not None:
                line_contour = valid_black_contour
                cx           = black_cx
                # draw secondary colour contours white if any were filtered out
                if len(filtered) > 1:
                    cv2.drawContours(im2, [c for _, c in filtered[1:]], -1,
                                     (255, 255, 255), thickness=cv2.FILLED)
                cv2.drawContours(im2, [line_contour], -1, (0, 255, 0),
                                 thickness=cv2.FILLED)
            else:
                cx = None

            if cx is not None:
                cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

                elapsed_time = time.perf_counter() - time_marker
                if elapsed_time <= 0:
                    elapsed_time = 0.0001

                error        = (320 - cx) / 320
                total_error += error * elapsed_time

                if not first:
                    diff_error = (error - last_error) / elapsed_time
                else:
                    first      = False
                    diff_error = 0

                pid        = kp * error + ki * total_error + kd * diff_error
                last_error = error
            else:
                print(f"No line found, steering {getSign(last_error)}")
                pid = getSign(last_error) * 2

            movement.move(clamp(base_speed + pid, -1, 1),
                          clamp(base_speed - pid, -1, 1))

        print(f"State:{robot_state} | side:{black_line_side} | pid:{pid:.3f}")

        '''
        cv2.imshow("contours", im2)
        if cv2.waitKey(1) == 27:
            movement.move(0, 0)
            break
        '''

    except (KeyboardInterrupt, Exception) as e:
        print(f"Error has occurred - {e}")
        break

movement.move(0, 0)
movement.pi.stop()
picam2.stop()
picam2.close()
# cv2.destroyAllWindows()