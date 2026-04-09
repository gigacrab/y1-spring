import cv2
import time
import numpy as np
import movement

# tune ret, calibrated colours

def clamp(value, min, max):
    if value > max:
        return max
    elif value < min:
        return min
    return value

# returns 1 if positive, -1 if negative
def getSign(n):
    return (n > 0) - (n < 0)

base_speed = 0.3
kp = 0.625
ki = 0.01
kd = 0.02

error = 0
total_error = 0
last_error = 0
diff_error = 0
first = True

color_follow = False
color_start = 0
color_period = 2
mask_black = False
mask_cooldown = 1
mask_start = 0
color_error = 0
black_error = 0

arrow_follow = False
arrow_error = 0

ret_thresh = 120 # 120

def stop():
    movement.move(0, 0)

def stop_forever():
    movement.move(0, 0)
    movement.pi.stop()
    cv2.destroyAllWindows()
 
def stop_for(seconds):
    movement.move(0, 0)
    time.sleep(seconds)

def turn_360():
    movement.move(1, -1)
    time.sleep(1.5)
    global last_error
    if getSign(last_error) == -1:
        last_error *= -1

def shift(dir):
    global arrow_follow, arrow_error
    if dir.lower() == "left":
        movement.move(1, -1)
        arrow_follow = True
        arrow_error = 1
        time.sleep(0.25)
    elif dir.lower() == "right":
        movement.move(-1, 1)
        arrow_follow = True
        arrow_error = -1
        time.sleep(0.25)

def calc_pid(cx, time_marker, ret):
    global error, total_error, last_error, diff_error, first

    elapsed_time = time.perf_counter() - time_marker
    if elapsed_time <= 0:
        elapsed_time = 0.0001

    # normalize error
    error = (320 - cx) / 320
    total_error += error * elapsed_time

    if not first:
        diff_error = (error - last_error) / elapsed_time
        #print(ret)
    else:
        first = False

    pid = kp * error + ki * total_error + kd * diff_error

    last_error = error

    return pid

def follow_line(frame):
    global color_follow, color_error, color_start, mask_black, \
        mask_start, last_error, black_error, arrow_follow, arrow_error
    ret = 0
    time_marker = time.perf_counter()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    roi = frame[240:480, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask    = cv2.inRange(hsv, np.array([90, 30,  30]), np.array([140, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
    color_mask = cv2.bitwise_or(red_mask, yellow_mask)

    imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # we now try gaussian blur
    #imgray = cv2.GaussianBlur(imgray, (5,5), 0)
    #cv2.imshow("gray", imgray)
    # 0 - values above this, assigned 255, the Otsu method adjusts according to lighting
    # however the Otsu method wasn't that good because it'd always find a region of threshold
    # also idc about the ret
    thresh = cv2.adaptiveThreshold(
        imgray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        91, 8
    )
    #_, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV) 

    #cv2.imshow("thresh", thresh)
    
    # print(f"ret {ret}")
    # print(f"mask {mask_black}")
    # print(f"color {color_follow}")
    if mask_black and time.perf_counter() - mask_start < mask_cooldown:
        print(f"error {color_error}")
        if color_error == -1: # color was on the right
            # mask left side so that the bot continues going right
            thresh[:, :240] = 0
        elif color_error == 1: # color was on the left
            # mask right side so that the bot continues going left
            thresh[:, 400:] = 0
    else:
        mask_black = False # set it to false once cooldown ends

    # hierarchy -> [next, previous, first_child, parent]
    black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    color_cnts, _ = cv2.findContours(color_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #im2 = np.zeros((240, 640, 3), dtype=np.uint8)

    black_target = color_target = None
    black_cx = color_cx = None

    if ret < ret_thresh:
        black_sorted = sorted(black_cnts, key=cv2.contourArea, reverse=True)
        if black_sorted and cv2.contourArea(black_sorted[0]) > 7500:
            black_target = black_sorted[0]
            M = cv2.moments(black_target)
            if M['m00'] != 0:
                black_cx = int(M['m10'] / M['m00'])

    color_sorted = sorted(color_cnts, key=cv2.contourArea, reverse=True)
    if color_sorted and cv2.contourArea(color_sorted[0]) > 2000:
        color_target = color_sorted[0]
        M = cv2.moments(color_target)
        if M['m00'] != 0:
            color_cx = int(M['m10'] / M['m00'])
    
    if color_cx is not None: 
        pid = calc_pid(color_cx, time_marker, ret)
        if not color_follow: # we take initial error so that we know where to turn at the end
            #color_error = -getSign(last_error)
            
            black_error = last_error
            color_start = time.perf_counter()
            color_follow = True
    elif ret < ret_thresh and black_cx is not None: # ret condition just added for guard
        pid = calc_pid(black_cx, time_marker, ret) * 2
        # ext_left_x = black_target[black_target[:, :, 0].argmin()][0][0]
        # ext_right_x = black_target[black_target[:, :, 0].argmax()][0][0]
        # print(f"max = {ext_left_x, ext_right_x}")
        if color_follow: # it just followed color earlier
            if time.perf_counter() - color_start > color_period:
                color_error = getSign(last_error - black_error)
                # as a result, we temporarily mask the black contours
                # so that it can resume following the track
                mask_black = True
                mask_start = time.perf_counter()
            color_follow = False
    else:
        print(f"we cannot find contours {getSign(last_error)}")
        # in case line is lost immediately after color follow ends
        if color_follow:
            last_error = color_error
            color_follow = False
        if arrow_follow:
            last_error = arrow_error
            arrow_error = 0
            arrow_follow = False
        pid = getSign(last_error) * 2            

    # cv2.imshow("threshold", thresh)
    # cv2.imshow("color", color_mask)
    # cv2.waitKey(1)

    right_pwm = base_speed + pid
    left_pwm = base_speed - pid

    clamped_right_pwm = clamp(right_pwm, -1, 1)
    clamped_left_pwm = clamp(left_pwm, -1, 1)

    movement.move(clamped_right_pwm, clamped_left_pwm)