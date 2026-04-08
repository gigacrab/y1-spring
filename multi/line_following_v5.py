import cv2
import time
import numpy as np
import movement

def clamp(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    return value

def getSign(n):
    return (n > 0) - (n < 0)

base_speed = 0.35
kp = 0.625
ki = 0.01
kd = 0.02

# ── PID state ─────────────────────────────────────────────────────────────────
error       = 0
total_error = 0
last_error  = 0
diff_error  = 0
first       = True

# ── State machine constants & variables ───────────────────────────────────────
STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"
STATE_TURN_90      = "TURN_90"
STATE_BLIND_TURN   = "BLIND_TURN"

state      = STATE_FOLLOW_BLACK
last_state = STATE_FOLLOW_BLACK

# ── Feature Settings ──────────────────────────────────────────────────────────
black_line_side = "right"
SEARCH_SPEED    = 0.35

TURN_90_SPEED   = 0.65
TURN_90_LOCKOUT = 0.5
turn_90_start   = 0
blind_turn_start = 0
BLIND_TURN_TIME = 0.6
turn_90_dir     = "right"

frame_count = 0

# ── Fork / Arrow navigation ───────────────────────────────────────────────────
fork_count           = 0
FORK_CONFIRM         = 5
fork_override_active = False   # True = arrow set direction, block colour geometry

def stop():
    movement.move(0, 0)
    movement.pi.stop()
    cv2.destroyAllWindows()

def stop_forever():
    movement.move(0, 0)
    movement.pi.stop()
    cv2.destroyAllWindows()

def stop_for(seconds):
    movement.move(0, 0)
    time.sleep(seconds)

def force_blind_turn(direction):
    """
    Called by main.py when an arrow sign is detected.
    Remembers the direction so when the black horizontal bar is reached,
    the robot turns that way — same mechanic as the colour turn but direction
    comes from the sign, not pixel geometry.
    """
    global turn_90_dir, fork_override_active
    print(f"[OVERRIDE] Arrow seen → will turn {direction} at black bar")
    turn_90_dir          = direction
    fork_override_active = True

# ── Main Loop ─────────────────────────────────────────────────────────────────
def follow_line(frame):
    global state, black_line_side, turn_90_start, turn_90_dir, \
        total_error, first, frame_count, last_error, diff_error, \
        blind_turn_start, last_state, fork_count, fork_override_active

    time_marker = time.perf_counter()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    roi   = frame[240:480, :]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Masks ─────────────────────────────────────────────────────────────
    red_mask    = cv2.inRange(hsv, np.array([105,  30, 100]), np.array([140, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
    colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

    imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Blindfold during SEARCH ────────────────────────────────────────────
    if state == STATE_SEARCH:
        if black_line_side == "left":
            thresh[:, :320] = 0
        elif black_line_side == "right":
            thresh[:, 320:] = 0

    # ── Contours ──────────────────────────────────────────────────────────
    color_cnts, _ = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_color_cnt = None
    color_cx        = None
    for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
        if 7500 <= cv2.contourArea(cnt) <= 40000:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                valid_color_cnt = cnt
                color_cx        = int(M['m10'] / M['m00'])
            break

    valid_black_cnt = None
    black_cx        = None
    if ret < 150:
        black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
            if 7500 <= cv2.contourArea(cnt) <= 40000:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    valid_black_cnt = cnt
                    black_cx        = int(M['m10'] / M['m00'])
                break

    # ── Side memory ───────────────────────────────────────────────────────
    if state == STATE_FOLLOW_BLACK and valid_color_cnt is not None and color_cx is not None:
        black_line_side = "right" if color_cx < 320 else "left"

    # ── Colour horizontal check ────────────────────────────────────────────
    color_is_horizontal = False
    if valid_color_cnt is not None:
        (_, (rw, rh), _) = cv2.minAreaRect(valid_color_cnt)
        long_side  = max(rw, rh)
        short_side = min(rw, rh)
        if short_side > 0 and long_side / short_side > 2.5 and short_side > 80:
            color_is_horizontal = True

    # ── Black horizontal check (exact same logic as colour) ───────────────
    # The black bar at a fork looks wide in the ROI, same as colour at a 90° turn.
    # FORK_CONFIRM frames required to prevent false triggers on diagonal crossings.
    black_is_horizontal = False
    if valid_black_cnt is not None:
        (_, (bw, bh), _) = cv2.minAreaRect(valid_black_cnt)
        b_long  = max(bw, bh)
        b_short = min(bw, bh)
        if b_short > 0 and (b_long / b_short) > 2.5 and b_short > 80:
            fork_count += 1
            print(f"[Fork] count={fork_count} ratio={b_long/b_short:.2f}")
        else:
            fork_count = 0
    else:
        fork_count = 0
    black_is_horizontal = (fork_count >= FORK_CONFIRM)

    # ── State Machine Transitions ─────────────────────────────────────────
    if state == STATE_TURN_90:
        elapsed_turn = time.perf_counter() - turn_90_start
        if elapsed_turn > TURN_90_LOCKOUT:
            fork_override_active = False   # turn done, re-enable colour geometry
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

    elif state == STATE_BLIND_TURN:
        elapsed_blind = time.perf_counter() - blind_turn_start
        if elapsed_blind > BLIND_TURN_TIME and valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK
        elif elapsed_blind > 2.0:
            state = STATE_SEARCH

    elif state == STATE_SEARCH:
        if valid_color_cnt is not None and not color_is_horizontal:
            state = STATE_FOLLOW_COLOR
        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    else:  # Normal FOLLOW states
        # ── Black bar hit → same as colour turn, direction from arrow ──────
        if black_is_horizontal:
            if not fork_override_active:
                # No arrow seen — fall back to pixel count on black mask
                left_px  = cv2.countNonZero(thresh[:, :320])
                right_px = cv2.countNonZero(thresh[:, 320:])
                turn_90_dir = "left" if left_px > right_px else "right"
                print(f"[Fork] No arrow — pixel fallback → {turn_90_dir}")
            else:
                print(f"[Fork] Arrow override → {turn_90_dir}")
            turn_90_start = time.perf_counter()
            state = STATE_TURN_90

        # ── Colour bar hit → only allowed if no arrow override pending ─────
        # Prevents colour geometry from overwriting the arrow direction
        # before the robot even reaches the black bar.
        elif color_is_horizontal and not fork_override_active:
            left_px  = cv2.countNonZero(colour_mask[:, :320])
            right_px = cv2.countNonZero(colour_mask[:, 320:])
            turn_90_dir   = "left" if left_px > right_px else "right"
            turn_90_start = time.perf_counter()
            state = STATE_TURN_90

        elif valid_color_cnt is not None:
            state = STATE_FOLLOW_COLOR

        elif state == STATE_FOLLOW_COLOR:
            if valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK
            else:
                state = STATE_BLIND_TURN
                blind_turn_start = time.perf_counter()
            print(f"Color line ended → {state} (mem: {black_line_side})")

        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    # ── PID Reset on state change ──────────────────────────────────────────
    if state != last_state:
        print(f"[STATE] {last_state} → {state}")
        total_error          = 0
        last_error           = 0
        first                = True
        fork_count           = 0
        fork_override_active = False
        last_state           = state

    # ── Motor Control ─────────────────────────────────────────────────────
    im2 = np.zeros((240, 640, 3), dtype=np.uint8)

    if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
        line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
        cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

        if line_contour is not None and cx is not None:
            cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

            elapsed_time = max(time.perf_counter() - time_marker, 0.0001)

            error        = (320 - cx) / 320
            total_error += error * elapsed_time

            if not first:
                diff_error = (error - last_error) / elapsed_time
            else:
                first      = False
                diff_error = 0

            pid        = kp * error + ki * total_error + kd * diff_error
            last_error = error

            left_pwm  = base_speed + pid
            right_pwm = base_speed - pid

        else:
            pid       = getSign(last_error) * 2
            left_pwm  = base_speed + pid
            right_pwm = base_speed - pid

    elif state == STATE_BLIND_TURN:
        if black_line_side == "left":
            left_pwm  = -base_speed * 1.2
            right_pwm =  base_speed * 1.2
        else:
            left_pwm  =  base_speed * 1.2
            right_pwm = -base_speed * 1.2

    elif state == STATE_SEARCH:
        turn_pwm  = SEARCH_SPEED if black_line_side == "left" else -SEARCH_SPEED
        left_pwm  = base_speed + turn_pwm
        right_pwm = base_speed - turn_pwm

    elif state == STATE_TURN_90:
        # Identical output for both colour bar and black bar turns
        turn_pwm  = TURN_90_SPEED if turn_90_dir == "left" else -TURN_90_SPEED
        left_pwm  = base_speed + turn_pwm
        right_pwm = base_speed - turn_pwm

    else:
        left_pwm  = base_speed
        right_pwm = base_speed

    clamped_left_pwm  = clamp(left_pwm,  -1, 1)
    clamped_right_pwm = clamp(right_pwm, -1, 1)
    movement.move(clamped_left_pwm, clamped_right_pwm)

    # ── Debug Display ─────────────────────────────────────────────────────
    '''frame_count += 1
    if frame_count % 5 == 0:
        if valid_color_cnt is not None:
            x, y, w, h = cv2.boundingRect(valid_color_cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.putText(im2, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(im2, f"MEM: {black_line_side}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(im2, f"OVERRIDE: {fork_override_active} dir={turn_90_dir}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("1. Color Mask", colour_mask)
        cv2.imshow("2. Tracking & Math", im2)
        if cv2.waitKey(1) == 27:
            pass'''