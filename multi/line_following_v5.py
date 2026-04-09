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
STATE_SEARCH       = "SEARCH"       # colour lost → sweep toward black_line_side
STATE_TURN_90      = "TURN_90"      # executing a 90-degree turn on the colour line
STATE_BLIND_TURN     = "BLIND_TURN"     # blindfolded during the 90-degree turn, ignoring colour

state      = STATE_FOLLOW_BLACK
last_state = STATE_FOLLOW_BLACK     # Used to trigger the PID memory reset

# ── Feature Settings ──────────────────────────────────────────────────────────
black_line_side = "right"  # Memory of where the black line is relative to color
SEARCH_SPEED    = 0.35     # Motor speed when sweeping for a lost line

TURN_90_SPEED   = 0.65     # Hard-turn PWM offset during the 90° manoeuvre
TURN_90_LOCKOUT = 0.5      # Seconds to ignore re-acquisition (prevents double triggering)
turn_90_start   = 0
blind_turn_start = 0
BLIND_TURN_TIME = 1.0
turn_90_dir     = "right"

frame_count = 0

# ── Fork / Arrow navigation ──
pending_turn = None
branch_memory = None
fork_count = 0
horizontal_count  = 0
fork_cooldown_end = 0.0
last_time         = None
FORK_CONFIRM = 3
# Module level
color_lost_count = 0
COLOR_LOST_CONFIRM = 4

def stop():
    movement.move(0, 0)

def stop_forever():
    """Called by main.py finally block to kill motors and clean up."""
    movement.move(0, 0)
    movement.pi.stop()
    cv2.destroyAllWindows()
 
def stop_for(seconds):
    """Halt motors for a fixed duration, then return so line following resumes."""
    movement.move(0, 0)
    time.sleep(seconds)

def turn_360():
    movement.move(-1, 1)
    time.sleep(2)
    global last_error
    if getSign(last_error) == -1:
        last_error *= -1

def force_blind_turn(direction):
    """Instantly overrides the state machine to perform a blind turn."""
    global state, black_line_side, blind_turn_start, branch_memory
    print(f"[OVERRIDE] Arrow detected! Forcing immediate blind turn: {direction}")
    black_line_side = direction
    blind_turn_start = time.perf_counter()
    branch_memory = direction
    state = STATE_BLIND_TURN

# ── Main Loop ─────────────────────────────────────────────────────────────────
def follow_line(frame):
    global state, black_line_side, turn_90_start, turn_90_dir, \
        total_error, first, frame_count, last_error, diff_error, \
        blind_turn_start, last_state, error, \
        horizontal_count, pending_turn, fork_count, branch_memory, \
        fork_cooldown_end, last_time, color_lost_count
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    roi   = frame[240:480, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Masks ─────────────────────────────────────────────────────────────
    # Color Mask
    red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
    colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

    # Black Mask
    imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # THE BLINDFOLD FIX
    # If searching, black out the side of the camera we DON'T want to look at.
    if state == STATE_SEARCH:
        if black_line_side == "left":
            # We want the right side. Blindfold the left (columns 0 to 320).
            thresh[:, :320] = 0
        elif black_line_side == "right":
            # We want the left side. Blindfold the right (columns 320 to end).
            thresh[:, 320:] = 0

    # ── Contours & Centroids ──────────────────────────────────────────────
    # Color Contours
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

    # Black Contours
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

    # ── Geometry & Memory Updates ─────────────────────────────────────────
    # NEW - use where the red line appears to predict where black exits
    if state == STATE_FOLLOW_BLACK and valid_color_cnt is not None and color_cx is not None:
        # Red appears on LEFT (cx < 320) → robot will exit red toward the RIGHT black arm
        # Red appears on RIGHT (cx > 320) → robot will exit red toward the LEFT black arm
        black_line_side = "right" if color_cx < 320 else "left"
        # Uncomment the next line if you want to verify it in the console during testing!
        # print(f"Red line spotted at cx={color_cx}, predicting exit turn: {black_line_side}")

    # ── Colour-line horizontal check (with confirmation counter) ──────────
    # Uses minAreaRect so a slightly curved or diagonal line doesn't spike the width.
    # Only triggers after HORIZONTAL_CONFIRM consecutive wide frames to filter
    # the transient diagonal crossing at junctions.
    color_is_horizontal = False
    if valid_color_cnt is not None:
        (_, (rw, rh), _) = cv2.minAreaRect(valid_color_cnt)
        long_side  = max(rw, rh)
        short_side = min(rw, rh)
        if short_side > 0 and long_side / short_side > 2.5 and short_side > 80:
            color_is_horizontal = True
 
    # ── Black-line fork check (with confirmation counter) ─────────────────
    # A T-junction or fork makes the black line appear wide in the ROI,
    # just like the colour 90° check above.
    black_is_fork = False
    if valid_black_cnt is not None:
        (_, (bw, bh), _) = cv2.minAreaRect(valid_black_cnt)
        b_long  = max(bw, bh)
        b_short = min(bw, bh)
        if b_short > 0 and (b_long / b_short) > 2.5 and b_short > 80:
            fork_count += 1
            print(f"Fork-like contour detected (count={fork_count}): bw={bw:.1f}, bh={bh:.1f}, ratio={b_long/b_short:.2f}")
        else:
            fork_count = 0
    else:
        fork_count = 0
    black_is_fork = (fork_count >= FORK_CONFIRM) and (time.perf_counter() >= fork_cooldown_end)

    # ── State Machine Transitions ─────────────────────────────────────────
    if state == STATE_TURN_90:
        elapsed_turn = time.perf_counter() - turn_90_start
        if elapsed_turn > TURN_90_LOCKOUT:
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

    elif state == STATE_BLIND_TURN:
        elapsed_blind = time.perf_counter() - blind_turn_start
        if elapsed_blind > BLIND_TURN_TIME and valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK
            fork_cooldown_end = time.perf_counter() + 1.2
        elif elapsed_blind > 2.0:
            state = STATE_SEARCH

    elif state == STATE_SEARCH:
        if valid_color_cnt is not None and not color_is_horizontal:
            state = STATE_FOLLOW_COLOR
        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    else: # Normal FOLLOW states
        '''if black_is_fork and state == STATE_FOLLOW_BLACK:
            if branch_memory is not None:
                print(f"[Fork EXIT] Turning {branch_memory} onto exit leg")
                black_line_side  = branch_memory
                branch_memory    = None
                blind_turn_start = time.perf_counter()
                state            = STATE_BLIND_TURN
            else:
                # Pixel-vote fallback — no arrow, no memory
                left_px  = cv2.countNonZero(thresh[:, :320])
                right_px = cv2.countNonZero(thresh[:, 320:])
                turn_dir = "left" if left_px > right_px else "right"
                print(f"[Fork FALLBACK] Pixels (L:{left_px} R:{right_px}) → {turn_dir}")
                black_line_side  = turn_dir
                blind_turn_start = time.perf_counter()
                state            = STATE_BLIND_TURN'''

        if color_is_horizontal:
            left_px  = cv2.countNonZero(colour_mask[:, :320])
            right_px = cv2.countNonZero(colour_mask[:, 320:])
            turn_90_dir   = "left" if left_px > right_px else "right"
            turn_90_start = time.perf_counter()
            state = STATE_TURN_90

        elif valid_color_cnt is not None:
            color_lost_count = 0        # ADD
            state = STATE_FOLLOW_COLOR

        elif state == STATE_FOLLOW_COLOR:
            color_lost_count += 1
            if color_lost_count >= COLOR_LOST_CONFIRM:
                color_lost_count = 0
                if valid_black_cnt is not None:
                    if branch_memory is not None:                        # ADD
                        black_line_side  = branch_memory                # ADD
                        branch_memory    = None                         # ADD
                        blind_turn_start = time.perf_counter()          # ADD
                        state            = STATE_BLIND_TURN             # ADD
                        print(f"[Color Exit] Memory consumed → {black_line_side}")  # ADD
                    else:
                        state = STATE_FOLLOW_BLACK
                else:
                    state = STATE_BLIND_TURN
                    blind_turn_start = time.perf_counter()
                print(f"Color line ended → {state} (mem: {black_line_side})")

        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    # ✨ THE PID RESET FIX ✨
    # Clear the integral memory so past curvy turns don't ruin straight lines
    if state != last_state:
        print(f"[STATE CHANGE] {last_state} ---> {state}")
        total_error = 0
        last_error = 0
        first = True
        horizontal_count = 0
        fork_count       = 0
        color_lost_count = 0
        last_state = state

    # ── Motor Control ─────────────────────────────────────────────────────
    im2 = np.zeros((240, 640, 3), dtype=np.uint8)

    if state in (STATE_FOLLOW_COLOR, STATE_FOLLOW_BLACK):
        line_contour = valid_color_cnt if state == STATE_FOLLOW_COLOR else valid_black_cnt
        cx           = color_cx        if state == STATE_FOLLOW_COLOR else black_cx

        if line_contour is not None and cx is not None:
            cv2.drawContours(im2, [line_contour], -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.line(im2, (cx, 0), (cx, 240), (0, 255, 255), 3)

            current_time = time.perf_counter()
            if last_time is None:
                elapsed_time = 0.033
            else:
                elapsed_time = max(current_time - last_time, 0.0001)
            last_time = current_time

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
            # Lost mid-frame fallback
            pid       = getSign(last_error) * 2
            left_pwm  = base_speed + pid
            right_pwm = base_speed - pid

    elif state == STATE_BLIND_TURN:
        # Sweeping turn to find the black line
        if black_line_side == "left":
            left_pwm  = -base_speed * 1.2
            right_pwm = base_speed * 1.2
        else:
            left_pwm  = base_speed * 1.2
            right_pwm = -base_speed * 1.2

    elif state == STATE_SEARCH:
        # Sweeping turn to find the black line
        turn_pwm  = SEARCH_SPEED if black_line_side == "left" else -SEARCH_SPEED
        left_pwm  = base_speed + turn_pwm
        right_pwm = base_speed - turn_pwm

    elif state == STATE_TURN_90:
        # Hard 90-degree turn
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
        
        cv2.imshow("1. Color Mask", colour_mask)
        cv2.imshow("2. Tracking & Math", im2)

        if cv2.waitKey(1) == 27: # Press ESC to quit
            break'''