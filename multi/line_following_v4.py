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

# ── State machine ─────────────────────────────────────────────────────────────
STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
STATE_SEARCH       = "SEARCH"
STATE_TURN_90      = "TURN_90"
STATE_BLIND_TURN   = "BLIND_TURN"

state      = STATE_FOLLOW_BLACK
last_state = STATE_FOLLOW_BLACK

# ── Line following settings ───────────────────────────────────────────────────
black_line_side  = "right"
SEARCH_SPEED     = 0.35
TURN_90_SPEED    = 0.65
TURN_90_LOCKOUT  = 0.5
turn_90_start    = 0
turn_90_dir      = "right"
blind_turn_start = 0
BLIND_TURN_TIME  = 0.6

# ── Fork / Arrow settings ─────────────────────────────────────────────────────
# Maps detect_object() output strings → "left" / "right" / None (ignore)
ARROW_MAP = {
    "Arrow (LEFT)":  "left",
    "Arrow (RIGHT)": "right",
    "Arrow (UP)":    None,
    "Arrow (DOWN)":  None,
}

pending_turn        = None   # direction set by detected arrow, waiting for a fork
pending_turn_time   = 0      # when pending_turn was set (for expiry)
PENDING_TURN_EXPIRY = 3.0    # seconds before an unused arrow instruction expires

branch_memory       = None   # direction used at entry fork, reused at exit fork
branch_memory_used  = False  # True once branch_memory has been used at exit

fork_count          = 0      # consecutive frames seeing a wide black line
FORK_CONFIRM        = 3      # frames needed to confirm a real fork

fork_lockout_start  = 0      # timestamp of last completed fork turn
FORK_LOCKOUT        = 1.5    # seconds to ignore new forks after completing one

frame_count = 0

# ── Arrow map translator (called by main.py when worker sends a result) ───────
def set_pending_turn(detection_result):
    """
    Call this from main.py when result_q gives a new detection.
    detection_result is the string from detect_object(), e.g. "Arrow (LEFT)".
    Translates it and stores in pending_turn with a timestamp.
    """
    global pending_turn, pending_turn_time
    if detection_result in ARROW_MAP:
        direction = ARROW_MAP[detection_result]
        if direction is not None:
            pending_turn      = direction
            pending_turn_time = time.perf_counter()
            print(f"[Arrow] Pending turn set → {pending_turn}")
        else:
            print(f"[Arrow] Ignored direction: {detection_result}")

def stop_forever():
    movement.move(0, 0)
    movement.pi.stop()
    cv2.destroyAllWindows()

def stop_for(seconds):
    movement.move(0, 0)
    time.sleep(seconds)

# ── Main follow function ───────────────────────────────────────────────────────
def follow_line(frame):
    global state, black_line_side, turn_90_start, turn_90_dir, \
        total_error, first, frame_count, last_error, diff_error, \
        blind_turn_start, last_state, \
        pending_turn, pending_turn_time, \
        branch_memory, branch_memory_used, \
        fork_count, fork_lockout_start

    time_marker = time.perf_counter()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    roi   = frame[240:480, :]
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Masks ─────────────────────────────────────────────────────────────────
    red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
    colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

    imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Blindfold during search ────────────────────────────────────────────────
    if state == STATE_SEARCH:
        if black_line_side == "left":
            thresh[:, :320] = 0
        elif black_line_side == "right":
            thresh[:, 320:] = 0

    # ── Colour contour ─────────────────────────────────────────────────────────
    color_cnts, _   = cv2.findContours(colour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_color_cnt = None
    color_cx        = None
    for cnt in sorted(color_cnts, key=cv2.contourArea, reverse=True):
        if 7500 <= cv2.contourArea(cnt) <= 40000:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                valid_color_cnt = cnt
                color_cx        = int(M['m10'] / M['m00'])
        break

    # ── Black contour ──────────────────────────────────────────────────────────
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

    # ── black_line_side memory update ──────────────────────────────────────────
    if state == STATE_FOLLOW_BLACK and valid_color_cnt is not None and color_cx is not None:
        black_line_side = "right" if color_cx < 320 else "left"

    # ── SUGGESTION 2: Expire pending_turn if fork never arrives ───────────────
    now = time.perf_counter()
    if pending_turn is not None and (now - pending_turn_time) > PENDING_TURN_EXPIRY:
        print(f"[Arrow] pending_turn '{pending_turn}' expired without seeing a fork")
        pending_turn = None

    # ── Colour horizontal check (minAreaRect) ──────────────────────────────────
    color_is_horizontal = False
    if valid_color_cnt is not None:
        (_, (rw, rh), _) = cv2.minAreaRect(valid_color_cnt)
        long_side  = max(rw, rh)
        short_side = min(rw, rh)
        if short_side > 0 and long_side / short_side > 2.5 and short_side > 80:
            color_is_horizontal = True

    # ── Black fork check with confirmation counter ─────────────────────────────
    # SUGGESTION 5: Also respect fork lockout after a completed blind turn
    in_fork_lockout = (now - fork_lockout_start) < FORK_LOCKOUT

    if valid_black_cnt is not None and not in_fork_lockout:
        (_, (bw, bh), _) = cv2.minAreaRect(valid_black_cnt)
        b_long  = max(bw, bh)
        b_short = min(bw, bh)
        if b_short > 0 and (b_long / b_short) > 2.5 and b_short > 80:
            fork_count += 1
        else:
            fork_count = 0
    else:
        fork_count = 0

    black_is_fork = (fork_count >= FORK_CONFIRM)

    # ── State machine transitions ──────────────────────────────────────────────
    if state == STATE_TURN_90:
        elapsed_turn = now - turn_90_start
        if elapsed_turn > TURN_90_LOCKOUT:
            if valid_color_cnt is not None and not color_is_horizontal:
                state = STATE_FOLLOW_COLOR
            elif valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK

    elif state == STATE_BLIND_TURN:
        elapsed_blind = now - blind_turn_start
        if elapsed_blind > BLIND_TURN_TIME and valid_black_cnt is not None:
            state              = STATE_FOLLOW_BLACK
            fork_lockout_start = now     # SUGGESTION 5: start lockout
        elif elapsed_blind > 2.0:
            state              = STATE_SEARCH
            fork_lockout_start = now

    elif state == STATE_SEARCH:
        if valid_color_cnt is not None and not color_is_horizontal:
            state = STATE_FOLLOW_COLOR
        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    else:
        # ── Normal follow states ───────────────────────────────────────────────
        if black_is_fork:
            # ── SUGGESTION 1/2/3/4: Arrow priority logic ───────────────────────
            if pending_turn is not None:
                # ENTRY: Arrow was seen recently — use it and save to branch_memory
                turn_dir           = pending_turn
                branch_memory      = pending_turn   # SUGGESTION 3: save for exit
                branch_memory_used = False
                pending_turn       = None
                print(f"[Fork ENTRY] Arrow → turning {turn_dir}. Saved to branch_memory.")

            elif branch_memory is not None and not branch_memory_used:
                # EXIT: No arrow, but we have the entry memory — reuse it once
                turn_dir           = branch_memory
                branch_memory_used = True           # SUGGESTION 3: only use once
                print(f"[Fork EXIT] Memory → turning {turn_dir}. Memory now spent.")

            else:
                # FALLBACK: No arrow, no valid memory — pixel count decides
                left_px  = cv2.countNonZero(thresh[:, :320])
                right_px = cv2.countNonZero(thresh[:, 320:])
                turn_dir = "left" if left_px > right_px else "right"
                branch_memory      = None
                branch_memory_used = False
                print(f"[Fork] No arrow/memory — pixel fallback → {turn_dir}")

            # Trigger blind turn in the decided direction
            black_line_side  = turn_dir
            blind_turn_start = now
            state            = STATE_BLIND_TURN
            fork_count       = 0

        elif color_is_horizontal:
            left_px       = cv2.countNonZero(colour_mask[:, :320])
            right_px      = cv2.countNonZero(colour_mask[:, 320:])
            turn_90_dir   = "left" if left_px > right_px else "right"
            turn_90_start = now
            state         = STATE_TURN_90

        elif valid_color_cnt is not None:
            state = STATE_FOLLOW_COLOR

        elif state == STATE_FOLLOW_COLOR:
            if valid_black_cnt is not None:
                state = STATE_FOLLOW_BLACK
            else:
                state            = STATE_BLIND_TURN
                blind_turn_start = now
            print(f"[Main] Color ended → {state} (mem: {black_line_side})")

        elif valid_black_cnt is not None:
            state = STATE_FOLLOW_BLACK

    # ── PID reset on state change ──────────────────────────────────────────────
    if state != last_state:
        total_error = 0
        last_error  = 0
        first       = True
        fork_count  = 0
        last_state  = state

    # ── Motor control ──────────────────────────────────────────────────────────
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
            left_pwm  =  base_speed * 1.2
            right_pwm = -base_speed * 1.2
        else:
            left_pwm  = -base_speed * 1.2
            right_pwm =  base_speed * 1.2

    elif state == STATE_SEARCH:
        if black_line_side == "left":
            left_pwm  = -base_speed
            right_pwm =  base_speed
        else:
            left_pwm  =  base_speed
            right_pwm = -base_speed

    elif state == STATE_TURN_90:
        turn_pwm  = TURN_90_SPEED if turn_90_dir == "left" else -TURN_90_SPEED
        left_pwm  = base_speed + turn_pwm * 2
        right_pwm = base_speed - turn_pwm * 2

    else:
        left_pwm  = base_speed
        right_pwm = base_speed

    movement.move(clamp(left_pwm, -1, 1), clamp(right_pwm, -1, 1))

    # ── Debug display ──────────────────────────────────────────────────────────
    frame_count += 1
    if frame_count % 5 == 0:
        if valid_color_cnt is not None:
            x, y, w, h = cv2.boundingRect(valid_color_cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)

        cv2.putText(im2, f"STATE: {state}",           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(im2, f"MEM: {black_line_side}",   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),  2)
        cv2.putText(im2, f"ARROW: {pending_turn}",    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100),  2)
        cv2.putText(im2, f"BRANCH: {branch_memory}",  (10,115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0),  2)
        cv2.putText(im2, f"FORK_CNT: {fork_count}",   (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255),2)

        cv2.imshow("1. Color Mask", colour_mask)
        cv2.imshow("2. Tracking & Math", im2)
        cv2.waitKey(1)