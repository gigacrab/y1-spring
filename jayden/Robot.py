"""
robot_main.py  –  Autonomous robot: line following + shape detection

Architecture:
  Process 1 (main)   – Picamera2 capture, PID, motor control. Runs as fast as possible.
  Process 2 (worker) – Heavy detect_object() vision work. Completely independent pace.

Communication:
  frame_q  (maxsize=1): main  → worker. Main drops frames if worker is busy.
  result_q (maxsize=1): worker → main.  Main reads latest result non-blocking.

This means the motors NEVER wait for shape detection. The two processes are
fully decoupled — they share no memory, only the two queues.
"""

import cv2
import time
import numpy as np
import sys
import multiprocessing

from picamera2 import Picamera2
import movement

# ── Import the detection function from your Object.py ────────────────────────
# Make sure Object.py is in the same directory as this file.
from detect_object import detect_object


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS 2: THE SHAPE DETECTION WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def detection_worker(frame_q, result_q):
    """
    This function IS Process 2. It runs in a completely separate Python
    interpreter — it has no access to the camera, motors, or PID variables.

    Its entire job is a simple infinite loop:
      1. Wait for a frame to appear in frame_q (this blocks, but that's fine
         because this process has nothing else to do while waiting).
      2. Run detect_object() on that frame (slow — can take 50-500ms).
      3. Put the result into result_q for the main process to read later.

    The `None` sentinel value is the agreed shutdown signal: when the main
    process wants to stop, it puts None into frame_q. This worker sees it,
    breaks the loop, and exits cleanly.
    """
    print("[Worker] Shape detection process started.")

    while True:
        # block=True means we patiently wait here until a frame arrives.
        # This is the correct behaviour — the worker should be idle, not spinning.
        frame = frame_q.get(block=True)

        # The shutdown sentinel: main process signals us to quit.
        if frame is None:
            print("[Worker] Received shutdown signal. Exiting.")
            break

        # Run the heavy detection. This may take hundreds of milliseconds.
        # While this runs, the main process is completely unaffected — it keeps
        # driving the robot because it never waits for this to finish.
        result = detect_object(frame)

        # Send the result back. If result_q already has an unread result in it
        # (main process was busy), we discard the old one and put the new one in.
        # This ensures main always sees the freshest detection, never a stale one.
        if result is not None:
            try:
                # Try to empty any old unread result before putting the new one.
                result_q.get_nowait()
            except Exception:
                pass  # Queue was already empty — that's fine.
            try:
                result_q.put_nowait(result)
            except Exception:
                pass  # Extremely rare race condition — just drop it.

    print("[Worker] Process cleanly exited.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS 1: MAIN — LINE FOLLOWING + MOTOR CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

def clamp(value, min_val, max_val):
    if value > max_val:
        return max_val
    elif value < min_val:
        return min_val
    return value

def getSign(n):
    return (n > 0) - (n < 0)


def main():
    # ── Command-line arguments ────────────────────────────────────────────────
    if len(sys.argv) == 5:
        base_speed = float(sys.argv[1])
        kp         = float(sys.argv[2])
        ki         = float(sys.argv[3])
        kd         = float(sys.argv[4])
    else:
        raise Exception("Usage: python robot_main.py base_speed kp ki kd")

    # ── Build the two queues BEFORE spawning the worker process ──────────────
    #
    # maxsize=1 is the key design decision for BOTH queues:
    #
    # frame_q (main → worker):
    #   If the worker is still processing the previous frame when main tries to
    #   send a new one, the queue is already full (it has 1 item). Main catches
    #   the queue.Full exception and simply drops the new frame. The robot
    #   keeps driving. No lag, no backlog.
    #
    # result_q (worker → main):
    #   If main hasn't read the last result yet when the worker finishes a new
    #   one, the worker discards the old result (see detection_worker above) and
    #   replaces it. Main always reads the freshest detection.
    #
    frame_q  = multiprocessing.Queue(maxsize=1)
    result_q = multiprocessing.Queue(maxsize=1)

    # ── Spawn the worker process ──────────────────────────────────────────────
    # daemon=True means if the main process dies unexpectedly (crash, kill),
    # the worker is automatically cleaned up by the OS. Without this, a crashed
    # main process could leave an orphaned worker process running forever.
    worker = multiprocessing.Process(
        target=detection_worker,
        args=(frame_q, result_q),
        daemon=True
    )
    worker.start()
    print(f"[Main] Worker process started with PID {worker.pid}")

    # ── Camera initialisation ─────────────────────────────────────────────────
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Let auto-exposure settle.

    # ── PID state ─────────────────────────────────────────────────────────────
    error       = 0
    total_error = 0
    last_error  = 0
    diff_error  = 0
    first       = True

    # ── State machine ─────────────────────────────────────────────────────────
    STATE_FOLLOW_BLACK = "FOLLOW_BLACK"
    STATE_FOLLOW_COLOR = "FOLLOW_COLOR"
    STATE_SEARCH       = "SEARCH"
    STATE_TURN_90      = "TURN_90"

    state      = STATE_FOLLOW_BLACK
    last_state = STATE_FOLLOW_BLACK

    colour_entry_sign = 0
    SEARCH_SPEED      = 0.35

    TURN_90_SPEED   = 0.65
    TURN_90_LOCKOUT = 0.5
    turn_90_start   = 0
    turn_90_dir     = "right"

    # ── Shape detection result (shared across loop iterations) ───────────────
    # This holds the LAST completed detection result from the worker.
    # It persists across frames so the robot "remembers" what it last saw,
    # even while the worker is busy computing the next result.
    latest_detection = None

    # Rate-limiter: don't send a new frame to the worker more than once per
    # this many seconds. The worker is slow — sending every frame wastes queue
    # bandwidth and causes old frames to queue behind new ones.
    DETECTION_INTERVAL = 0.0   # seconds between frame submissions
    last_sent_time     = 0.0

    frame_count = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            time_marker = time.perf_counter()

            # ── Capture ──────────────────────────────────────────────────────
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            roi   = frame[240:480, :]
            hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # ── Non-blocking result read from worker ──────────────────────────
            # get_nowait() returns immediately. If the worker hasn't finished
            # yet, it raises queue.Empty and we just keep latest_detection as-is.
            # This is the key to non-blocking communication: we NEVER call
            # result_q.get() without nowait, because that would freeze the motors.
            try:
                latest_detection = result_q.get_nowait()
                print(f"[Main] New detection: {latest_detection}")
            except Exception:
                pass  # No result ready yet — keep driving with the last known one.

            # ── Non-blocking frame send to worker ─────────────────────────────
            # We send a COPY of the full frame (not the cropped roi) so the
            # worker has the complete image context for contour hierarchy analysis.
            # We also rate-limit sends to avoid flooding the queue faster than
            # the worker can consume.
            now = time.perf_counter()
            if now - last_sent_time >= DETECTION_INTERVAL:
                try:
                    # put_nowait raises queue.Full if the worker is still busy.
                    # We catch it and drop the frame — never block.
                    '''small_frame = cv2.resize(frame.copy(), (320, 240))
                    frame_q.put_nowait(small_frame)'''
                    frame_q.put_nowait(frame.copy())
                    last_sent_time = now
                except Exception:
                    # Worker is busy — drop this frame silently. The robot keeps going.
                    pass

            # ── Colour mask ───────────────────────────────────────────────────
            red_mask    = cv2.inRange(hsv, np.array([105, 30,  100]), np.array([140, 255, 255]))
            yellow_mask = cv2.inRange(hsv, np.array([ 85, 100, 180]), np.array([105, 255, 255]))
            colour_mask = cv2.bitwise_or(red_mask, yellow_mask)

            # ── Black mask ────────────────────────────────────────────────────
            imgray      = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # ── Find best colour contour ──────────────────────────────────────
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

            # ── Find best black contour ───────────────────────────────────────
            valid_black_cnt = None
            black_cx        = None
            black_cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(black_cnts, key=cv2.contourArea, reverse=True):
                if 7500 <= cv2.contourArea(cnt) <= 40000:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        valid_black_cnt = cnt
                        black_cx        = int(M['m10'] / M['m00'])
                break

            # ── 90° detection ─────────────────────────────────────────────────
            color_is_horizontal = False
            bbox = None
            if valid_color_cnt is not None:
                bbox       = cv2.boundingRect(valid_color_cnt)
                x, y, w, h = bbox
                if h > 0 and w > (h * 2.5) and w > 150:
                    color_is_horizontal = True

            # ── State machine transitions ─────────────────────────────────────
            if state == STATE_TURN_90:
                elapsed_turn = time.perf_counter() - turn_90_start
                if elapsed_turn > TURN_90_LOCKOUT:
                    if valid_color_cnt is not None and not color_is_horizontal:
                        state = STATE_FOLLOW_COLOR
                    elif valid_black_cnt is not None:
                        state = STATE_FOLLOW_BLACK

            elif state == STATE_SEARCH:
                if valid_color_cnt is not None and not color_is_horizontal:
                    colour_entry_sign = -1 if cv2.countNonZero(colour_mask[:, :320]) > cv2.countNonZero(colour_mask[:, 320:]) else 1
                    state = STATE_FOLLOW_COLOR
                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK

            else:
                if color_is_horizontal:
                    left_px       = cv2.countNonZero(colour_mask[:, :320])
                    right_px      = cv2.countNonZero(colour_mask[:, 320:])
                    turn_90_dir   = "left" if left_px > right_px else "right"
                    turn_90_start = time.perf_counter()
                    state         = STATE_TURN_90

                elif valid_color_cnt is not None:
                    if state != STATE_FOLLOW_COLOR:
                        # Entry capture: which half has more colour pixels?
                        # LEFT dominant → camera over-rotated right → entered from RIGHT → sign=-1
                        # RIGHT dominant → camera over-rotated left → entered from LEFT → sign=+1
                        entry_left  = cv2.countNonZero(colour_mask[:, :320])
                        entry_right = cv2.countNonZero(colour_mask[:, 320:])
                        colour_entry_sign = -1 if entry_left > entry_right else 1
                        print(f"[Main] COLOUR ENTRY  L={entry_left} R={entry_right}  sign={colour_entry_sign}")
                    state = STATE_FOLLOW_COLOR

                elif valid_black_cnt is not None:
                    state = STATE_FOLLOW_BLACK

                elif state == STATE_FOLLOW_COLOR:
                    state = STATE_SEARCH
                    print(f"[Main] FOLLOW_COLOR → SEARCH  sign={colour_entry_sign}")

            # ── PID reset on state change ─────────────────────────────────────
            # Clear integral and derivative memory when switching contexts,
            # so historical corrections from one line type don't bias the other.
            if state != last_state:
                total_error = 0
                last_error  = 0
                first       = True
                last_state  = state

            # ── Motor control ─────────────────────────────────────────────────
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

            elif state == STATE_SEARCH:
                # colour_entry_sign=+1 → was going left  → sweep left  → turn_pwm negative
                # colour_entry_sign=-1 → was going right → sweep right → turn_pwm positive
                turn_pwm  = -SEARCH_SPEED * colour_entry_sign
                left_pwm  = base_speed + turn_pwm
                right_pwm = base_speed - turn_pwm

            elif state == STATE_TURN_90:
                turn_pwm  = -TURN_90_SPEED if turn_90_dir == "left" else TURN_90_SPEED
                left_pwm  = base_speed + turn_pwm
                right_pwm = base_speed - turn_pwm

            else:
                left_pwm  = base_speed
                right_pwm = base_speed

            movement.move(clamp(left_pwm, -1, 1), clamp(right_pwm, -1, 1))

            # ── Debug display (every 5th frame) ───────────────────────────────
            frame_count += 1
            if frame_count % 5 == 0:
                if valid_color_cnt is not None and bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 255), 2)

                cv2.putText(im2, f"STATE: {state}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(im2, f"sign: {colour_entry_sign}", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Show the latest shape detection result on screen.
                # This updates whenever the worker finishes a new result.
                detection_text = str(latest_detection) if latest_detection else "None"
                cv2.putText(im2, f"Object: {detection_text}", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

                cv2.imshow("1. Color Mask", colour_mask)
                cv2.imshow("2. Tracking & Math", im2)

                if cv2.waitKey(1) == 27:
                    break

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received.")

    finally:
        # ── Graceful shutdown sequence ────────────────────────────────────────
        # Order matters here: stop motors first (safety), then clean up resources.
        print("[Main] Stopping motors...")
        movement.move(0, 0)
        movement.pi.stop()

        print("[Main] Sending shutdown signal to worker...")
        # Put the None sentinel so the worker exits its blocking get() cleanly.
        # Use block=True with a timeout so we don't hang forever if the queue is full.
        try:
            frame_q.put(None, block=True, timeout=2)
        except Exception:
            pass

        print("[Main] Waiting for worker to finish (max 5 seconds)...")
        worker.join(timeout=5)

        # If the worker didn't exit cleanly within the timeout, force-terminate it.
        if worker.is_alive():
            print("[Main] Worker didn't stop cleanly — terminating forcefully.")
            worker.terminate()
            worker.join()

        print("[Main] Closing camera...")
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("[Main] Shutdown complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
# This guard is REQUIRED for multiprocessing on some platforms (particularly
# Windows, but also a good habit on Linux). Without it, spawning a child process
# would re-execute this entire file, which would spawn another child, which
# would spawn another... causing an infinite fork bomb.
if __name__ == "__main__":
    main()