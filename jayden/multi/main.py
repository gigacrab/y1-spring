"""
robot_main.py  –  Multiprocessing entry point
===============================================
Architecture
------------
                        ┌─ lock_line  ─► shm_line  ◄─ lock_line  ─┐
  Picamera2 ──► Camera ─┤                                           ├─ LineFollower
                        └─ lock_shape ─► shm_shape ◄─ lock_shape ─┐│
                                                                    ││
                                                     ShapeDetector─┘│
                                                         │          │
                                                      result_q (maxsize=1)
                                                                    │
                                                       LineFollower─┘

Key properties
--------------
* Two independent SharedMemory blocks so the slow shape detector can never
  block the camera from writing a fresh frame for the line follower.
* The camera uses a non-blocking lock attempt for the shape buffer: if the
  shape detector is still reading, the camera simply skips that write —
  the line follower is NEVER affected.
* Graceful shutdown via atexit: shared memory is always unlinked even on
  KeyboardInterrupt or an unhandled exception in the main process.

Usage
-----
    python robot_main.py  base_speed  kp  ki  kd
    python robot_main.py  0.5  0.8  0.01  0.05
"""

import atexit
import sys
import time

import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

import object_detection
import line_following
import movement

# ── Shared memory geometry ─────────────────────────────────────────────────────
FRAME_SHAPE  = (480, 640, 3)
FRAME_DTYPE  = np.uint8
FRAME_NBYTES = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _attach_shm(name: str):
    """Attach to an existing block; return (shm, numpy_view)."""
    shm = shared_memory.SharedMemory(name=name)
    buf = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
    return shm, buf


# ──────────────────────────────────────────────────────────────────────────────
# Process targets
# ──────────────────────────────────────────────────────────────────────────────

def camera_process(shm_line_name:  str,
                   shm_shape_name: str,
                   lock_line:      mp.Lock,
                   lock_shape:     mp.Lock,
                   line_event:     mp.Event,
                   shape_event:    mp.Event,
                   stop_event:     mp.Event):
    """
    Capture frames as fast as Picamera2 allows.

    * Line buffer  : always written (blocking lock — contention is minimal
                     because the line follower only holds it for a memcpy).
    * Shape buffer : non-blocking acquire.  If the shape detector is still
                     reading, we skip this write and move on immediately —
                     the line follower is never affected.
    """
    from picamera2 import Picamera2

    shm_l, buf_l = _attach_shm(shm_line_name)
    shm_s, buf_s = _attach_shm(shm_shape_name)

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    try:
        while not stop_event.is_set():
            raw   = picam2.capture_array()                      # BGRA from video pipe
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)       # normalise once

            # ── Line-follower buffer (always written, fast) ────────────────────
            with lock_line:
                np.copyto(buf_l, frame)
            line_event.set()

            # ── Shape-detector buffer (skip if detector is busy) ───────────────
            if lock_shape.acquire(block=False):
                try:
                    np.copyto(buf_s, frame)
                finally:
                    lock_shape.release()
                shape_event.set()
            # else: shape detector still reading — just drop this frame silently

    finally:
        shm_l.close()
        shm_s.close()
        picam2.stop()
        picam2.close()


def shape_detection_process(shm_name:    str,
                             lock:        mp.Lock,
                             shape_event: mp.Event,
                             result_q:    mp.Queue,
                             stop_event:  mp.Event):
    shm, buf = _attach_shm(shm_name)

    try:
        while not stop_event.is_set():
            # Use a timeout so stop_event is checked even when idle
            if not shape_event.wait(timeout=1.0):
                continue
            shape_event.clear()

            with lock:                  # lock_shape — independent of lock_line
                frame = buf.copy()

            pred = object_detection.detect_object(frame)

            if pred is not None:
                # Keep the queue at exactly one item: the *latest* prediction.
                # Drain any stale entries before inserting so line follower
                # never acts on an outdated detection.
                while not result_q.empty():
                    try:
                        result_q.get_nowait()
                    except Exception:
                        break
                try:
                    result_q.put_nowait(pred)
                except Exception:
                    pass

    finally:
        shm.close()
        object_detection.stop()


def line_following_process(shm_name:   str,
                            lock:       mp.Lock,
                            line_event: mp.Event,
                            result_q:   mp.Queue,
                            stop_event: mp.Event,
                            base_speed: float,
                            kp:         float,
                            ki:         float,
                            kd:         float):
    shm, buf = _attach_shm(shm_name)
    follower = line_following.LineFollower(base_speed, kp, ki, kd)

    try:
        while not stop_event.is_set():
            if not line_event.wait(timeout=1.0):
                continue
            line_event.clear()

            with lock:                  # lock_line — independent of lock_shape
                frame = buf.copy()

            # Consume the latest shape prediction non-blocking
            # (drain loop in case result_q somehow has >1 item)
            shape = None
            while not result_q.empty():
                try:
                    shape = result_q.get_nowait()
                except Exception:
                    break
            if shape is not None:
                print(f"[LineFollower] Shape detected: {shape}")
                # TODO: pass `shape` into a decision layer here

            left_pwm, right_pwm = follower.process_frame(frame)
            movement.move(left_pwm, right_pwm)

            # ESC in the debug window stops the robot cleanly
            # (waitKey is already called inside follower._debug_display)

    finally:
        movement.move(0, 0)
        movement.pi.stop()
        shm.close()
        follower.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python robot_main.py  base_speed  kp  ki  kd\n"
            "Example: python robot_main.py  0.5  0.8  0.01  0.05"
        )

    base_speed = float(sys.argv[1])
    kp         = float(sys.argv[2])
    ki         = float(sys.argv[3])
    kd         = float(sys.argv[4])

    # ── Create independent shared-memory blocks (one per consumer) ─────────────
    shm_line  = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    shm_shape = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)

    # ── IPC primitives ─────────────────────────────────────────────────────────
    result_q    = mp.Queue(maxsize=1)   # holds at most the single latest prediction
    stop_event  = mp.Event()
    line_event  = mp.Event()
    shape_event = mp.Event()
    lock_line   = mp.Lock()             # guards shm_line  only
    lock_shape  = mp.Lock()             # guards shm_shape only  (independent!)

    processes = [
        mp.Process(
            target=camera_process,
            args=(shm_line.name, shm_shape.name,
                  lock_line, lock_shape,
                  line_event, shape_event,
                  stop_event),
            name="Camera",
            daemon=True,
        ),
        mp.Process(
            target=shape_detection_process,
            args=(shm_shape.name, lock_shape, shape_event, result_q, stop_event),
            name="ShapeDetector",
            daemon=True,
        ),
        mp.Process(
            target=line_following_process,
            args=(shm_line.name, lock_line, line_event, result_q, stop_event,
                  base_speed, kp, ki, kd),
            name="LineFollower",
            daemon=True,
        ),
    ]

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    # atexit fires on normal exit AND on KeyboardInterrupt, so the shared
    # memory is always unlinked — no RAM leak even on a crash.
    def _shutdown():
        print("\n[Main] Shutting down …")
        stop_event.set()
        line_event.set()    # unblock any blocking .wait() calls in workers
        shape_event.set()

        for p in processes:
            p.join(timeout=3)
            if p.is_alive():
                print(f"[Main] Force-terminating {p.name}")
                p.terminate()

        # Unlink in a finally-style guard so a crash here doesn't leave leaks
        for label, shm in [("shm_line", shm_line), ("shm_shape", shm_shape)]:
            try:
                shm.close()
                shm.unlink()
                print(f"[Main] {label} unlinked.")
            except Exception as exc:
                print(f"[Main] Warning: could not unlink {label}: {exc}")

    atexit.register(_shutdown)

    # ── Start & wait ───────────────────────────────────────────────────────────
    for p in processes:
        p.start()
        print(f"[Main] Started {p.name} (PID {p.pid})")

    try:
        # Block until all workers finish (they're daemons so this only returns
        # if they exit cleanly or stop_event terminates them).
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt — _shutdown() will run via atexit.")
    # _shutdown is registered with atexit and will always execute here.