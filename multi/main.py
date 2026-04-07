import object_detection
import line_following
import multiprocessing as mp
from picamera2 import Picamera2
import time

def camera_process(frame_q_shape, frame_q_line, stop_event):
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    while not stop_event.is_set():
        frame = picam2.capture_array()

        # Doesn't block if a queue is full, drops the current frame instead
        if not frame_q_shape.full():
            frame_q_shape.put(frame)
        if not frame_q_line.full():
            frame_q_line.put(frame)
    
    picam2.stop()
    picam2.close()

def shape_detect_process(frame_q, result_q, stop_event):
    while not stop_event.is_set():
        frame = frame_q.get()  # blocks until frame available

        pred = object_detection.detect_object(frame)

        if pred is not None and not result_q.full():
            result_q.put(pred)

    object_detection.stop()

def line_follow_process(frame_q, result_q, stop_event):
    current_action = "follow"  # default state

    while not stop_event.is_set():
        frame = frame_q.get()

        # check for new shape
        if not result_q.empty():
            shape = result_q.get_nowait()
            print(f"Detected: {shape}")

        # always follow line regardless
        line_following.follow_line(frame)

    line_following.stop()

'''def decide_action(shape):
    return {
        "Arrow":    "turn",
        "Octagon":  "stop",
        "Plus":     "choose_branch",
        # etc.
    }.get(shape, "follow")'''

if __name__ == "__main__": # what was ran with python
    frame_q_shape = mp.Queue(maxsize=2)
    frame_q_line = mp.Queue(maxsize=2)
    result_q = mp.Queue(maxsize=5)

    stop_event = mp.Event()

    processes = [
        mp.Process(target=camera_process, args=(frame_q_shape, frame_q_line, stop_event)),
        mp.Process(target=shape_detect_process, args=(frame_q_shape, result_q, stop_event)),
        mp.Process(target=line_follow_process, args=(frame_q_line,  result_q, stop_event)),
    ]

    for p in processes:
        p.daemon = True  # dies with main process
        p.start()

    try:
        for p in processes:
            p.join() # blocks here
    except KeyboardInterrupt:
        stop_event.set()
        '''for p in processes:
            p.terminate()'''