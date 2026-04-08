import object_detection
import line_following
import multiprocessing as mp
from multiprocessing import shared_memory
from picamera2 import Picamera2
import time
import numpy as np

FRAME_SHAPE = (480, 640, 4)
FRAME_DTYPE = np.uint8
FRAME_NBYTES = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize

def camera_process(shm_name, lock, shape_event, line_event, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buf = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.set_controls({
        "ExposureTime": 5000,      # microseconds — try 2000-5000
        "AnalogueGain": 10.0,       # increase gain to compensate for less light
        "AeEnable": False,          # disable auto exposure or it'll fight you
    })
    picam2.start()
    time.sleep(2)
    
    try:
        while not stop_event.is_set():
            frame = picam2.capture_array()
            with lock:
                np.copyto(frame_buf, frame)
            shape_event.set()
            line_event.set()
    
    finally:
        shm.close()
        picam2.stop()
        picam2.close()

def shape_detection_process(shm_name, lock, shape_event, result_q, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buf = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
    
    try:
        while not stop_event.is_set():
            shape_event.wait()
            shape_event.clear()
            with lock:
                frame = frame_buf.copy()
            
            pred = object_detection.detect_object(frame)

            if pred is not None and not result_q.full():
                result_q.put(pred)
    finally:
        shm.close()
        object_detection.stop()

def line_following_process(shm_name, lock, line_event, result_q, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buf = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)

    try:
        while not stop_event.is_set():
            #time_marker = time.perf_counter()
            
            line_event.wait()
            line_event.clear()
            with lock:
                frame = frame_buf.copy()
            
            #time_marker2 = time.perf_counter()
            #print(f"duration1 {time_marker2 - time_marker}")

            # check for new shape
            if not result_q.empty():
                shape = result_q.get_nowait()
                print(f"Detected: {shape}")

            # always follow line regardless
            line_following.follow_line(frame)
            #print(f"duration2 {time.perf_counter() - time_marker2}")
    finally:
        shm.close()
        line_following.stop()

'''def decide_action(shape):
    return {
        "Arrow":    "turn",
        "Octagon":  "stop",
        "Plus":     "choose_branch",
        # etc.
    }.get(shape, "follow")'''

if __name__ == "__main__": # what was ran with python
    shm = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    result_q = mp.Queue(maxsize=5)

    stop_event = mp.Event()
    shape_event = mp.Event()
    line_event = mp.Event()
    lock = mp.Lock()

    processes = [
        mp.Process(target=camera_process, 
                   args=(shm.name, lock, shape_event, line_event, stop_event)),
        mp.Process(target=shape_detection_process, 
                   args=(shm.name, lock, shape_event, result_q, stop_event)),
        mp.Process(target=line_following_process, 
                   args=(shm.name, lock, line_event, result_q, stop_event)),
    ]

    for p in processes:
        p.daemon = True  # dies with main process
        p.start()

    try:
        for p in processes:
            p.join() # blocks here
    except KeyboardInterrupt:
        stop_event.set()
        shape_event.set()
        line_event.set()
        '''for p in processes:
            p.terminate()'''
    finally:
        shm.close()
        shm.unlink() # deletes the shared memory, used only once