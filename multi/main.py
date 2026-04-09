import object_detection
import line_following_v5 as line_following
import face_rec
import multiprocessing as mp
from multiprocessing import shared_memory
from picamera2 import Picamera2
import time
import numpy as np
import cv2

# changed exposure time from 5000
# changed analogue gain from 25.0


FRAME_SHAPE = (480, 640, 4)
FRAME_DTYPE = np.uint8
FRAME_NBYTES = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize

def camera_process(shm_name, lock, shape_event, line_event, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buf = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    picam2.set_controls({
        "ExposureTime": 3000,      # microseconds — try 2000-5000
        "AnalogueGain": 13.0,       # increase gain to compensate for less light
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
    cooldown_period = 4
    cooldown_start = -cooldown_period
    clear = False

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
            #print(f"time {cooldown_start}")
            if time.perf_counter() - cooldown_start > cooldown_period:
                if clear:
                    try:
                        item = result_q.get_nowait()
                    except mp.queues.Empty:
                        item = None
                    clear = False
                if not result_q.empty():
                    shape = result_q.get_nowait()
                    print(f"Detected: {shape}") # already handles shape detection
                    shape = shape[0]
                    action = decide_action(shape)

                    if action != shape:
                        if action == "Biometrics":
                            line_following.stop()
                            stop = False
                            while not stop:

                                stop = face_rec.recognize_face(frame)
                                line_event.wait()
                                line_event.clear()
                                with lock:
                                    frame = frame_buf.copy()
                                
                            cv2.destroyWindow("Face Recognition")

                            # must add cooldown afterwards to avoid triggering this again
                        elif action == "360 Turn":
                            line_following.turn_360()
                            print("turning 360")
                            
                        elif action == "Stop":
                            line_following.stop_for(5)

                        elif action in ("Left branch", "Right branch"):
                            direction = "left" if action == "Left branch" else "right"
                            line_following.force_blind_turn(direction)

                        else:
                            pass # follow branch
                        cooldown_start = time.perf_counter()
                    clear = True
            # always follow line regardless
            line_following.follow_line(frame) # we should pass left / right branch as parameter
            #print(f"duration2 {time.perf_counter() - time_marker2}")
    finally:
        shm.close()
        line_following.stop_forever()

def decide_action(shape):
    return {
        "Arrow (LEFT)": "Left branch",
        "Arrow (RIGHT)": "Right branch",
        #"Arrow (UP)": "Up branch",
        #"Arrow (DOWN)": "Down branch",
        "Recycle": "360 Turn",
        "Fingerprint": "Biometrics",
        "QR Code": "Biometrics",
        "Press Button": "Stop",
        "Danger": "Stop",
    }.get(shape, shape)

if __name__ == "__main__": # what was ran with python
    shm = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    result_q = mp.Queue(maxsize=1)

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