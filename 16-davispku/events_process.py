from multiprocessing import Process, Event, Queue
import multiprocessing as mp
import os
if os.name == 'nt': # Windows
    import keyboard
elif os.name == 'posix': # Linux or MacOS
    import signal
else:
    raise Exception("Unsupported OS")
import time
import numpy as np
import csv

import dv_processing as dv
# import cv2 as cv
from datetime import timedelta
from integrate_events_to_frame import integrate_events_to_one_frame_1bit_optimized_numpy

def events_process(events_timestamp_value1:mp.Value, image_array1:mp.Array, con1:mp.Condition, stop_event:mp.Event, is_camera:bool, frame_delay:int, file_path:str, time_sleep:float):
    try:
        # 尝试打开 CSV 文件
        csv_file =  open("events_process.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["perf_counter_ns", "log_type", "log_info", "events_timestamp"])
    except Exception as e:
        print(f"Error in events_process: {e}")
        exit(1)  # 终止子程序，返回状态码 1 表示异常退出

    # Open the camera or file
    if is_camera:
        reader = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)
    else:
        reader = dv.io.MonoCameraRecording(file_path)

    # Initialize a multi-stream slicer
    slicer = dv.EventStreamSlicer()

    # Callback method for time based slicing
    tim_total_1 = 0
    tim_total_2 = 0
    def display_preview(events):
        nonlocal tim_total_1, tim_total_2
        nonlocal events_timestamp_value1, image_array1, con1

        # Convert events to numpy array
        events_numpy = events.numpy()

        # Check if events_numpy is empty
        if len(events_numpy) == 0:
            # print("No events received. Skipping processing.")
            csv_writer.writerow([time.perf_counter_ns(), "LOG", "No events received. Skipping processing.", "None"])
            return
        events_timestamp = events_numpy[0][0] & 0xffffffff # Convert to 32-bit unsigned integer

        tim_total_1 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_1, "PULSE", tim_total_1 - tim_total_2, events_timestamp])

        image = integrate_events_to_one_frame_1bit_optimized_numpy(events_numpy)
        image = image[0] # image.shape = (86,65)

        # WRITE TO SHARED MEMORY
        with con1:
            events_timestamp_value1.value = events_timestamp
            # image_array[:] = image.flatten()
            np.copyto(np.frombuffer(image_array1.get_obj(), dtype=np.uint8), image.flatten())
            con1.notify_all()

        tim_total_2 = time.perf_counter_ns()
        csv_writer.writerow([tim_total_2, "TOTAL", tim_total_2 - tim_total_1, events_timestamp])

    # Register a job to be performed every frame_delay milliseconds
    slicer.doEveryTimeInterval(timedelta(milliseconds=frame_delay), display_preview)

    # Continue the loop while both cameras are connected
    while reader.isRunning():
        if stop_event.is_set():
            print("events_process stop")
            break
        # if not queue_full_flag:
        events = reader.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
        time.sleep(time_sleep)

    # close csv file
    csv_file.close()
    exit(0)
    # cv.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    stop_event = Event()
    def on_press_callback(event):
        if event.name == 'a':
            print('You pressed the A key')
        if event.name == 'q':
            stop_event.set()
            print('You pressed the Q key, exiting...')
    def signal_handler(signal, frame):
        stop_event.set()
    if os.name == 'nt': # Windows
        keyboard.on_press(on_press_callback)
    elif os.name == 'posix': # Linux or MacOS
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        raise Exception("Unsupported OS")

    IS_CAMERA = False
    FRAME_DELAY = 33
    FILE_PATH = "D:/DV/test/dvSave-2024_09_03_16_05_45.aedat4"
    TIME_SLEEP = 0

    events_timestamp_value1 = mp.Value('L', lock=True)
    image_array1 = mp.Array('B', 86*65, lock=True)
    con1 = mp.Condition()

    # Start the process
    p = Process(target=events_process, args=(events_timestamp_value1, image_array1, con1, stop_event, IS_CAMERA, FRAME_DELAY, FILE_PATH, TIME_SLEEP))
    p.start()

    events_timestamp_main = 0

    # Example of consuming data from the queue in the main process
    while p.is_alive():
        try:
            with con1:
                if events_timestamp_value1.value == events_timestamp_main:
                    con1.wait(timeout=2)
                events_timestamp_main = events_timestamp_value1.value
                image_main = np.frombuffer(image_array1.get_obj(), dtype=np.uint8).reshape((65, 86))

                # write image_main to file
                np.save("image_main.npy", image_main)
                print("Received events timestamp:", events_timestamp_main)
                print("Received image shape:", image_main)
            # else:
                # print("Queue is empty.")
        except Exception as e:
            print("Error in main process: ", e)
            pass

        time.sleep(0.01)

    p.join()
