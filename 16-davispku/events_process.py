from multiprocessing import Process, Event, Queue
import os
if os.name == 'nt': # Windows
    import keyboard
elif os.name == 'posix': # Linux or MacOS
    import signal
else:
    raise Exception("Unsupported OS")
import time
import numpy as np

import dv_processing as dv
import cv2 as cv
from datetime import timedelta

def events_process(queue, stop_event, is_camera, frame_delay, file_path, time_sleep):
    # Open the camera or file
    if is_camera:
        reader = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)
    else:
        reader = dv.io.MonoCameraRecording(file_path)

    # Initialize a multi-stream slicer
    slicer = dv.EventStreamSlicer()

    # Initialize a visualizer for the overlay
    visualizer = dv.visualization.EventVisualizer(reader.getEventResolution(), 
                                                  dv.visualization.colors.white(),
                                                  dv.visualization.colors.green(), 
                                                  dv.visualization.colors.red())

    # Create a window for image display
    cv.namedWindow("Preview", cv.WINDOW_NORMAL)
    cv.resizeWindow("Preview", 800, 600)

    # Flag to indicate if the queue is full
    queue_full_flag = False

    # Callback method for time based slicing
    def display_preview(events):
        # Generate a preview and show the final image (if frame_time_counter == 0)
        cv.imshow("Preview", visualizer.generateImage(events))


        # If escape button is pressed (code 27 is escape key), exit the program cleanly
        if cv.waitKey(2) == 27:
            exit(0)

        # Convert events to numpy array
        events_numpy = events.numpy()

        # Check if events_numpy is empty
        if len(events_numpy) == 0:
            # print("No events received. Skipping processing.")
            return

        # Extract (x, y, polarity) from events
        events_numpy_x_y_polarity = np.array([(event[1], event[2], event[3]) for event in events_numpy]) # memory_ratio = 0.375

        # Extract the first timestamp
        events_numpy_first_timestamp = events_numpy[0][0]

        # Check if the queue is full
        if queue.full():
            queue_full_flag = True
            # print("Queue is full. Stopping event processing.")
        else:
            # queue.put((events_numpy_x_y_polarity, events_numpy_first_timestamp))
            queue.put(events_numpy_x_y_polarity)
            queue_full_flag = False
            # print("Queue is not full. Event data added to queue.")

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

    cv.destroyAllWindows()

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

    IS_CAMERA = True
    FRAME_DELAY = 33
    FILE_PATH = "D:/DV/test/dvSave-2024_09_03_16_05_45.aedat4"
    TIME_SLEEP = 0.01

    queue = Queue(maxsize=10)  # Set the maximum size of the queue

    # Start the process
    p = Process(target=events_process, args=(queue, stop_event, IS_CAMERA, FRAME_DELAY, FILE_PATH, TIME_SLEEP))
    p.start()

    # Example of consuming data from the queue in the main process
    while p.is_alive():
        
        try:
            if not queue.empty():
                data = queue.get(timeout=1)
                events_numpy_x_y_polarity, events_numpy_first_timestamp = data
                print("Received data:", len(events_numpy_x_y_polarity), events_numpy_first_timestamp)
            else:
                print("Queue is empty.")
        except:
            pass

        time.sleep(0.1)

    p.join()
