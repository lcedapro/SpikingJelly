# 我现在要用multiprocessing模块，将下述代码的所有流程写入一个进程。该进程接受四个参数（IS_CAMERA, FRAME_DELAY, FILE_PATH, TIME_SLEEP），并不停的把（events_numpy_x_y_polarity, events_numpy_first_timestamp）放到一个Queue中。如果Queue已满，不仅需要在回调函数display_preview中停止写入Queue，还需要停止执行events = reader.getNextEventBatch()（接受USB包）这句话以防止内存溢出。请帮我修改代码。

import dv_processing as dv
import cv2 as cv
from datetime import timedelta
import time
import numpy as np

IS_CAMERA = False
FRAME_DELAY = 1
FILE_PATH = "D:/DV/test/dvSave-2024_09_03_16_05_45.aedat4"
TIME_SLEEP = 0.01

# Open the camera, just use first detected DAVIS camera
if IS_CAMERA:
    reader = dv.MonoCamera()
else:
    reader = dv.io.MonoCameraRecording(FILE_PATH)

# Initialize a multi-stream slicer
slicer = dv.EventStreamSlicer()

# Initialize a visualizer for the overlay
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution(), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red())

# Create a window for image display
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Callback method for time based slicing
def display_preview(events):
    # Generate a preview and show the final image
    cv.imshow("Preview", visualizer.generateImage(events))

    # If escape button is pressed (code 27 is escape key), exit the program cleanly
    if cv.waitKey(2) == 27:
        exit(0)

    # Convert events to numpy array
    events_numpy = events.numpy()
    print(events_numpy)
    # print(type(events_numpy)) # <class 'numpy.ndarray'>
    # print(type(events_numpy[0])) # <class 'numpy.void'> # event
    # print(type(events_numpy[0][0])) # <class 'numpy.int64'> # timestamp
    # print(type(events_numpy[0][1])) # <class 'numpy.int16'> # x
    # print(type(events_numpy[0][2])) # <class 'numpy.int16'> # y
    # print(type(events_numpy[0][3])) # <class 'numpy.int8'> # polarity

    # extract (x, y, polarity) from events
    events_numpy_x_y_polarity = ...
    # print(type(events_x_y_polarity)) # <class 'numpy.ndarray'>
    # print(type(events_x_y_polarity[0])) # <class 'numpy.void'> # event
    # print(type(events_x_y_polarity[0][0])) # <class 'numpy.int16'> # x
    # print(type(events_x_y_polarity[0][1])) # <class 'numpy.int16'> # y
    # print(type(events_x_y_polarity[0][2])) # <class 'numpy.int8'> # polarity

    # extract the first timestamp
    events_numpy_first_timestamp = events_numpy[0][0]
    # print(type(events_numpy_first_timestamp)) # <class 'numpy.int64'> # timestamp

    # return events_numpy_x_y_polarity and events_numpy_first_timestamp
    return events_numpy_x_y_polarity, events_numpy_first_timestamp

# Register a job to be performed every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=FRAME_DELAY), display_preview)

# Continue the loop while both cameras are connected
while reader.isRunning():
    events = reader.getNextEventBatch()
    if events is not None:
        slicer.accept(events)
    time.sleep(TIME_SLEEP)

# terminal output (events.numpy())(shape: ((timestamp, x, y, polarity), ...)
# [(1725350749441214, 43, 160, 0) (1725350749441219,  4, 236, 0)
#  (1725350749441224,  0, 180, 1) ... (1725350749474161, 11, 213, 1)
#  (1725350749474163,  1, 210, 1) (1725350749474163, 22, 222, 0)]

# preview window (cv.namedWindow)
#  (a window showing the events captured by the camera, visualized as green and red dots)