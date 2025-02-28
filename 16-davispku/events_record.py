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

WRITER_FILE_PATH = "mono_writer_sample.aedat4"
FRAME_DELAY = 33
TIME_SLEEP = 0

# Sample VGA resolution, same as the DAVIS346 camera
resolution = (346, 260)

# Open the camera
capture = dv.io.CameraCapture("", dv.io.CameraCapture.CameraType.DAVIS)

# Get the camera name
# name = dv.io.CameraCapture.getCameraName()

# Event only configuration
config = dv.io.MonoCameraWriter.DAVISConfig("DAVIS346", resolution)

# Create the writer instance, it will only have a single event output stream.
writer = dv.io.MonoCameraWriter(WRITER_FILE_PATH, config)

# Initialize a multi-stream slicer
slicer = dv.EventStreamSlicer()

# Initialize a visualizer for the overlay
visualizer = dv.visualization.EventVisualizer(capture.getEventResolution(), 
                                                  dv.visualization.colors.white(),
                                                  dv.visualization.colors.green(), 
                                                  dv.visualization.colors.red())

# Create a window for image display
cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.resizeWindow("Preview", 800, 600)

# Callback method for time based slicing
def display_preview(events):
    # Generate a preview and show the final image (if frame_time_counter == 0)
    cv.imshow("Preview", visualizer.generateImage(events))

    # If escape button is pressed (code 27 is escape key), exit the program cleanly
    if cv.waitKey(1) == 27:
            exit(0)

    # Write the packet using the writer, the data is not going be written at the exact
    # time of the call to this function, it is only guaranteed to be written after
    # the writer instance is destroyed (destructor has completed)
    writer.writeEvents(events)

# Register a job to be performed every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=FRAME_DELAY), display_preview)

# Continue the loop while both cameras are connected
while capture.isRunning():
    events = capture.getNextEventBatch()
    if events is not None:
        slicer.accept(events)
    time.sleep(TIME_SLEEP)