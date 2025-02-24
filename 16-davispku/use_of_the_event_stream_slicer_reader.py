import dv_processing as dv
import cv2 as cv
from datetime import timedelta

# Open the camera, just use first detected DAVIS camera
reader = dv.io.MonoCameraRecording("D:/DV/test/dvSave-2024_09_03_16_05_45.aedat4")

# Initialize a multi-stream slicer
slicer = dv.EventStreamSlicer()

# Initialize a visualizer for the overlay
visualizer = dv.visualization.EventVisualizer(reader.getEventResolution(), dv.visualization.colors.white(),
                                              dv.visualization.colors.green(), dv.visualization.colors.red())

# Create a window for image display
cv.namedWindow("Preview", cv.WINDOW_NORMAL)

# Callback method for time based slicing
def display_preview(data):
    # Retrieve event data
    events = data

    # Generate a preview and show the final image
    cv.imshow("Preview", visualizer.generateImage(events))

    # If escape button is pressed (code 27 is escape key), exit the program cleanly
    if cv.waitKey(2) == 27:
        exit(0)
    
    # return events numpy
    print(events.numpy())

# Register a job to be performed every 33 milliseconds
slicer.doEveryTimeInterval(timedelta(milliseconds=33), display_preview)

# Continue the loop while both cameras are connected
while reader.isRunning():
    events = reader.getNextEventBatch()
    if events is not None:
        slicer.accept(events)

# terminal output (events.numpy())
#  (1725350750505579,  88, 232, 0) (1725350750505614,  88, 238, 1)
#  (1725350750505653,  66, 238, 1) (1725350750505655,  64, 234, 1)
#  (1725350750505680,  71, 240, 1) (1725350750505719,  96, 207, 1)
#  (1725350750505727,  87, 248, 1) (1725350750505742,  81, 246, 1)
#  (1725350750505830,  90, 235, 1) (1725350750505831,  88, 246, 1)
#  (1725350750505887,  77, 247, 0) (1725350750505898,  61, 231, 1)
#  (1725350750505939,  73, 227, 1) (1725350750505981,  68, 242, 1)
#  (1725350750505997,  67, 241, 1) (1725350750506034,  68, 236, 1)
# ...

# preview window (cv.namedWindow)
#  (a window showing the events captured by the camera, visualized as green and red dots)