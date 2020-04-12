
import cv2
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
import time
import sys

#tracker_types could be 'KCF' or 'GOTURN'
tracker_type = 'KCF'

if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()

video = cv2.VideoCapture("data/highway_small.mp4")

if not video.isOpened():
    print("Could not open video")

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')

# Decide the bounding box, press ESC to finish
bbox = cv2.selectROI(frame, False)
       
# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()