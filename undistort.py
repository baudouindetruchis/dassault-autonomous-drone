import time
import cv2
import numpy as np
from datetime import datetime
import os

path_data = 'D:/code#/[large_data]/dassault/backgrounds_real/'

frame_height, frame_width = (None, None)

DIM = (720, 480)
K = np.array([[353.7, 0.0, 371.9], [0.0, 385.3, 291.5], [0.0, 0.0, 1.0]])
D = np.array([[-0.03728], [0.03595], [-0.09144], [0.07880]])

image_list = os.listdir(path_data)


for image in image_list:
    frame = cv2.imread(path_data + image)

    # Get image shape
    if (frame_height == None) or (frame_width == None):
        frame_height, frame_width = frame.shape[:2]

    # Undistort frame
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    frame_undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Save the frame
    cv2.imwrite(path_data + image, frame_undistorted)

    # Display the frame
    cv2.imshow('frame_undistorted', frame_undistorted)
    cv2.waitKey(0)

# Release capture
cv2.destroyAllWindows()
