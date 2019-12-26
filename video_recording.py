import time
import cv2
import numpy as np
from datetime import datetime

cap = cv2.VideoCapture(0)

frame_height, frame_width = (None, None)

DIM = (720, 480)
K = np.array([[353.7, 0.0, 371.9], [0.0, 385.3, 291.5], [0.0, 0.0, 1.0]])
D = np.array([[-0.03728], [0.03595], [-0.09144], [0.07880]])


while True:
	start = time.time()

	# Capture frame-by-frame
	grabbed, frame = cap.read()
	if not grabbed:
		break

	# Get image shape
	if (frame_height == None) or (frame_width == None):
		frame_height, frame_width = frame.shape[:2]

    # Undistort frame
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
	frame_undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	# Compute fps rate
	end = time.time()
	fps = str(round((1/(end-start)),1)) + ' fps'

	# Display the frame
	cv2.putText(frame_undistorted, fps, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
	cv2.imshow('frame_undistorted', frame_undistorted)

    # Save the frame
	# date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # cv2.imwrite('dataset/testflight_undist_' + date + '.jpg', frame_undistorted)

    # Check key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release capture
cap.release()
cv2.destroyAllWindows()
