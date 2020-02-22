import time
import cv2
import numpy as np
from datetime import datetime
import imutils
import os


# ========== FUNCTIONS ==========

def process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold):
	# Reset bounding boxes, confidences
	boxes = []
	confidences = []
	class_ids = []

	for output in outputs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]

			if confidence > conf_threshold:
				# Scale bounding boxes back to frame
				box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
				(center_x, center_y, width, height) = box.astype("int")

				# Upper-left corner
				x = int(center_x - (width / 2))
				y = int(center_y - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	# Apply non-maxima suppression
	selected = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	return boxes, confidences, class_ids, selected

def draw_predictions(frame, boxes, confidences, class_ids, selected, labels, colors):
	if len(selected) > 0:
		for i in selected.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in colors[class_ids[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.2f}".format(labels[class_ids[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

	return frame


# ========== YOLO SETUP ==========

label_path = '/media/bdn/Data/code#/[large_data]/dassault/MODEL/yolo.names'
config_path = '/media/bdn/Data/code#/[large_data]/dassault/MODEL/yolov3_custom_train.cfg'
weights_path = '/media/bdn/Data/code#/[large_data]/dassault/MODEL/yolov3_custom_train_6000_v2.weights'
# video_input_path = '../[large_data]/airport.mp4'
# writer_path = '../../../Desktop/yolo_adrien_video.avi'

# Import labels & color setup
labels = open(label_path).read().strip().split("\n")
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# Load model
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Parameters setup
conf_threshold = 0.25    # Confidence threshold
nms_threshold = 0.5    # Non-maximum suppression threshold


# ========== DISTORTION SETUP ==========

frame_height, frame_width = (None, None)

DIM = (720, 480)
K = np.array([[353.7, 0.0, 371.9], [0.0, 385.3, 291.5], [0.0, 0.0, 1.0]])
D = np.array([[-0.03728], [0.03595], [-0.09144], [0.07880]])


# ========== RUNNING ==========

# cap = cv2.VideoCapture(0)

path = '/media/bdn/Data/code#/[large_data]/dassault/generated_1/'
# path = '/media/bdn/Data/code#/[large_data]/dassault/red_arrow_real/'
images_list = os.listdir(path)
count = 0

while True:
	start = time.time()

	frame = cv2.imread(path + images_list[count])
	count = count + 1

	# time.sleep(1)


	# Capture frame-by-frame
	# grabbed, frame = cap.read()
	# if not grabbed:
	# 	break

	# Get image shape
	if (frame_height == None) or (frame_width == None):
		frame_height, frame_width = frame.shape[:2]

	# Undistort frame
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
	frame_undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	# Transform frame in 416x416 blob
	blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

	# Forward pass
	net.setInput(blob)
	outputs = net.forward(ln)

	# Post-processing
	boxes, confidences, class_ids, selected = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)
	frame = draw_predictions(frame, boxes, confidences, class_ids, selected, labels, colors)

	# Save the frame
	# date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	# cv2.imwrite('dataset/testflight_undist_' + date + '.jpg', frame_undistorted)

	# Compute fps rate
	end = time.time()
	fps = str(round((1/(end-start)),1)) + ' fps'

	# Display the frame
	cv2.putText(frame, fps, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
	cv2.imshow('frame_undistorted', frame)

	# Check key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release capture
cap.release()
cv2.destroyAllWindows()
