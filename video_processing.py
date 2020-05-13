import time
import cv2
import numpy as np
import os
import random

# ========== REQUIREMENTS ==========
# input size = 720x576px
#
# communication folder : pipeline/
# model files in folder : yolo_model/
# predictions in folder : predictions/
# ==================================

# ========== INFORMATION ===========
# 0.5 FPS on CPU locally
#
# dist_6k_v2 : 40/83 --> 48% true+ (5% false+)
# dist_3k_v3 : 76/83 --> 91% true+ (0% false+)
# dist_6k_v3 : 59/83 --> 71% true+ (1% false+)
# ==================================

# ========== FUNCTIONS ==========

def process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold):
	"""Process dnn outputs --> output selected boxes in pixels with upper-left corner as reference"""
	# Reset bounding boxes, class_ids & confidences
	boxes = []
	class_ids = []
	confidences = []

	for output in outputs:
		for detection in output:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = round(scores[class_id],2)

			if confidence > conf_threshold:
				# Scale bounding boxes back to frame
				box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
				(center_x, center_y, width, height) = box.astype(int)

				# Upper-left corner coordinates
				x = int(center_x - (width // 2))
				y = int(center_y - (height // 2))

				boxes.append([x, y, int(width), int(height)])
				class_ids.append(class_id)
				confidences.append(round(float(confidence),2))

	# Apply non-maxima suppression
	selected = np.array(cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)).flatten()		# NMSBoxes returns an empty tuple when no box
	for i in reversed(range(len(boxes))):
		if i not in selected:
			del boxes[i]
			del confidences[i]
			del class_ids[i]

	return boxes, confidences, class_ids

def draw_predictions(frame, boxes, confidences, class_ids, labels, colors):
	"""Take boxes in pixels with upper-left corner as reference --> draw bounding boxes on frame"""
	for i in range(len(boxes)):
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		color = [int(c) for c in colors[class_ids[i]]]

		# Draw bounding box
		cv2.rectangle(frame, (x, y), (x + w, y + h), color=color, thickness=1)

		# Print label + confidence
		text = str(labels[class_ids[i]]) + ' ' + str(confidences[i])
		(text_width, text_height) = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, thickness=1)[0]
		cv2.rectangle(frame, (x, y-text_height-1), (x+text_width, y), color=color, thickness=cv2.FILLED)
		cv2.putText(frame, text, org=(x, y-1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0,0,0), thickness=1)

	return frame

def pipeline(boxes, confidences, class_ids, labels, frame_height, frame_width, path_folder):
	"""Push to pipeline detected shapes in .txt file"""
	if len(boxes) > 0:
		with open(path_folder + 'pipeline/detection.txt', 'w') as file:
			file.write('detected\n')

			for i in range(len(boxes)):
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# Coordinates relative to frame center
				relative_x = x - (frame_width//2) + (w//2)
				relative_y = -y + (frame_height//2) - (h//2)

				# Write detected label + relative to center coordinates
				file.write(str(labels[class_ids[i]]) + ' ' + str(relative_x) + ' ' + str(relative_y) + '\n')
	else:
		with open(path_folder + 'pipeline/detection.txt', 'w') as file:
			file.write('nothing\n')

# ========== YOLO SETUP ==========

path_folder = 'D:/code#/[large_data]/dassault/'
filename_weights = 'yolov3_custom_train_3000_v3.weights'

# Labels & color setup
labels = open(path_folder + 'yolo_model/yolo.names').read().strip().split("\n")
colors = np.array([[0, 20, 229],
				   [135, 118, 100],
				   [37, 0, 162],
				   [115, 0, 216],
				   [208, 114, 244],
				   [239, 80, 0],
				   [0, 200, 227]])

# Load model
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(path_folder + 'yolo_model/yolov3_custom_train.cfg', path_folder + 'yolo_model/' + filename_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Parameters setup
conf_threshold = 0.25	# Confidence minimum threshold
nms_threshold = 0.1		# Non-maximum suppression threshold : overlap maximum threshold

# ========== DISTORTION SETUP ==========

frame_height, frame_width = (None, None)

DIM = (720, 480)
K = np.array([[353.7, 0.0, 371.9], [0.0, 385.3, 291.5], [0.0, 0.0, 1.0]])
D = np.array([[-0.03728], [0.03595], [-0.09144], [0.07880]])


# ========== RUNNING ==========

while True:
	start = time.time()

	# Read raw image
	image_list = os.listdir('D:/code#/[large_data]/dassault/real_dataset/dataset_redarrow_ISEP_first')
	frame = cv2.imread(path_folder + 'real_dataset/dataset_redarrow_ISEP_first/' + random.choice(image_list))
	# frame = cv2.imread(path_folder + 'pipeline/capture_raw.jpg')
	if frame is None:
		time.sleep(0.1)
		print('[WARNING] cannot load input image')
		continue

	# Get image shape
	if (frame_height == None) or (frame_width == None):
		frame_height, frame_width = frame.shape[:2]

	# Undistort frame
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
	# frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	# Transform frame in 416x416 blob + forward pass
	blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(ln)

	# Post-processing
	boxes, confidences, class_ids = process_outputs(outputs, frame_width, frame_height, conf_threshold, nms_threshold)
	frame = draw_predictions(frame, boxes, confidences, class_ids, labels, colors)

	# Compute fps rate
	end = time.time()
	fps = str(round((1/(end-start)),1)) + ' fps'
	cv2.putText(frame, fps, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)

	# Display the frame
	cv2.imshow('frame_undistorted', frame)

	# Communicate through pipeline
	pipeline(boxes, confidences, class_ids, labels, frame_height, frame_width, path_folder)
	cv2.imwrite(path_folder + 'pipeline/capture_processed.jpg', frame)

	# Save predictions
	timestamp = str(int(time.time())) + '_' + str(int(time.time()*1000))[-3:]
	cv2.imwrite(path_folder + 'predictions/prediction_' + timestamp + '.jpg', frame)

	# Check key press
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release capture
cv2.destroyAllWindows()
