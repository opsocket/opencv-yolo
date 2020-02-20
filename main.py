import cv2
import numpy as np

# Define a capture device
cam = cv2.VideoCapture(0)

# Load YOLOv3-320 (mAP: 51.5 / FPS: 45)
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

# Load COCO names
names = []
with open("coco.names","r") as f:
	names = [ line.strip() for line in f.readlines() ]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255, size=(len(names), 3))

while(True):
	# Capture frame-by-frame
	ret, frame = cam.read()
	# Our operations on the frame start here
	height,width,channels = frame.shape
	# Create blob from frame
	blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False)
	# Set blob as net input source
	net.setInput(blob)
	# Runs forward pass
	outs = net.forward(outputlayers)
	# Filters results with low uncertainty
	name_ids, confidences, boxes = [], [], []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			name_id = np.argmax(scores)
			confidence = scores[name_id]
			if confidence > 0.5:
				center_x, center_y = int(detection[0]*width), int(detection[1]*height) # Compute center position
				w, h = int(detection[2]*width), int(detection[3]*height) # Compute dimensions
				x, y = int(center_x - w/2), int(center_y - h/2) # Compute origin
				boxes.append([x,y,w,h]) # Append object's box position
				confidences.append(float(confidence)) # Append object's confidence percentage
				name_ids.append(name_id) # Append object's class id

	# Performs non maximum suppression given boxes and corresponding scores to retrieve indexes
	indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
	# Append boxes, confidence and label on frame
	for i in range(len(boxes)):
		if i in indexes:
			x,y,w,h = boxes[i]
			label = str(names[name_ids[i]])
			cv2.rectangle(frame,(x,y),(x+w,y+h),colors[i],2)
			cv2.putText(frame,"%s %d %%" % (label, confidences[i]*100),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),1)

	# Show frame
	cv2.imshow("Detection", frame)
	# Sleep for 1ms to process the draw requests from cv2.imshow()
	cv2.waitKey(1)

# When everything done, release the capture and destroy all windows
cam.release()
cv2.destroyAllWindows()