"""
Real-time Object Detection with OpenCV and YOLOv4-tiny 
Copyright (C) 2020-2023 opsocket <opsocket@pm.me>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import cv2
import threading
import numpy as np

# Load COCO names
names = [ line.strip() for line in open("assets/coco/names","r").readlines() ]

# Load the YOLOv4 model
net = cv2.dnn.readNet('assets/yolov4-tiny/yolov4-tiny.weights', 'assets/yolov4-tiny/yolov4-tiny.cfg')

# retrieve output layers identifiers
layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# compute box colors
colors = np.random.uniform(50,255, size=(len(names), 3))

# open capture device
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to initialize video capture object")
    exit()

# font parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = .5
thickness = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # unpack frame's shape
    height, width, channels = frame.shape 
    
    # resize frame to neural net input shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), True, crop=False)
    
    # define neural net input
    net.setInput(blob)
    
    # run forward pass and filter results with low confidence
    outs = net.forward(outputlayers)
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > .25:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                boxes.append([int(center_x - w/2), int(center_y - h/2), w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform non-maximum suppression on boxes and corresponding scores to retrieve indexes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    # draw boxes and labels on frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            name = names[class_ids[i]]
            label = f"{name} {confidences[i] * 100:.2f}%"
            color = colors[names.index(name)]           
            # retrieve text size to compute background rectangle position 
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            # draw box
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            # draw label background
            cv2.rectangle(frame, (int(x-1), int(y)), (int(x+text_size[0]+10), int(y-text_size[1]-14)), color, -1)
            # draw label text
            cv2.putText(frame, label, (int(x+5), int(y-10)), font, font_scale, (0,0,0), thickness, cv2.LINE_AA, False)

    # display the resulting frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
