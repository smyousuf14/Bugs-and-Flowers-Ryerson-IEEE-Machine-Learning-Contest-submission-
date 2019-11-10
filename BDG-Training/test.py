#Import statements
#First it is necessary to import all the required packages
import argparse
import time
import numpy as np
import os
import cv2

#google colab required package

# Next set the values for confidence and threshold
confidence = 0.5
threshold = 0.3

#Store values to determine if there are bugs/flowers in the image 
bugs = 0
flowers = 0

#Load the coco class labels that show all the avaliable classes that our YOLOv3 model was trained for.
labelsPath = "yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of random colours
np.random.seed(42)
COLORS = np.random.randint(0, 255, size =(len(LABELS), 3), dtype = "uint8")

#Gets the paths to the YOLO weights and model configuration. The weights is what we obtained after training. 
weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"

#load our yolo object detector trained on the custom dataset of images, around 100 images of bugs and flowers. 
print("[MESSAGE] loading YOLO Please wait...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get the input image from the user and load the input image. The image must be stored in the images folder
image_destination =   input("Enter the name of the input image: ");
image = cv2.imread("images/" + image_destination + ".jpg")

image = cv2.resize(image,(600,500))
(H, W) = image.shape[:2]

# determine only the output layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass. This will give us our bounding boxes and associated percentage error
blob = cv2.dnn.blobFromImage(image, 1 / 255.0 , (416,416), swapRB = True, crop = False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show the timing information on YOLO
print("The YOLO model took {:.6f} seconds".format(end - start))

# Initialize our list of detected bounding boxes, confidences, and class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
    #loop over each of the detections
    for detection in output:
        #Its time to extract the class IDs and confidence of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        #filter out the weak predictions by ensuring the detected probability is greater than the minimum probability
        if confidence > 0.80:

            # scale the bounding box coordinates back relative to the
            # size of the image,
            box = detection[0:4] * np.array([W,H,W,H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)- coordinates to get the top and left corner of the bounding box
            x = int(centerX - (width/2))
            y = int(centerY - (height / 2))

            #update our list of bounding box coordinates, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

#apply non-maxima supression to supress weak and/or overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

# ensure that at least one detection exists before attempting to make any bounding boxes.
if len(idxs) > 0:
    #loop over the indexes we are keeping
    for i in idxs.flatten():
        #extract the bounding box coordinates
        (x,y) = (boxes[i][0], boxes[i][1])
        (w,h) = (boxes[i][2], boxes[i][3])

        #draw a bounding box rectangle and label on the image.
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x,y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Check if it is a bug or flower
        if LABELS[classIDs[i]] == "bug":
          bugs = bugs + 1
        elif LABELS[classIDs[i]] == "flower":
          flowers = flowers + 1

#Now print out the message depending on if the what the image contains.
if bugs == 0 and flowers == 0:
  cv2.putText(image, "The image contains neither a bug nor a flower", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2 )
elif bugs == 0 and flowers >= 1:
  cv2.putText(image, "The image contains a flower and no bug", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2 )
elif bugs  >= 1 and flowers == 0:
  cv2.putText(image, "The image contains a bug and no flower", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2 )
elif bugs >= 1 and flowers >= 1:
  cv2.putText(image, "The image contains a flower and a bug", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2 )
          

#show the output image
cv2.imshow("image", image)
cv2.waitKey(0)