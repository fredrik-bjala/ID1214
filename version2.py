from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch
#############################################################################
#### Change value for corresponding object, see object_list.txt for options
input_object = 5.
#############################################################################
#### Change input image to detect object
img = cv2.imread("photos/bus.jpg")
#############################################################################


#### Color prediction models paths###########################################
protoTxtPath = 'models/colorization_deploy_v2.prototxt'
modelPath = 'models/colorization_release_v2.caffemodel'
kernelPath = 'models/pts_in_hull.npy'

#### Predict Color Model ####################################################
net = cv2.dnn.readNetFromCaffe(protoTxtPath, modelPath)
color_points = np.load(kernelPath)
color_points = color_points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [color_points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313, 1, 1], 2.606, np.float32)]

normed = img.astype(np.float32) / 255.0
lab = cv2.cvtColor(normed, cv2.COLOR_BGR2Lab)
resized = cv2.resize(lab, (255, 255))
l = cv2.split(resized)[0]
l -= 50

net.setInput(cv2.dnn.blobFromImage(l))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

l = cv2.split(lab)[0]
colorized = np.concatenate((l[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = (255 * np.clip(colorized, 0, 1)).astype(np.uint8)
#############################################################################
#### File to temporary store the color prediction
cv2.imwrite('photos/Colorized_image.jpg', colorized)

#### Predict Object Model ###################################################
#Using Ultralytics yolov8m-seg model, can choose larger or smaller models by changing "m-"   
model = YOLO("yolov8m-seg.pt") 
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.5
#### Save objects found in results
results = model.predict(img, conf=conf)

#Duplicate to temporary work with masks and not change original image
object_mask = img.copy()
background = img.copy()

fill_color = [0,0,0]
mask_value = 255

#Loop to go through all objects found in img
for result in results:
    #Loop to cut out all the objects from img copy (background)  
    for i in range(0,len(result.boxes.cls)):
        if result.boxes.cls[i] == input_object:
            for mask, box in zip(result.masks[i].xy, result.boxes[i]):
                points = np.int32([mask])                           
                cv2.fillPoly(background, points, fill_color)
    #Loop to only select all the selected objects images from predict_color and then add them to background                  
    for i in range(0,len(result.boxes.cls)):
        predicted_color = cv2.imread('photos/Colorized_image.jpg')
        if result.boxes.cls[i] == input_object:   
            for mask, box in zip(result.masks[i].xy, result.boxes[i]):
                #Create mask for object
                points = np.int32([mask])
                #Remove all other colors from anything that is not the object
                stencil =np.zeros(predicted_color.shape[:-1]).astype(np.uint8)
                cv2.fillPoly(stencil, points, mask_value)
                sel = stencil != mask_value 
                predicted_color[sel] = fill_color  
                #Add the cut out mask to background
                background = cv2.add(background,predicted_color)
              
cv2.imshow("final",background)
#Save the final image to the path as a jpg
cv2.imwrite('photos/final.jpg',background)
cv2.waitKey(0)
cv2.destroyAllWindows() 
