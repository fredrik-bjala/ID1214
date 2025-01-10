from ultralytics import YOLO
import cv2
import numpy as np



#############################################################################
# Change value for corresponding object, see object_list.txt for options
input_object = 0.
#############################################################################
# Change input image to detect object
img = cv2.imread("photos/bus.jpg")
#############################################################################
# Input color (B,G,R)
input_color = (0,255,255)
#############################################################################
# bgr color code
# blue = (255,0,0)
# green = (0,255,0)
# red = (0,0,255)
# yellow = (255, 255, 0)
# magenta = (255, 0, 255)
# orange = (0,160, 255)

#### Color overlay ##########################################################
overlay = np.zeros_like(img, np.uint8)
#### Gray overlay ###########################################################
overlay_gray = np.zeros_like(img, np.uint8)

#### Predict Object Model ###################################################
#Using Ultralytics yolov8m-seg model, can choose larger or smaller models by changing "m-"
model = YOLO("yolov8m-seg.pt")
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
conf = 0.5

#### Save objects found in results ##########################################
results = model.predict(img, conf=conf)

#### Copy of img in one channel and then back into 3 channels (BGR) ######### 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
#### Loop to go through all found objects matching the input and changing the overlay to match input color
for result in results:
    for i in range(0,len(result.boxes.cls)):
        if result.boxes.cls[i] == input_object:
            # print(result.boxes.cls[i])
            for mask, box in zip(result.masks[i].xy, result.boxes[i]):
                points = np.int32([mask])
                #### Gray out the object
                cv2.fillPoly(overlay_gray,points,input_color)
                #### Fill in the object with input color
                ##cv2.fillPoly(overlay, points, input_color)
                cv2.imshow('overlay gray',overlay_gray)
        

#### Change this to lower/increase transparency. Standard 0.5 ################
alpha = 0.5

gray_out = img.copy()
gray_mask = overlay_gray.astype(bool)
gray_out[gray_mask] = cv2.addWeighted(gray, alpha, overlay_gray, 1 - alpha, 0)[gray_mask]

cv2.imshow('gray',gray_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

