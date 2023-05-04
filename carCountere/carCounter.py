from ultralytics import YOLO
import cv2
import cvzone
import math
import _tkinter
import tkinter
from sort import *

# TRACKING
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3) # iou = overlap of bounding boxes

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/cars.mp4")
# cap.set(3, 1280) # set height + width
# cap.set(4, 720)

model = YOLO('../yolo-weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("../Images/mask.png")

while True:
    success, img = cap.read() # bool, tensors of frame
    # overlay image on video
    imgRegion = cv2.bitwise_and(img, mask) # put image on mask

    results = model(img, stream=True) # stream=True -> more efficient

    for r in results:
        # get bounding box of each result
        boxes = r.boxes

        for box in boxes:
            # find x, y of each box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9) # creating bounding boxes

            conf = math.ceil((box.conf[0]*100)) / 100
            clss = box.cls # gives tensor
            cls = int(box.cls[0]) # gives class' tensor - need to index on class labels to find correect label
            
            currentClass = classNames[cls] # returns the label based off index

            if currentClass.lower() == "car" or currentClass.lower() == "truck" or currentClass.lower() == "bus"\
                    or currentClass.lower() == "motorbike" and conf > 0.3: # if id is only car, > 30%
                cvzone.putTextRect(img, f'class: {classNames[cls]}: {conf}%', (max(0, x1), max(35, y1)), scale=0.5) # putting text values on screen
            # prevent less than 0 values on x
                cvzone.cornerRect(img, (x1, y1, w, h), l=9) # creating bounding boxes
                currentArray = np.array([x1, y1, x2, y2, conf]) # params, score
                detections = np.vstack((detections, currentArray)) # stacking arrays

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        print("reesult", result)

    cv2.imshow("Image", img)
    cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)

