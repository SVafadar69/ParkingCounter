from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../Videos/motorbikes.mp4")
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

while True:
    success, img = cap.read() # bool, tensors of frame

    results = model(img, stream=True) # stream=True -> more efficient

    for r in results:
        # get bounding box of each result
        boxes = r.boxes

        for box in boxes:
            # find x, y of each box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h)) # creating bounding boxes

            conf = math.ceil((box.conf[0]*100)) / 100
            clss = box.cls # gives tensor
            cls = int(box.cls[0]) # gives class' tensor - need to index on class labels to find correect label
            print("Class id", cls)
            cvzone.putTextRect(img, f'class: {classNames[cls]}: {conf}%', (max(0, x1), max(35, y1)), scale=0.5) # putting text values on screen
            # prevent less than 0 values on x

            print("Confidence", conf)



    cv2.imshow("Image", img)
    cv2.waitKey(1)

