from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8l.pt') # n = nano, l = large (bigger box)

results = model('../Images/3.png', show=True) # show = see img
cv2.waitKey(0)

