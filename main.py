import cv2
from ultralytics import YOLO

model = YOLO("../Yolo-Weights/ssss.pt")

results = model(0, show=True)

cv2.waitKey(0)

