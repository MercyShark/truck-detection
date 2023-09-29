import cv2
from ultralytics import YOLO
import cvzone
import math

model = YOLO("./weights/yolov8x.pt")
camera = cv2.VideoCapture("./assets/demo3.mp4")

while True:
    success, image = camera.read()
    results = model.predict(image, stream=True)
    for result in results:
        boxes = result.boxes
        allObjects = result.names
        counter = 0
        for box in boxes:
            currentClass = int(box.cls[0])
            currentClassName = allObjects[currentClass]
            conf = math.ceil(box.conf[0] * 100) / 100
            if currentClassName == "truck" and conf > 0.3:
                counter = counter + 1
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width = x2 - x1
                height = y2 - y1
                cvzone.cornerRect(image, (x1, y1, width, height), l=1)
                conf = math.ceil(box.conf[0] * 100) / 100
                print("confidence level :", conf)
                print("classification number:", int(currentClass))
                cvzone.putTextRect(
                    image,
                    f"{conf} : {currentClassName}",
                    (max(0, x1), max(35, y1)),
                    scale=0.6,
                    thickness=1,
                )
                print("Counter : ", counter)
    cv2.imshow("Team Tracko", image)
    cv2.waitKey(1)
