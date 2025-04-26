import cv2
import time
import json
import os
import numpy as np

class ForgottenObjectDetector:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = json.load(f)

        self.prototxt_path = config["prototxt_path"]
        self.model_path = config["model_path"]
        self.confidence_threshold = config.get("confidence_threshold", 0.5)

        print(f"Prototxt path: {self.prototxt_path}")
        print(f"Model path: {self.model_path}")

        if not os.path.exists(self.prototxt_path):
            print(f"File {self.prototxt_path} does not exist")
        if not os.path.exists(self.model_path):
            print(f"File {self.model_path} does not exist")

        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не вдалося відкрити камеру")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()