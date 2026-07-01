import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "model", "final_emotion_model.keras")
FALLBACK_VIDEO = os.path.join(BASE_DIR, "results", "preprocessing_test", "input_video.mp4")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(f"Webcam unavailable, falling back to {FALLBACK_VIDEO}", file=sys.stderr)
    cap = cv2.VideoCapture(FALLBACK_VIDEO)
    if not cap.isOpened():
        print("No video source available", file=sys.stderr)
        sys.exit(1)

print("Reading video stream ...")
print()

last_pred = 0.0          
last_label = None        
last_confidence = 0
last_box = None        

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        now = time.monotonic()
        if now - last_pred >= 1.0:
            last_pred = now
            print("Preprocessing ...")
            ts = time.strftime("%H:%M:%S")

            if len(faces) == 0:
                print(f"{ts}s : NoFace , 0%")
                print()
                last_label = None
                last_box = None
            else:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48)).astype("float32") / 255.0
                face = face.reshape(1, 48, 48, 1)

                pred = model.predict(face, verbose=0)[0]
                idx = int(np.argmax(pred))
                last_label = EMOTIONS[idx]
                last_confidence = int(round(float(pred[idx]) * 100))
                last_box = (x, y, w, h)

                print(f"{ts}s : {last_label} , {last_confidence}%")
                print()

        if last_box is not None and last_label is not None:
            x, y, w, h = last_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{last_label} : {last_confidence}%"
            cv2.putText(frame, text, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detector (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
