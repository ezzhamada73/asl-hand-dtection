import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Model configuration paths (relative paths for portability)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_DETECTION_PATH = os.path.join(BASE_DIR, "best.pt")
ASL_MODEL_PATH = os.path.join(BASE_DIR, "best_asl_model.keras")

INPUT_SIZE = (224, 224)
VIDEO_SOURCE = 0
CROP_PADDING = 20

CLASS_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

print("Loading models...")
hand_detector = YOLO(HAND_DETECTION_PATH)
asl_classifier = tf.keras.models.load_model(ASL_MODEL_PATH)
print("Models loaded successfully.")


def preprocess_crop(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, INPUT_SIZE)
    crop = crop.astype("float32")
    crop = preprocess_input(crop)
    crop = np.expand_dims(crop, axis=0)
    return crop


def classify_hand(crop):
    batch = preprocess_crop(crop)
    preds = asl_classifier.predict(batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else "?"
    return label, confidence


def draw_prediction(frame, box, label, confidence):
    x1, y1, x2, y2 = box
    color = (0, 200, 0) if confidence >= 0.6 else (0, 165, 255)
    text = f"{label} ({confidence * 100:.0f}%)" if confidence >= 0.6 else "?"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"couldnt open video source {VIDEO_SOURCE}")
        return

    frame_h, frame_w = None, None
    print("running , press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_h is None:
            frame_h, frame_w = frame.shape[:2]

        results = hand_detector.predict(frame, conf=0.6, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            x1 = max(0, x1 - CROP_PADDING)
            y1 = max(0, y1 - CROP_PADDING)
            x2 = min(frame_w, x2 + CROP_PADDING)
            y2 = min(frame_h, y2 + CROP_PADDING)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label, confidence = classify_hand(crop)
            draw_prediction(frame, (x1, y1, x2, y2), label, confidence)

        cv2.imshow("asl detection (yolo + efficientnetb4)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
