import cv2
import json
import time
from collections import deque

def main():
    with open("labels.json", "r") as f:
        id_to_name = {int(k): v for k, v in json.load(f).items()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # ===== Tunables =====
    UNKNOWN_THRESHOLD = 70.0
    STABLE_SEC = 0.3
    MIN_STABLE_FRAMES = 6
    HISTORY_SEC = 1.5

    # ===== State =====
    last_announced = None
    candidate_name = None
    candidate_start = None
    history = deque()  # (t, name)

    print("[INFO] Running (debounced, no reset). Press Ctrl+C to stop.")

    def stable_enough(now: float, name: str) -> bool:
        while history and (now - history[0][0]) > HISTORY_SEC:
            history.popleft()

        total = len(history)
        if total < MIN_STABLE_FRAMES:
            return False

        hits = sum(1 for _, n in history if n == name)
        return hits / total >= 0.75  # majority vote

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) == 0:
                # No face: do nothing, don't clear candidate/history/last_announced.
                continue

            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            label_id, dist = recognizer.predict(face_roi)
            if dist <= UNKNOWN_THRESHOLD:
                name = id_to_name.get(label_id, "Unknown")
            else:
                name = "Unknown"

            history.append((now, name))

            if candidate_name != name:
                candidate_name = name
                candidate_start = now

            held_long_enough = (now - candidate_start) >= STABLE_SEC
            if held_long_enough and stable_enough(now, candidate_name):
                if candidate_name != last_announced:
                    print(f"{candidate_name} (dist={dist:.1f})")
                    last_announced = candidate_name

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()