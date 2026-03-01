import cv2
import os
import time
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Person name (label)")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--count", type=int, default=300, help="Images to capture")
    args = ap.parse_args()

    out_dir = os.path.join("faces", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Haar cascade (ships with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    saved = 0
    last_save = 0.0

    print(f"[INFO] Collecting {args.count} images for '{args.name}' into {out_dir}")
    print("[INFO] Press 'q' to quit early.")

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        # draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # save best face (largest) every ~0.15s
        now = time.time()
        if len(faces) > 0 and (now - last_save) > 0.15:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            face_roi = gray[y:y+h, x:x+w]

            # normalize size (LBPH is more stable with consistent size)
            face_roi = cv2.resize(face_roi, (200, 200))

            path = os.path.join(out_dir, f"{saved:04d}.png")
            cv2.imwrite(path, face_roi)
            saved += 1
            last_save = now

        cv2.putText(frame, f"Saved: {saved}/{args.count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("Collect Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} images for '{args.name}'")

if __name__ == "__main__":
    main()
