import cv2
import os
import json
import numpy as np

def main():
    base_dir = "faces"
    if not os.path.isdir(base_dir):
        raise RuntimeError("No 'faces/' directory found. Run collect.py first.")

    # Build label map
    people = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not people:
        raise RuntimeError("No people folders found in faces/. Add some with collect.py.")

    name_to_id = {name: i for i, name in enumerate(people)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    X = []
    y = []

    for name in people:
        person_dir = os.path.join(base_dir, name)
        for fn in os.listdir(person_dir):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(person_dir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # ensure consistent shape
            img = cv2.resize(img, (200, 200))
            X.append(img)
            y.append(name_to_id[name])

    if len(X) < 2:
        raise RuntimeError("Not enough training images. Collect more.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    recognizer.train(X, np.array(y))

    recognizer.write("trainer.yml")
    with open("labels.json", "w") as f:
        json.dump(id_to_name, f)

    print("[DONE] Trained model saved to trainer.yml")
    print("[DONE] Labels saved to labels.json")
    print("People:", people)

if __name__ == "__main__":
    main()
