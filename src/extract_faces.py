from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "data" / "processed" / "frames"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "video_faces"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

detector = cv2.CascadeClassifier(cascade_path)

if detector.empty():
    raise RuntimeError("Face detector failed to load.")

total_faces = 0

for image_path in sorted(INPUT_DIR.glob("*.jpg")):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping: {image_path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    print(f"{image_path.name}: {len(faces)} faces")

    for index, (x, y, w, h) in enumerate(faces):
        face = image[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))

        output_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_face_{index+1}.jpg"
        )

        if not cv2.imwrite(str(output_path), face):
            raise RuntimeError("Failed to save face crop")

        total_faces += 1

print(f"Total face crops saved: {total_faces}")
print("Video face extraction completed!")