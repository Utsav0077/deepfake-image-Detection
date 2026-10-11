from pathlib import Path
import cv2

# Project directories
ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "processed" / "frames"
OUTPUT_DIR = ROOT / "data" / "processed" / "video_faces_v2"

PROTOTXT = ROOT / "models" / "deploy.prototxt"
WEIGHTS = (
    ROOT / "models"
    / "res10_300x300_ssd_iter_140000.caffemodel"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verify model files
for path in [PROTOTXT, WEIGHTS]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")

# Load pretrained face detector
net = cv2.dnn.readNetFromCaffe(
    str(PROTOTXT),
    str(WEIGHTS)
)

CONFIDENCE_THRESHOLD = 0.80

total_frames = 0
total_faces = 0

frame_paths = sorted(INPUT_DIR.glob("*.jpg"))

if not frame_paths:
    raise FileNotFoundError(
        f"No JPG frames found in {INPUT_DIR}"
    )

for image_path in frame_paths:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipping unreadable image: {image_path.name}")
        continue

    height, width = image.shape[:2]

    # Convert frame to DNN input
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()

    total_frames += 1
    saved_in_frame = 0

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        box = detections[0, 0, i, 3:7]

        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        # Add 20% margin around face
        box_w = x2 - x1
        box_h = y2 - y1

        margin_x = int(box_w * 0.20)
        margin_y = int(box_h * 0.20)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            continue

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face = cv2.resize(face, (224, 224))

        output_path = (
            OUTPUT_DIR /
            f"{image_path.stem}_face_{saved_in_frame + 1}.jpg"
        )

        if not cv2.imwrite(str(output_path), face):
            raise RuntimeError(
                f"Failed to save: {output_path}"
            )

        saved_in_frame += 1
        total_faces += 1

    print(
        f"{image_path.name}: "
        f"{saved_in_frame} faces saved"
    )

print("\n=== Improved Face Detection Report ===")
print(f"Frames processed: {total_frames}")
print(f"Face crops saved: {total_faces}")
print("DNN face extraction completed!")