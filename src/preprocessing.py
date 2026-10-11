from pathlib import Path
import cv2

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_IMAGE = PROJECT_DIR / "data" / "test.jpg"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load OpenCV's pretrained face detector
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Face detector could not be loaded.")

# Read image
image = cv2.imread(str(INPUT_IMAGE))

if image is None:
    raise FileNotFoundError(
        f"Image not found or unreadable: {INPUT_IMAGE}"
    )

# Convert to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40, 40)
)

print(f"Faces detected: {len(faces)}")

if len(faces) == 0:
    print("No faces detected.")
else:
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Crop face
        face = image[y:y+h, x:x+w]

        # Resize to model input size
        face = cv2.resize(
            face,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        # Save processed image
        output_path = OUTPUT_DIR / f"face_{i+1}.jpg"

        success = cv2.imwrite(str(output_path), face)

        if not success:
            raise RuntimeError(
                f"Failed to save {output_path}"
            )

        print(f"Saved: {output_path}")

print("Preprocessing finished!")