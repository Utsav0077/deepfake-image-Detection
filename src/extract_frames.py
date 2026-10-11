from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_DIR / "data" / "raw" / "sample.mp4"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video = cv2.VideoCapture(str(VIDEO_PATH))

if not video.isOpened():
    raise FileNotFoundError(
        f"Cannot open video: {VIDEO_PATH}"
    )

fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    video.release()
    raise RuntimeError("Could not determine video FPS.")

# Extract approximately one frame per second
frame_interval = max(1, round(fps))

frame_count = 0
saved_count = 0

while True:
    success, frame = video.read()

    if not success:
        break

    if frame_count % frame_interval == 0:
        output_path = (
            OUTPUT_DIR / f"frame_{saved_count:04d}.jpg"
        )

        if not cv2.imwrite(str(output_path), frame):
            video.release()
            raise RuntimeError(
                f"Failed to save {output_path}"
            )

        saved_count += 1

    frame_count += 1

video.release()

print(f"Video FPS: {fps:.2f}")
print(f"Frames extracted: {saved_count}")
print("Frame extraction completed!")