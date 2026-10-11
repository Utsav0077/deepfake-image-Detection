from pathlib import Path

# Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset folders
REAL_DIR = PROJECT_DIR / "data" / "real"
FAKE_DIR = PROJECT_DIR / "data" / "fake"

# Supported image formats
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(folder):
    return sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in EXTENSIONS
    )

real_count = count_images(REAL_DIR)
fake_count = count_images(FAKE_DIR)

print("=== Deepfake Dataset Report ===")
print(f"Real images: {real_count}")
print(f"Fake images: {fake_count}")
print(f"Total images: {real_count + fake_count}")

if real_count == 0 or fake_count == 0:
    print("Dataset is not ready for training yet.")
else:
    print("Both image classes contain files.")