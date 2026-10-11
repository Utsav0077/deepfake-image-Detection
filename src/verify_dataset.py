from pathlib import Path
from PIL import Image
import hashlib

ROOT = Path(__file__).resolve().parent.parent / "data"

splits = ["train", "val", "test"]
classes = ["real", "fake"]

seen = {}
duplicates = []
invalid = []
total = 0

for split in splits:
    for label in classes:
        folder = ROOT / split / label
        files = sorted(folder.glob("*.jpg"))

        print(f"{split}/{label}: {len(files)} images")

        for path in files:
            total += 1

            try:
                with Image.open(path) as img:
                    img.verify()

                digest = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()

                if digest in seen:
                    previous = seen[digest]
                    if previous[0] != split:
                        duplicates.append(
                            (str(path), previous[1])
                        )
                else:
                    seen[digest] = (split, str(path))

            except Exception as error:
                invalid.append((str(path), str(error)))

print("\n=== Dataset Verification ===")
print("Total images:", total)
print("Unreadable images:", len(invalid))
print("Cross-split exact duplicates:", len(duplicates))

for item in duplicates[:10]:
    print("Duplicate:", item)

for item in invalid[:10]:
    print("Invalid:", item)

if total == 2800 and not invalid and not duplicates:
    print("Basic dataset verification PASSED!")
else:
    print("Dataset requires review before training.")