from pathlib import Path
import zipfile
import random

ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = Path.home() / "Downloads" / "archive.zip"
OUTPUT_DIR = ROOT / "data"

SPLITS = {
    "train": ("train", 1000),
    "val": ("valid", 200),
    "test": ("test", 200)
}

CLASSES = ["real", "fake"]
rng = random.Random(42)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    names = archive.namelist()

    for output_split, (source_split, count) in SPLITS.items():
        for label in CLASSES:

            prefix = (
                "real_vs_fake/real-vs-fake/"
                f"{source_split}/{label}/"
            )

            images = [
                name for name in names
                if name.startswith(prefix)
                and name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            print(
                f"{source_split}/{label}: "
                f"{len(images)} images available",
                flush=True
            )

            if len(images) < count:
                raise ValueError(
                    f"Not enough images in {prefix}"
                )

            selected = rng.sample(images, count)

            destination = OUTPUT_DIR / output_split / label
            destination.mkdir(parents=True, exist_ok=True)

            for index, name in enumerate(selected, start=1):
                output_path = destination / Path(name).name

                with archive.open(name) as source:
                    with open(output_path, "wb") as target:
                        import shutil
                        shutil.copyfileobj(source, target)

                if index % 100 == 0:
                    print(
                        f"{output_split}/{label}: "
                        f"{index}/{count}",
                        flush=True
                    )

print("\nDataset extraction completed!")