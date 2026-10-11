from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def create_dataloader(split="train", batch_size=4):
    folder = DATA_DIR / split

    dataset = datasets.ImageFolder(
        root=folder,
        transform=transform
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No images found in {folder}. "
            "Add labeled dataset images before training."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0
    )

    return loader, dataset.class_to_idx