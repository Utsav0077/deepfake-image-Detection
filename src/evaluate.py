from pathlib import Path

import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
MODEL_PATH = ROOT / "models" / "best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

assert (
    checkpoint["class_to_idx"]
    == test_dataset.class_to_idx
), "Class mapping mismatch!"

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

true_labels = []
predicted_labels = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(
            predictions.cpu().tolist()
        )

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

print("\n=== Model Evaluation Report ===")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1]
    )
)

print("\nEvaluation completed!")