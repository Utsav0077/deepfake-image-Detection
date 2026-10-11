from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from dataset import create_dataloader

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 5
LEARNING_RATE = 0.0001
BATCH_SIZE = 4

def main():
    train_loader, train_classes = create_dataloader(
        "train", BATCH_SIZE
    )
    val_loader, val_classes = create_dataloader(
        "val", BATCH_SIZE
    )

    expected = {"fake": 0, "real": 1}

    if train_classes != expected or val_classes != expected:
        raise ValueError("Unexpected class labels!")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=2
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    print("Training on:", DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                correct += (
                    predictions == labels
                ).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = (
            correct / len(val_loader.dataset)
        ) * 100

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_classes,
                    "epoch": epoch + 1,
                },
                MODEL_DIR / "best_model.pth"
            )

            print("Best model saved!")

    print("Training completed!")


if __name__ == "__main__":
    main()