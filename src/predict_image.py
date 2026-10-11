from pathlib import Path
import argparse

import torch
import timm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# -------------------------------
# 1. Project paths
# -------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pth"

# -------------------------------
# 2. Device
# -------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# -------------------------------
# 3. Image preprocessing
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# 4. Load trained model
# -------------------------------

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]

    if (
        set(class_to_idx.keys()) != {"fake", "real"}
        or set(class_to_idx.values()) != {0, 1}
    ):
        raise ValueError(
            f"Unexpected class mapping: {class_to_idx}"
        )

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

    print("Trained model loaded successfully!")
    print("Checkpoint epoch:", checkpoint["epoch"])

    return model, class_to_idx

# -------------------------------
# 5. Predict image
# -------------------------------

def predict_image(image_path, model, class_to_idx):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Cannot read image: {image_path}"
        ) from error

    input_tensor = input_tensor.to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    fake_score = probabilities[
        class_to_idx["fake"]
    ].item()

    real_score = probabilities[
        class_to_idx["real"]
    ].item()

    predicted_index = outputs.argmax(dim=1).item()

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    prediction = idx_to_class[predicted_index]

    print("\n========== AI PREDICTION ==========")
    print("Image:", image_path.name)
    print("Prediction:", prediction.upper())
    print(f"Fake score: {fake_score * 100:.2f}%")
    print(f"Real score: {real_score * 100:.2f}%")
    print("===================================")

    return prediction, fake_score, real_score

# -------------------------------
# 6. Main function
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Real vs Synthetic Face Classifier"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image"
    )

    args = parser.parse_args()

    model, class_to_idx = load_model()

    predict_image(
        args.image,
        model,
        class_to_idx
    )

if __name__ == "__main__":
    main()