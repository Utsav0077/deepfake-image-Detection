from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import timm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# =====================================
# 1. Page configuration
# =====================================

st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🛡️",
    layout="wide"
)

# =====================================
# 2. Project configuration
# =====================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================
# 3. Custom CSS
# =====================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f8fafc;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-real {
    color: #22c55e;
    font-size: 30px;
    font-weight: bold;
}

.result-fake {
    color: #fb7185;
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 4. Image preprocessing
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# 5. Load trained model
# =====================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    class_to_idx = checkpoint["class_to_idx"]

    if class_to_idx != {"fake": 0, "real": 1}:
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

    model = model.to(DEVICE)
    model.eval()

    return model, class_to_idx

# =====================================
# 6. Prediction function
# =====================================

def predict(image, model, class_to_idx):

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        output = model(tensor)
        probabilities = torch.softmax(
            output, dim=1
        )[0]

    fake_score = probabilities[
        class_to_idx["fake"]
    ].item()

    real_score = probabilities[
        class_to_idx["real"]
    ].item()

    prediction = (
        "FAKE"
        if fake_score > real_score
        else "REAL"
    )

    return prediction, fake_score, real_score

# =====================================
# 7. Main application
# =====================================

def main():

    st.markdown(
        '<div class="main-title">'
        '🛡️ DEEPFAKE AI DETECTOR'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Real vs Synthetic Face Classification '
        'Using EfficientNet-B0'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    with st.sidebar:
        st.header("⚙️ Model Information")

        st.write("Architecture: EfficientNet-B0")
        st.write("Framework: PyTorch")
        st.write(f"Device: {DEVICE}")
        st.write("Dataset: 140k Real and Fake Faces")
        st.write("Training subset: 2,000 images")
        st.write("Test accuracy: 72.75%")

        st.divider()

        st.info(
            "This is a research prototype. "
            "Predictions are not forensic proof."
        )

    try:
        model, class_to_idx = load_model()

    except Exception as error:
        st.error(f"Failed to load model: {error}")
        st.stop()

    st.success("AI Model Loaded Successfully!")

    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image to analyze",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        try:
            # Limit image input size
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("Maximum image size is 10 MB.")
                st.stop()

            image_bytes = uploaded_file.getvalue()

            with Image.open(BytesIO(image_bytes)) as img:
                if img.width * img.height > 25_000_000:
                    st.error("Image dimensions are too large.")
                    st.stop()

                image = img.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):
            st.error("Invalid or unsupported image.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True
            )

        with col2:
            st.subheader("AI Analysis")

            if st.button(
                "🔍 Analyze Image",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Analyzing image..."):
                    prediction, fake, real = predict(
                        image,
                        model,
                        class_to_idx
                    )

                if prediction == "REAL":
                    st.markdown(
                        '<div class="result-real">'
                        '✅ REAL PREDICTED'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-fake">'
                        '⚠️ FAKE PREDICTED'
                        '</div>',
                        unsafe_allow_html=True
                    )

                st.write("### Prediction Scores")

                st.write(f"Real score: {real * 100:.2f}%")
                st.progress(real)

                st.write(f"Fake score: {fake * 100:.2f}%")
                st.progress(fake)

                st.caption(
                    "Scores are uncalibrated model outputs, "
                    "not verified probabilities."
                )

                st.warning(
                    "This model was trained on real and "
                    "GAN-generated face images. It may not "
                    "generalize to other image manipulations "
                    "or face-swapped videos."
                )

    st.divider()

    st.caption(
        "Deepfake Detection Research Project | "
        "EfficientNet-B0 + PyTorch + Streamlit"
    )


if __name__ == "__main__":
    main()