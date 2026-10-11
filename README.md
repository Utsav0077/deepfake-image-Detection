AI-Based Deepfake Image Detection System

An AI-powered image classification system developed using EfficientNet-B0, PyTorch, and Streamlit to distinguish real facial images from GAN-generated synthetic faces.

Project Overview

This project uses transfer learning to classify facial images as REAL or FAKE. It includes a trained deep learning model, an image prediction pipeline, and an interactive Streamlit web application.

Technologies Used

- Python
- PyTorch and Torchvision
- EfficientNet-B0 (timm)
- Streamlit
- OpenCV
- Pillow
- Scikit-learn

Dataset

140k Real and Fake Faces dataset from Kaggle.

A balanced subset of 2,800 images was used:

Split| Real| Fake
Training| 1,000| 1,000
Validation| 200| 200
Testing| 200| 200

Dataset: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Dataset files are not included in this repository.

Model Performance

The EfficientNet-B0 baseline model achieved the following results on the selected 400-image test set:

Metric| Result
Test Accuracy| 72.75%
Macro Precision| 0.73
Macro Recall| 0.73
Macro F1-score| 0.73

Confusion matrix:

                Predicted
                Fake  Real
Actual Fake      149    51
Actual Real       58   142

Features

- Upload JPG, JPEG, or PNG images.
- Predict REAL or FAKE using a trained classifier.
- Display image previews and model scores.
- GPU acceleration when CUDA is available.
- Streamlit-based user interface.

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/Deepfake-Image-Detection.git
cd Deepfake-Image-Detection

Create and activate a Python virtual environment:

python -m venv venv
.\venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

Model Checkpoint

The trained model checkpoint is not included in the Git repository.

To reproduce training, obtain the dataset from Kaggle, prepare the required "data/train" and "data/val" class folders, and run:

python src/train_v2.py

This creates "models/best_model.pth". Reproduced training results may vary.

Run the Web Application

Once the trained checkpoint is available:

python -m streamlit run app/app.py

Open the local URL displayed in the terminal, normally "http://localhost:8501".

Limitations

This is a research prototype trained on real and GAN-generated facial images. It is not validated for all types of deepfakes, face swaps, or image manipulations. The displayed model scores are not calibrated probabilities and should not be considered forensic evidence.

Future Scope

- Larger and more diverse training datasets.
- Improved generalization across synthetic image generators.
- Additional model evaluation and calibration.
- Improved user interface.

Project Status

Working image classification prototype with Streamlit interface and evaluated EfficientNet-B0 baseline model.