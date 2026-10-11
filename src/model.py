import torch
import timm

# Select GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# Load pretrained EfficientNet-B0
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)

# Move model to GPU or CPU
model = model.to(device)

# Set evaluation mode for testing
model.eval()

# Create a dummy image batch
dummy_input = torch.randn(
    1, 3, 224, 224,
    device=device
)

# Test model forward pass
with torch.no_grad():
    output = model(dummy_input)

print("Model created successfully!")
print("Output shape:", output.shape)
print("Model ready for training!")