import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet Model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjust classes based on your dataset

# Load Trained Weights
model.load_state_dict(torch.load("resnet_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction Function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "PNEUMONIA" if predicted.item() == 1 else "NORMAL"
