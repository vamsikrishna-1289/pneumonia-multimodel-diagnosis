from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from datetime import datetime
import os
from gradcam import generate_cam

app = Flask(__name__)

# Load the trained model
model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load('pneumonia_densenet.pth', map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img_name = file.filename
    image = Image.open(file.stream).convert('RGB')
    transformed = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(transformed)
        _, predicted = torch.max(output, 1)
        result = 'PNEUMONIA' if predicted.item() == 1 else 'NORMAL'

    # Log the prediction
    with open("log_predictions.txt", "a") as log:
        log.write(f"{datetime.now()} - File: [Hidden for privacy] - Prediction: {result}\n")

    # Generate heatmap only for pneumonia
    cam_path = None
    if result == 'PNEUMONIA':
        cam_path = generate_cam(model, transformed, predicted.item(), img_name)

    # Build response
    response = {"result": result}
    if cam_path:
        response["cam_image"] = "/" + cam_path

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
