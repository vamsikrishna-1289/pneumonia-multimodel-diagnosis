from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

app = Flask(__name__)

# Load the trained DenseNet121 model for pneumonia detection
pneumonia_model = models.densenet121(weights=None)

pneumonia_model.classifier = torch.nn.Linear(pneumonia_model.classifier.in_features, 2)
pneumonia_model.load_state_dict(torch.load('pneumonia_densenet.pth'))
pneumonia_model.eval()

# Define the transform for pneumonia detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to handle chatbot responses
def ask_pneumonia_chatbot(question):
    model = genai.GenerativeModel("gemini-pro")

    # Ensuring chatbot answers only pneumonia-related queries
    prompt = f"""
    You are an AI chatbot specialized in pneumonia and lung health.
    If the question is unrelated, politely refuse to answer.

    User's Question: {question}
    """

    response = model.generate_content(prompt)
    return response.text

# API endpoint for chatbot
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = ask_pneumonia_chatbot(question)
    return jsonify({"response": answer})

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = pneumonia_model(image)
        _, predicted = torch.max(output, 1)
        result = 'PNEUMONIA' if predicted.item() == 1 else 'NORMAL'

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
