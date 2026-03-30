from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secure API Key Handling
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Missing Gemini API key! Set GEMINI_API_KEY in a .env file.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Define health-related questions for pneumonia
questions = [
    {"question": "Do you have a persistent cough with mucus?", "severity": 20},
    {"question": "Are you experiencing shortness of breath?", "severity": 30},
    {"question": "Do you have a high fever with chills?", "severity": 25},
    {"question": "Are you feeling chest pain, especially when breathing or coughing?", "severity": 40},
    {"question": "Do you have fatigue or weakness?", "severity": 15},
    {"question": "Are you experiencing nausea, vomiting, or diarrhea?", "severity": 10},
    {"question": "Do you have a rapid heartbeat?", "severity": 20},
    {"question": "Are you experiencing confusion (especially in older adults)?", "severity": 35}
]

@app.route('/')
def home():
    return render_template('index.html', questions=questions)

@app.route('/api/quiz', methods=['POST'])
def process_quiz():
    data = request.get_json()
    answers = data.get('answers', [])

    if not answers or len(answers) != len(questions):
        return jsonify({"error": "Incomplete answers"}), 400

    # Calculate danger percentage based on "Yes" answers
    danger_percentage = 0
    max_danger = sum(q["severity"] for q in questions)  # Total possible severity
    selected_symptoms = []  # Store selected symptoms

    for i, answer in enumerate(answers):
        if answer == "1":  # "Yes" response
            danger_percentage += questions[i]["severity"]
            selected_symptoms.append(questions[i]["question"])

    # Normalize to percentage (0 - 100)
    danger_percentage = (danger_percentage / max_danger) * 100
    danger_percentage = round(danger_percentage, 2)

    # Determine danger level
    if danger_percentage <= 20:
        danger_level = "Low"
        message = "You have a low risk of serious health issues. Stay healthy!"
    elif 20 < danger_percentage <= 50:
        danger_level = "Moderate"
        message = "Your symptoms suggest a moderate risk. Monitor your health and rest well."
    elif 50 < danger_percentage <= 80:
        danger_level = "High"
        message = "You are at high risk. Consider consulting a doctor soon."
    else:
        danger_level = "Critical"
        message = "Seek medical attention immediately! Your symptoms are serious."

    # Generate AI-based precautions using Gemini API
    if selected_symptoms:
        prompt = (
            f"The patient reported the following symptoms: {', '.join(selected_symptoms)}.\n"
            f"Based on these symptoms, provide recommended precautions, home remedies, and steps to manage their condition.\n"
            f"Format the response as a short, easy-to-understand list."
        )

        response = model.generate_content(prompt)
        precautions = response.text.strip() if response and response.text else "No specific precautions available."
    else:
        precautions = "You have no major symptoms, but it's always good to stay hydrated and maintain a healthy lifestyle."

    return jsonify({
        "danger_percentage": danger_percentage,
        "danger_level": danger_level,
        "detailed_response": message,
        "precautions": precautions
    })

if __name__ == '__main__':
    app.run(debug=True)
