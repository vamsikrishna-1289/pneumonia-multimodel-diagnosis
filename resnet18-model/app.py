from flask import Flask, request, render_template
from model import predict_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        prediction = predict_image(file_path)
        return render_template("index.html", image=file.filename, result=prediction)

    return render_template("index.html", image=None, result=None)

if __name__ == "__main__":
    app.run(debug=True)
