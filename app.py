from flask import Flask, request, render_template_string, url_for
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np
import os
from joblib import load

# Configuration
MODEL_PATH = "all_embeddings.json"  # Precomputed embeddings
MODEL_OUTPUT_FILE = "svm_model.joblib"  # Trained SVM model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load DINOv2 model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_vits14.to(device)

# Load SVM classifier
clf = load(MODEL_OUTPUT_FILE)

# Image transformation
def transform_image(img_path):
    img = Image.open(img_path)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5])
    ])
    return transform(img)[:3].unsqueeze(0)

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'static/uploads' directory exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# HTML templates
home_page_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Classification App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        header {
            background-color: #6200ea;
            color: white;
            padding: 20px;
            text-align: center;
            width: 100%;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        footer {
            background-color: #6200ea;
            color: white;
            padding: 10px;
            text-align: center;
            width: 100%;
            position: fixed;
            bottom: 0;
        }
        .content {
            text-align: center;
            margin: 20px;
        }
        input[type="file"] {
            padding: 10px;
            margin: 20px;
        }
        button {
            padding: 10px 20px;
            background-color: #6200ea;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #3700b3;
        }
    </style>
</head>
<body>
    <header>
        <h1>Image Classification App</h1>
    </header>
    <div class="content">
        <h2>Upload an Image for Classification</h2>
        <form action="/classify" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <br>
            <button type="submit">Classify Image</button>
        </form>
    </div>
    <footer>
        <p>&copy; 2025 Image Classification App</p>
    </footer>
</body>
</html>
"""

result_page_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Classification Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            text-align: center;
            padding: 20px;
        }
        .result {
            background-color: #6200ea;
            color: white;
            padding: 20px;
            border-radius: 10px;
            display: inline-block;
            margin-top: 20px;
        }
        img {
            margin-top: 20px;
            max-width: 80%;
            height: auto;
            border: 2px solid #6200ea;
            border-radius: 10px;
        }
        a {
            text-decoration: none;
            color: #6200ea;
        }
    </style>
</head>
<body>
    <header>
        <h1>Image Classification Result</h1>
    </header>
    <div class="result">
        <h2>Predicted Class: {{ prediction }}</h2>
        <img src="{{ image_url }}" alt="Uploaded Image">
    </div>
    <br>
    <a href="/">Classify Another Image</a>
</body>
</html>
"""

@app.route("/")
def home_page():
    return home_page_template

@app.route("/classify", methods=["POST"])
def classify_image():
    try:
        # Save uploaded file
        file = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Transform image
        img_tensor = transform_image(file_path).to(device)

        # Compute embedding
        with torch.no_grad():
            embedding = dinov2_vits14(img_tensor)
            embedding_np = np.array(embedding[0].cpu()).reshape(1, -1)

        # Predict class
        prediction = clf.predict(embedding_np)[0]

        # Generate URL for the uploaded image
        image_url = url_for('static', filename=f'uploads/{file.filename}')

        return render_template_string(result_page_template, prediction=prediction, image_url=image_url)

    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
