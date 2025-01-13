import os
import json
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn import svm
from joblib import dump, load
import warnings
import cv2
import matplotlib.pyplot as plt

# Suppress specific warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# Avoid issues with truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load pre-trained DINOv2 model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_vits14.to(device)

# Define image transformation pipeline
transform_image = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to load and transform an image
def load_image(img_path: str) -> torch.Tensor:
    try:
        img = Image.open(img_path).convert("RGB")
        transformed_img = transform_image(img).unsqueeze(0).to(device)
        return transformed_img
    except (OSError, ValueError) as e:
        print(f"Warning: Skipping image {img_path} due to error: {e}")
        return None

# Compute embeddings for all images
def compute_embeddings(files: list) -> dict:
    all_embeddings = {}
    with torch.no_grad():
        for i, file in enumerate(tqdm(files, desc="Processing Images")):
            img_tensor = load_image(file)
            if img_tensor is None:  # Skip problematic images
                continue
            embedding = dinov2_vits14(img_tensor)
            all_embeddings[file] = embedding.cpu().numpy().reshape(-1).tolist()
    
    with open("all_embeddings.json", "w") as f:
        json.dump(all_embeddings, f)
    
    return all_embeddings

# Set dataset path
ROOT_DIR = os.path.join(os.getcwd(), "Tomato-Classification-3/train")

# Create labels dictionary
labels = {}
for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):  # Skip non-folder entries
        continue
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            full_path = os.path.abspath(os.path.join(folder_path, file))
            labels[full_path] = folder

files = list(labels.keys())

# Load or compute embeddings
if os.path.exists("all_embeddings.json"):
    with open("all_embeddings.json", "r") as f:
        embeddings = json.load(f)
else:
    embeddings = compute_embeddings(files)

# Filter files to ensure matching keys
valid_files = [file for file in files if file in embeddings]
if not valid_files:
    print("No valid files found in embeddings. Regenerating embeddings...")
    embeddings = compute_embeddings(files)
    valid_files = [file for file in files if file in embeddings]

X = np.array([embeddings[file] for file in valid_files])
y = [labels[file] for file in valid_files]

# Load or train SVM classifier
if os.path.exists("svm_model.joblib"):
    clf = load("svm_model.joblib")
else:
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)
    dump(clf, "svm_model.joblib")

# Predict for a new image
new_image_path = "demo.jpg"  # Change to a valid image path
if not os.path.exists(new_image_path):
    raise FileNotFoundError(f"New image file not found: {new_image_path}")

new_image = load_image(new_image_path)
if new_image is None:
    raise ValueError(f"Failed to load the new image: {new_image_path}")

with torch.no_grad():
    embedding = dinov2_vits14(new_image).cpu().numpy().reshape(1, -1)
    prediction = clf.predict(embedding)

print(f"Predicted class: {prediction[0]}")

# Display the image
img = cv2.imread(new_image_path)
if img is None:
    raise FileNotFoundError(f"Could not load the image: {new_image_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
