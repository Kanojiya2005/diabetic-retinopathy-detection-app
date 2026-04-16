import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import gdown 

# Class Labels
CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative (Blindness Risk)"
}

# Medical Suggestions
SUGGESTIONS = {
    0: "No signs of Diabetic Retinopathy detected. Maintain healthy blood sugar levels and schedule routine eye checkups annually.",
    
    1: "Mild Diabetic Retinopathy detected. Monitor blood sugar regularly and consult an ophthalmologist within 6 months.",
    
    2: "Moderate Diabetic Retinopathy detected. It is advised to visit an eye specialist soon for further evaluation and possible treatment.",
    
    3: "Severe Diabetic Retinopathy detected. Immediate consultation with a retina specialist is strongly recommended.",
    
    4: "Proliferative Diabetic Retinopathy detected. High risk of vision loss. Urgent medical attention and treatment required."
}

# Image Transform
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

# Load Model (Google Drive)
def load_model(model_path="model/dr_model.pth", device="cpu"):

    # Create model folder if not exists
    if not os.path.exists("model"):
        os.makedirs("model")

    # Download model if not present
    if not os.path.exists(model_path):
        print("Downloading model from Google Drive...")

        file_id = "1quklJfOJfMHRkuXNfxQw8zbhU_SbvX3H"

        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    # Load model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# Predict Function
def predict_image(image, model, device="cpu"):
    transform = get_transform()
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    idx = predicted.item()

    return {
        "class_name": CLASSES[idx],
        "suggestion": SUGGESTIONS[idx]
    }
