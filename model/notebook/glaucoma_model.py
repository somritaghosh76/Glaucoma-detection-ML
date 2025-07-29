# %%
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# Define model class
class GlaucomaResNet(nn.Module):
    def __init__(self):
        super(GlaucomaResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.fc = nn.Linear(self.backbone.fc.in_features, 2)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


# Preprocessing pipeline
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# Prediction logic
def predict_image_from_path(image_path, image_name, model_path, model_type, cdr_csv_path=None):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "resnet":
            model = GlaucomaResNet()
        else:
            return {"error": "Model type not supported yet."}

        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval().to(device)

        image = Image.open(image_path).convert("RGB")
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)

        result = {
            "predicted_class": "glaucoma" if predicted.item() == 1 else "normal",
            "confidence": confidence.item(),
            "normal_probability": probs[0][0].item(),
            "glaucoma_probability": probs[0][1].item()
        }

        # Include CDR if available  
        if cdr_csv_path:
            df = pd.read_csv(cdr_csv_path)
            match = df[df['Image'] == image_name]
            if not match.empty:
                result['cdr_value'] = match.iloc[0]['CDR']

        return result

    except Exception as e:
        return {"error": str(e)}


# Streamlit-safe visualization
def visualize_prediction(result, image_path):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image)
    ax.set_title(f"Prediction: {result['predicted_class'].capitalize()} ({result['confidence']:.2f})")
    ax.axis("off") 
    st.pyplot(fig)
