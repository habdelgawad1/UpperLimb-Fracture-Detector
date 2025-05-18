import torch
from torchvision import models
import torch.nn as nn
import os

def get_model_path(body_part):
    # Map body part to model path
    model_map = {
        "wrist": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\wrist.pth',
        "hand": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\hand.pth',
        "elbow": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\elbow.pth',
        "shoulder": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\shoulder.pth',
        "forearm": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\forearm.pth',
        "humerus": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\humerus.pth',
        "finger": r'C:\Users\nerme\Documents\GitHub\Bone-Diagnosis\Models\finger.pth'
    }

    if body_part.lower() not in model_map:
        raise ValueError(f"❌ Unknown body part: {body_part}. Choose from: {list(model_map.keys())}")

    model_path = model_map[body_part.lower()]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    return model_path

def load_trained_model(model_path):
    # Define model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"✅ Loaded model from {model_path}")
    return model