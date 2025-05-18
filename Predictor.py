from PIL import Image
import torchvision.transforms as transforms
import torch

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def predict_fracture(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        prediction = "Fracture" if prob > 0.5 else "Normal"

    return {
        "injury_type": prediction,
        "confidence": prob
    }