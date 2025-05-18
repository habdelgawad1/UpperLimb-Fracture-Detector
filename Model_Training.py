import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Set Up Paths
# -----------------------------
base_path = r"C:\Users\nerme\Documents\GitHub\Bone-Diagnosis"
dataset_path = os.path.join(base_path, "MURA-v1.1", "MURA-v1.1", "train")
wrist_train_path = os.path.join(dataset_path, "XR_HUMERUS")

if not os.path.exists(wrist_train_path):
    raise FileNotFoundError(f"XR_HAND folder not found: {wrist_train_path}")

print("✅ Using hand X-rays from:", wrist_train_path)

# -----------------------------
# 2. Define Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# -----------------------------
# 3. Custom Dataset Class to Skip Invalid Files
# -----------------------------
class SafeImageFolder:
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for patient_id in os.listdir(root):
            patient_dir = os.path.join(root, patient_id)
            if not os.path.isdir(patient_dir):
                continue
            for study in os.listdir(patient_dir):
                study_dir = os.path.join(patient_dir, study)
                if not os.path.isdir(study_dir):
                    continue
                for image_file in os.listdir(study_dir):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not image_file.startswith("._"):
                        label = 0 if 'negative' in study_dir else 1
                        self.samples.append((os.path.join(study_dir, image_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except UnidentifiedImageError:
            print(f"⚠️ Skipping corrupted file: {path}")
            return self.__getitem__((idx + 1) % len(self.samples))

# -----------------------------
# 4. Load Wrist Images Only
# -----------------------------
full_dataset = SafeImageFolder(root=wrist_train_path, transform=transform)

print(f"📁 Total valid hand images found: {len(full_dataset)}")

if len(full_dataset) == 0:
    raise ValueError("❌ No valid images found — check folder or re-download dataset")

# Limit to 5,000 samples for fast training
sample_size = min(5000, len(full_dataset))
subset_indices = torch.randperm(len(full_dataset))[:sample_size]
subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

train_size = int(0.8 * sample_size)
val_size = sample_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(subset_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# 5. Define Model (ResNet18)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# 6. Training Loop
# -----------------------------
def train_model(model, num_epochs=5):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_hand_fracture_model.pth')
            print("Saved Best Model!")

    return model

# -----------------------------
# 7. Start Training
# -----------------------------
print("🚀 Starting training on hand X-ray images...")
model = train_model(model, num_epochs=5)

# Save final model
torch.save(model.state_dict(), 'humerus_fracture_model_final.pth')
print("✅ Final model saved as humerus_fracture_model_final.pth")