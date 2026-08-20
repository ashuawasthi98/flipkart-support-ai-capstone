import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image, ImageDraw
import numpy as np
import shutil

# 1. Classes Definition
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)


# 2. Set up transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)), # resize for ResNet consumption
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert 1 channel grayscale to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Data
print("Internet connection detected. Downloading actual Fashion-MNIST dataset...")
from torchvision.datasets import FashionMNIST
train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Initialize ResNet-18 Backbone
print("Loading pretrained ResNet-18 weights...")
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone parameters
for param in backbone.parameters():
    param.requires_grad = False

# Replace final fc layer with Identity to output 512 features
backbone.fc = nn.Identity()

# Move backbone to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = backbone.to(device)
backbone.eval()

# 5. Feature Caching (Optimization step)
print("Caching backbone features to RAM to accelerate head training...")
def cache_features(loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            features = backbone(imgs)
            all_features.append(features.cpu())
            all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

train_features, train_labels = cache_features(train_loader)
test_features, test_labels = cache_features(test_loader)

print(f"Cached train features shape: {train_features.shape}")
print(f"Cached test features shape: {test_features.shape}")

# Create DataLoader for cached features
cached_train_ds = TensorDataset(train_features, train_labels)
cached_test_ds = TensorDataset(test_features, test_labels)

cached_train_loader = DataLoader(cached_train_ds, batch_size=128, shuffle=True)
cached_test_loader = DataLoader(cached_test_ds, batch_size=128, shuffle=False)

# 8. Define Custom Classification Head
class ClassifierHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

head = ClassifierHead().to(device)

# 9. Train Classifier Head
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=0.005)

print("\n--- Training Classification Head ---")
epochs = 15
for epoch in range(1, epochs + 1):
    head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for feats, lbls in cached_train_loader:
        feats, lbls = feats.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = head(feats)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * feats.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Validation
    head.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for feats, lbls in cached_test_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            outputs = head(feats)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == lbls).sum().item()
            val_total += lbls.size(0)
    val_acc = val_correct / val_total
    
    print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4%} | Val Acc: {val_acc:.4%}")

# 10. Define and Save Combined Model (Backbone + Head)
class CombinedModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone.cpu() # move back to CPU for serialization
        self.head = head.cpu()
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

combined_model = CombinedModel(backbone, head)
torch.save(combined_model.state_dict(), "models/product_classifier.pt")
print(f"\nSaved combined model weights to models/product_classifier.pt")

# 11. Compute Confusion Matrix & Classification Metrics
print("\n--- Model Evaluation ---")
head.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for feats, lbls in cached_test_loader:
        feats = feats.to(device)
        outputs = head(feats)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(lbls.numpy())

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(all_targets, all_preds)
print("\nConfusion Matrix:")
print(cm)

# 12. Export 5 Sample Images from the Test Split for Agent Tool usage
print("\nExporting 5 sample images representing different categories...")
# We pick 5 categories: Trouser (1), Sandal (5), Sneaker (7), Bag (8), Ankle boot (9)
export_labels = [1, 5, 7, 8, 9]
exported_count = 0

for i in range(len(test_dataset)):
    img_tensor, label = test_dataset[i]
    if label in export_labels:
        # Convert tensor back to standard grayscale PIL Image to save
        # Grayscale PIL images should be saved as 28x28 grayscale
        raw_image_tensor = test_dataset.data[i] # raw tensor before transformation
        pil_img = transforms.ToPILImage()(raw_image_tensor) # Convert the tensor to PIL Image
        class_name = CLASSES[label].lower().replace("/", "_").replace(" ", "_")
        img_filename = f"data/sample_images/{label:02d}_{class_name}.png"
        
        pil_img.save(img_filename)
        print(f"Exported: {img_filename} (True Label: {CLASSES[label]})")
        
        export_labels.remove(label)
        exported_count += 1
        if len(export_labels) == 0:
            break

print(f"\nSuccessfully exported {exported_count} sample .png files to data/sample_images/")
