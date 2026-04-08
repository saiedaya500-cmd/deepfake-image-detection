import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DATA_DIR = "dataset"
BATCH_SIZE = 8
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.01
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_transform)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)

print("Classes:", train_ds.class_to_idx)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels_float = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels_float)
        probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()

        total_loss += loss.item() * images.size(0)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_labels, preds)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, auc, cm

best_val_auc = -1.0

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc, val_auc, val_cm = evaluate(model, val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")
        print("Meilleur modèle sauvegardé -> best_model.pth")

print("\n=== Test final ===")
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
test_loss, test_acc, test_auc, test_cm = evaluate(model, test_loader)

print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
print("Test Confusion Matrix:")
print(test_cm)