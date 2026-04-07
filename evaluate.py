print("EVALUATE.PY STARTED ✅")
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

DATA_DIR = "dataset"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 32
IMG_SIZE = 224

# Transform uniquement (pas d'augmentation)
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Dataset test
test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes:", test_ds.class_to_idx)
print("Test size:", len(test_ds))

# Modèle = même architecture que train.py
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model = model.to(DEVICE)

# Charger le meilleur modèle
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

criterion = nn.BCEWithLogitsLoss()

@torch.no_grad()
def evaluate(model, loader):
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels_f = labels.float().unsqueeze(1).to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels_f)

        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        total_loss += loss.item() * images.size(0)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, preds)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, auc, cm

test_loss, test_acc, test_auc, test_cm = evaluate(model, test_loader)

print("\n=== EVALUATION TEST (best_model.pth) ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Acc:  {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"Test AUC:  {test_auc:.4f}")
print("Confusion Matrix:")
print(test_cm)