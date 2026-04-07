import torch
from torchvision import transforms, models
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# load image
img_path = "test.jpg.png"   # j'ajoute ma photo ici 
image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# prediction
with torch.no_grad():
    output = model(image)
    prob = torch.sigmoid(output).item()

if prob >= 0.5:
    print(f"REAL (prob={prob:.2f})")
else:
    print(f"FAKE (prob={prob:.2f})")