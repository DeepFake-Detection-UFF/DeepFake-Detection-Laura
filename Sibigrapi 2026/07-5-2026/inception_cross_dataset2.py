import os
import random
import numpy as np
from PIL import Image
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------
# CONFIG
# ---------------------------
FFPP_DIR = r"C:\Users\laura\OneDrive\Documentos\Dataset\FaceForensics++_faces"
DFDC_DIR = r"C:\Users\laura\OneDrive\Documentos\Dataset\dfdc_faces"

RESULTS_FILE = "results_cross_dataset_clean.csv"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CLASSES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "original"]

random.seed(42)

# ---------------------------
# DATASET DFDC (TREINO)
# ---------------------------
class DFDCDataset(Dataset):
    def __init__(self, root, split, transform):
        self.samples = []
        self.transform = transform

        self.real_count = 0
        self.fake_count = 0

        for label in ["REAL", "FAKE"]:
            path = os.path.join(root, split, label)
            if not os.path.exists(path):
                continue

            for img in os.listdir(path):
                if img.endswith(".png"):
                    label_bin = 0 if label == "REAL" else 1
                    self.samples.append((os.path.join(path, img), label_bin))

                    if label_bin == 0:
                        self.real_count += 1
                    else:
                        self.fake_count += 1

        print(f"[DFDC] {split}: {len(self.samples)} samples")
        print(f"REAL: {self.real_count} | FAKE: {self.fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# ---------------------------
# DATASET FF++ (TESTE)
# ---------------------------
class FFPPBinaryDataset(Dataset):
    def __init__(self, root, split, transform):
        self.samples = []
        self.transform = transform

        for cls in ALL_CLASSES:
            path = os.path.join(root, cls, split)
            if not os.path.exists(path):
                continue

            for img in os.listdir(path):
                if img.endswith(".png"):
                    label = 0 if cls == "original" else 1
                    self.samples.append((os.path.join(path, img), label))

        print(f"[FF++] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ---------------------------
# TRAIN
# ---------------------------
def train(model, loader, real_count, fake_count):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 🔥 peso baseado no desbalanceamento REAL vs FAKE
    pos_weight_value = real_count / fake_count
    pos_weight = torch.tensor([pos_weight_value]).to(DEVICE)

    print(f"Using pos_weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"[TRAIN] Epoch {epoch+1} Batch {i} Loss {loss.item():.4f}")

        print(f"[EPOCH {epoch+1}] Avg Loss: {total_loss / len(loader):.4f}")

    return model

# ---------------------------
# EVALUATE
# ---------------------------
def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.sigmoid(outputs)

            for i in range(len(probs)):
                y_true.append(labels[i].item())
                y_scores.append(probs[i].item())

                pred = 1 if probs[i] > 0.5 else 0
                y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)

    return acc, f1, auc

# ---------------------------
# MAIN
# ---------------------------
print("\n" + "="*60)
print("CROSS DATASET: DFDC → FF++")
print("="*60)

train_dataset = DFDCDataset(DFDC_DIR, "train", transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = FFPPBinaryDataset(FFPP_DIR, "test", transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

print("🚀 Training...")
model = train(model, train_loader, train_dataset.real_count, train_dataset.fake_count)

print("🧪 Testing on FF++...")
acc, f1, auc = evaluate(model, test_loader)

print("\nRESULTS:")
print(f"Accuracy: {acc:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# salva no MESMO CSV
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["DFDC_to_FF++", acc, f1, auc])