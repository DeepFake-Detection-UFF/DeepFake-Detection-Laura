import os
import random
import numpy as np
from PIL import Image
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = r"C:\Users\laura\OneDrive\Documentos\Dataset\FaceForensics++_faces"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
THRESHOLD = 0.9

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CLASSES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "original"]

RESULTS_FILE = "results_open_set.csv"

random.seed(42)

# ---------------------------
# CSV HEADER
# ---------------------------
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "open_set_class",
            "accuracy",
            "f1",
            "auc",
            "unknown_detected",
            "unknown_ratio"
        ])

# ---------------------------
# DATASET
# ---------------------------
class FaceDataset(Dataset):
    def __init__(self, root_dir, split, classes, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx

        for cls in classes:
            cls_path = os.path.join(root_dir, cls, split)

            if not os.path.exists(cls_path):
                continue

            for img_name in os.listdir(cls_path):
                if img_name.endswith(".png"):
                    self.samples.append((
                        os.path.join(cls_path, img_name),
                        self.class_to_idx[cls]
                    ))

        print(f"Loaded {len(self.samples)} samples for split={split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ---------------------------
# TRANSFORMS
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# TRAIN
# ---------------------------
def train_model(model, train_loader, val_loader, model_path):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)

            if isinstance(outputs, tuple):
                outputs, aux_outputs = outputs
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

    return model

# ---------------------------
# OPEN-SET EVAL
# ---------------------------
def evaluate_open_set(model, loader, threshold, open_set_class, global_mapping, local_to_global):

    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    open_set_idx = global_mapping[open_set_class]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = F.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            for i in range(len(preds)):

                # Ground truth
                if labels[i].item() == open_set_idx:
                    y_true.append(-1)
                else:
                    y_true.append(labels[i].item())

                # Prediction
                if max_probs[i] < threshold:
                    y_pred.append(-1)
                else:
                    pred_local = preds[i].item()
                    pred_global = local_to_global[pred_local]
                    y_pred.append(pred_global)

                y_scores.append(max_probs[i].item())

    return np.array(y_true), np.array(y_pred), np.array(y_scores)

# ---------------------------
# MAIN
# ---------------------------
for OPEN_SET_CLASS in ["NeuralTextures"]: # "Face2Face", "FaceSwap", "NeuralTextures", "Deepfakes"

    print("\n" + "="*60)
    print(f"OPEN SET CLASS: {OPEN_SET_CLASS}")
    print("="*60)

    TRAIN_CLASSES = [c for c in ALL_CLASSES if c != OPEN_SET_CLASS]

    train_class_to_idx = {cls: i for i, cls in enumerate(TRAIN_CLASSES)}
    global_class_to_idx = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    local_to_global = {i: global_class_to_idx[cls] for cls, i in train_class_to_idx.items()}

    model_path = f"best_model_{OPEN_SET_CLASS}.pth"

    # DATASETS
    train_dataset = FaceDataset(BASE_DIR, "train", TRAIN_CLASSES, train_class_to_idx, transform)
    val_dataset   = FaceDataset(BASE_DIR, "val", TRAIN_CLASSES, train_class_to_idx, transform)
    test_dataset  = FaceDataset(BASE_DIR, "test", ALL_CLASSES, global_class_to_idx, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # MODEL
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(TRAIN_CLASSES))
    model = model.to(DEVICE)

    # TRAIN
    model = train_model(model, train_loader, val_loader, model_path)

    # LOAD BEST MODEL
    model.load_state_dict(torch.load(model_path))

    # EVAL
    y_true, y_pred, y_scores = evaluate_open_set(
        model,
        test_loader,
        THRESHOLD,
        OPEN_SET_CLASS,
        global_class_to_idx,
        local_to_global
    )

    unknown_count = np.sum(y_pred == -1)
    total = len(y_pred)
    unknown_ratio = unknown_count / total if total > 0 else 0

    mask_known = y_true != -1

    if np.sum(mask_known) > 0:
        acc = accuracy_score(y_true[mask_known], y_pred[mask_known])
        f1  = f1_score(y_true[mask_known], y_pred[mask_known], average='macro')
    else:
        acc, f1 = 0.0, 0.0

    y_true_binary = (y_true != -1).astype(int)
    auc = roc_auc_score(y_true_binary, y_scores) if len(np.unique(y_true_binary)) > 1 else 0.0

    cm = confusion_matrix(y_true, y_pred)

    print("\nRESULTADOS:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Unknown detected: {unknown_count} ({unknown_ratio:.2%})")
    print("\nConfusion Matrix:")
    print(cm)

    # SAVE CSV
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            OPEN_SET_CLASS,
            round(acc, 4),
            round(f1, 4),
            round(auc, 4),
            int(unknown_count),
            round(unknown_ratio, 4)
        ])