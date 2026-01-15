####################################
#🔹 STEP 0: Mount Google Drive
####################################

from google.colab import drive

# 1. SETUP & DRIVE MOUNTING
drive.mount('/content/drive')

# ============================================================
# Clinically Inspired Hierarchical Multi-Label CXR Classification
# COMPLETE + STABLE + PROGRESS BARS (Colab Pro GPU)
# ============================================================

import os, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
from collections import defaultdict

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.backends.cudnn.benchmark = True


# ============================================================
# PATHS
# ============================================================
train_csv = "/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/train.csv"
val_csv = "/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/valid.csv"
train_img_root = "/content/drive/MyDrive/Kaggel_direct_download"
val_img_root = "/content/drive/MyDrive/Kaggel_direct_download"

# ============================================================
# LABELS
# ============================================================
BASE_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]

HIERARCHY = {
    "Abnormal": ["Cardiac", "Opacity", "Other"],
    "Cardiac": ["Enlarged Cardiomediastinum", "Cardiomegaly"],
    "Opacity": ["Lung Opacity", "Lung Lesion", "Fluid Accumulation", "Missing Lung Tissue"],
    "Fluid Accumulation": ["Edema", "Consolidation", "Pneumonia", "Pleural Effusion"],
    "Missing Lung Tissue": ["Atelectasis", "Pneumothorax", "Pleural Other"],
    "Other": ["Fracture", "Support Devices"]
}

PARENT_LABELS = list(HIERARCHY.keys())
ALL_LABELS = BASE_LABELS + PARENT_LABELS
LABEL2IDX = {l: i for i, l in enumerate(ALL_LABELS)}

def get_balanced_indices(df, label_cols, max_per_class=600, seed=42):
    np.random.seed(seed)

    indices_per_class = defaultdict(list)

    for idx, row in df.iterrows():
        for lbl in label_cols:
            if row[lbl] == 1:
                indices_per_class[lbl].append(idx)

    selected = set()

    # Sample positives
    for lbl, idxs in indices_per_class.items():
        if len(idxs) > 0:
            sampled = np.random.choice(
                idxs,
                size=min(max_per_class, len(idxs)),
                replace=False
            )
            selected.update(sampled.tolist())

    # Optional: balance negatives
    neg_mask = (df[label_cols] == 1).sum(axis=1) == 0
    neg_indices = df[neg_mask].index.tolist()

    num_pos = len(selected)
    if len(neg_indices) > 0:
        neg_sampled = np.random.choice(
            neg_indices,
            size=min(num_pos, len(neg_indices)),
            replace=False
        )
        selected.update(neg_sampled.tolist())

    return sorted(list(selected))


# ============================================================
# DATASET
# ============================================================
class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_root, transform, indices=None):
        df = pd.read_csv(csv_path).dropna(subset=["Path"]).reset_index(drop=True)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        self.img_root = img_root
        self.transform = transform
        self.labels = self.df[BASE_LABELS].replace(-1, 0).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.df.iloc[idx]["Path"])
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            img = Image.new("RGB", (320, 320), (0, 0, 0))

        img = self.transform(img)
        label = torch.tensor(self.labels[idx])
        return img, label


# ============================================================
# TRANSFORMS
# ============================================================
transform_train = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# train_loader = DataLoader(
#     CheXpertDataset(train_csv, train_img_root, transform_train),
#     batch_size=16, shuffle=True, num_workers=4, pin_memory=True
# )

# val_loader = DataLoader(
#     CheXpertDataset(val_csv, val_img_root, transform_val),
#     batch_size=16, shuffle=False, num_workers=4, pin_memory=True
# )

# ---------------- TRAIN SET (BALANCED) ----------------
train_df = pd.read_csv(train_csv)

train_indices = get_balanced_indices(
    train_df,
    BASE_LABELS,
    max_per_class=1000
)

print(f"Balanced training samples: {len(train_indices)}")

train_dataset = CheXpertDataset(
    train_csv,
    train_img_root,
    transform_train,
    indices=train_indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# ---------------- VALIDATION SET (FULL) ----------------
val_dataset = CheXpertDataset(
    val_csv,
    val_img_root,
    transform_val
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# SAFE HIERARCHICAL LABEL EXPANSION (NO NaNs)
# ============================================================
def expand_hierarchical_labels(y):
    B = y.size(0)
    out = torch.zeros(B, len(ALL_LABELS), device=y.device)
    out[:, :len(BASE_LABELS)] = y

    for parent, children in HIERARCHY.items():
        valid_children = [LABEL2IDX[c] for c in children if c in LABEL2IDX]
        if len(valid_children) == 0:
            continue
        out[:, LABEL2IDX[parent]] = torch.max(out[:, valid_children], dim=1)[0]

    return out

# ============================================================
# MODEL (PAPER ARCHITECTURE)
# ============================================================
class DenseNet121_HBCE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.densenet121(weights=None)
        self.features = backbone.features
        self.conv = nn.Conv2d(1024, 512, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.bn(self.conv(x)))
        x = self.gap(x).flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc_out(x)


# ============================================================
# HBCE LOSS WITH WARM-UP (CRITICAL FIX)
# ============================================================
class HBCEWithLogitsLoss(nn.Module):
    def __init__(self, hierarchy, lambda_penalty=0.5, warmup_epochs=3):
        super().__init__()
        self.hierarchy = hierarchy
        self.lambda_penalty = lambda_penalty
        self.warmup_epochs = warmup_epochs
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, epoch):
        probs = torch.sigmoid(logits)
        loss = self.bce(logits, targets)

        scale = min(1.0, epoch / self.warmup_epochs)
        penalty = 0.0

        for parent, children in self.hierarchy.items():
            p = LABEL2IDX[parent]
            for c in children:
                if c not in LABEL2IDX:
                    continue
                c = LABEL2IDX[c]
                penalty += ((probs[:,p] < 0.5) & (probs[:,c] > 0.5)).float().mean()

        return loss + scale * self.lambda_penalty * penalty


# ============================================================
# METRICS
# ============================================================
def multilabel_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def hierarchy_violation_rate(probs):
    v, t = 0, 0
    for parent, children in HIERARCHY.items():
        p = LABEL2IDX[parent]
        for c in children:
            if c not in LABEL2IDX:
                continue
            c = LABEL2IDX[c]
            v += ((probs[:,p] < 0.5) & (probs[:,c] > 0.5)).sum()
            t += probs.shape[0]
    return (v / t).item()

# ============================================================
# INIT
# ============================================================
model = DenseNet121_HBCE(len(ALL_LABELS)).to(device)
criterion = HBCEWithLogitsLoss(HIERARCHY, lambda_penalty=0.5)
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

history = {
    "train_loss": [], "val_loss": [],
    "train_acc": [], "val_acc": [],
    "violation": []
}



# ============================================================
# TRAINING WITH PROGRESS BARS
# ============================================================
EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"\n{'='*60}\nEpoch {epoch+1}/{EPOCHS}\n{'='*60}")

    # ---------------- TRAIN ----------------
    model.train()
    tr_loss, tr_preds, tr_targs = 0, [], []

    for imgs, y in tqdm(train_loader, desc=f"Training {epoch+1}", dynamic_ncols=True):
        imgs = imgs.to(device, non_blocking=True)
        y = expand_hierarchical_labels(y.to(device, non_blocking=True))
        y = torch.nan_to_num(y, nan=0.0)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, y, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tr_loss += loss.item()
        tr_preds.append((torch.sigmoid(logits) > 0.5).int().cpu())
        tr_targs.append(y.cpu())

    tr_loss /= len(train_loader)
    tr_preds = torch.cat(tr_preds)
    tr_targs = torch.cat(tr_targs)
    tr_acc = multilabel_accuracy(tr_targs.numpy(), tr_preds.numpy())

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss, v_preds, v_probs, v_targs = 0, [], [], []

    with torch.no_grad():
        for imgs, y in tqdm(val_loader, desc=f"Validation {epoch+1}", dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            y = expand_hierarchical_labels(y.to(device, non_blocking=True))
            y = torch.nan_to_num(y, nan=0.0)

            logits = model(imgs)
            loss = criterion(logits, y, epoch)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)
            v_probs.append(probs.cpu())
            v_preds.append((probs > 0.5).int().cpu())
            v_targs.append(y.cpu())

    val_loss /= len(val_loader)
    v_probs = torch.cat(v_probs)
    v_preds = torch.cat(v_preds)
    v_targs = torch.cat(v_targs)

    val_acc = multilabel_accuracy(v_targs.numpy(), v_preds.numpy())
    viol = hierarchy_violation_rate(v_probs.numpy())

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    history["violation"].append(viol)

    print(f"Train Loss {tr_loss:.4f} | Acc {tr_acc:.4f}")
    print(f"Val   Loss {val_loss:.4f} | Acc {val_acc:.4f}")
    print(f"Hierarchy Violation Rate {viol:.4f}")

    torch.cuda.empty_cache()
    gc.collect()



# ============================================================
# FINAL EVALUATION
# ============================================================
print("\nCLASSIFICATION REPORT")
print(classification_report(
    v_targs.numpy(),
    v_preds.numpy(),
    target_names=ALL_LABELS,
    zero_division=0
))
print("Overall Accuracy:", multilabel_accuracy(v_targs.numpy(), v_preds.numpy()))

cm = confusion_matrix(v_targs.numpy().ravel(), v_preds.numpy().ravel())
ConfusionMatrixDisplay(cm, ["Neg","Pos"]).plot(cmap="Blues")
plt.title("Global Confusion Matrix")
plt.show()

plt.figure(figsize=(7,7))
for i,l in enumerate(ALL_LABELS):
    p,r,_ = precision_recall_curve(v_targs[:,i], v_probs[:,i])
    ap = average_precision_score(v_targs[:,i], v_probs[:,i])
    plt.plot(r,p,label=f"{l[:10]} AP={ap:.2f}")
plt.legend(bbox_to_anchor=(1.05,1))
plt.title("Precision–Recall Curves")
plt.grid()
plt.show()

fpr,tpr,_ = roc_curve(v_targs.numpy().ravel(), v_probs.numpy().ravel())
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.title("Overall ROC Curve")
plt.legend(); plt.grid(); plt.show()

print("ROC-AUC:", roc_auc)

# ============================================================
# TRAINING vs VALIDATION PLOTS
# ============================================================

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(14, 5))

# -------- LOSS --------
plt.subplot(1, 2, 1)
plt.plot(epochs, history["train_loss"], marker='o', label="Train Loss")
plt.plot(epochs, history["val_loss"], marker='o', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

# -------- ACCURACY --------
plt.subplot(1, 2, 2)
plt.plot(epochs, history["train_acc"], marker='o', label="Train Accuracy")
plt.plot(epochs, history["val_acc"], marker='o', label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
plt.plot(epochs, history["violation"], marker='o', color='red')
plt.xlabel("Epoch")
plt.ylabel("Violation Rate")
plt.title("Hierarchy Violation Rate per Epoch")
plt.grid(True)
plt.show()
