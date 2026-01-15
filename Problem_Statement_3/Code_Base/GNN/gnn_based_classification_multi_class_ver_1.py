import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random # Added for sampling

from torch_geometric.data import Data, Batch # Added Batch
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
pd.set_option('future.no_silent_downcasting', True)

from tqdm.autonotebook import tqdm


# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
SSL_WEIGHT = 0.2

CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural_Effusion",
    ]
import os
# ============================================================
# DATASET
# ============================================================
class CheXpertDataset(Dataset):
    def __init__(self, image_root, max_samples_per_class=1000):
        self.image_root = image_root
        self.data_list = [] # Will store (relative_path, label_tensor)

        # Iterate through each CHEXPERT_LABEL to find corresponding folders
        for class_idx, class_name in enumerate(CHEXPERT_LABELS):
            class_folder_path = os.path.join(image_root, class_name)

            if not os.path.isdir(class_folder_path):
                print(f"Warning: Class folder not found for label '{class_name}' at {class_folder_path}. Skipping.")
                continue

            # List all image files in the current class folder
            image_files_in_folder = [f for f in os.listdir(class_folder_path)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Apply max_samples_per_class if specified and if there are too many images
            if max_samples_per_class and len(image_files_in_folder) > max_samples_per_class:
                image_files_in_folder = random.sample(image_files_in_folder, max_samples_per_class)

            # Create a label tensor for this class (one-hot encoding)
            base_label_vector = [0.0] * len(CHEXPERT_LABELS)
            base_label_vector[class_idx] = 1.0
            label_tensor = torch.tensor(base_label_vector, dtype=torch.float32)

            for image_file in image_files_in_folder:
                # Store relative path to reconstruct full path in __getitem__
                relative_image_path = os.path.join(class_name, image_file)
                self.data_list.append((relative_image_path, label_tensor))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229])
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        relative_image_path, labels = self.data_list[idx]
        # Construct full image path
        img_path = os.path.join(self.image_root, relative_image_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, labels


# ============================================================
# CNN BACKBONE
# ============================================================
class CNNBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        base = models.resnet50(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.proj = nn.Conv2d(2048, out_dim, 1)

    def forward(self, x):
        return self.proj(self.features(x))  # (B, 512, H, W)


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
def build_graph(feature_map):
    graphs, batch_indices = [], [] # Renamed 'batch' to 'batch_indices' for clarity if it were used

    for b in range(feature_map.size(0)):
        fmap = feature_map[b]
        C, H, W = fmap.shape
        x = fmap.view(C, -1).T

        edges = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if j + 1 < W:
                    edges.append([idx, idx + 1])
                if i + 1 < H:
                    edges.append([idx, idx + W])

        if not edges:
            # If no edges, create an empty edge_index tensor with shape (2, 0)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T

        graphs.append(Data(x=x, edge_index=edge_index))
        # Keep returning batch_indices for consistency with the call signature in SSGL_CheXpert,
        # even if it's discarded with `_`
        batch_indices.append(torch.full((x.size(0),), b))

    return graphs, torch.cat(batch_indices)

# ============================================================
# GRAPH AUGMENTATIONS (SSL)
# ============================================================
def drop_edge(data, p=0.2):
    mask = torch.rand(data.edge_index.size(1)) > p
    return Data(x=data.x, edge_index=data.edge_index[:, mask])

def mask_node_features(data, p=0.2):
    x = data.x.clone()
    mask = torch.rand(x.size(0)) < p
    x[mask] = 0
    return Data(x=x, edge_index=data.edge_index)

def graph_augment(data):
    data = drop_edge(data)
    data = mask_node_features(data)
    return data

# ============================================================
# GNN ENCODER
# ============================================================
class GraphEncoder(nn.Module):
    def __init__(self, in_dim=512, hidden=256):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=4)
        self.gat2 = GATConv(hidden * 4, hidden)

    def forward(self, data, batch):
        x = F.elu(self.gat1(data.x, data.edge_index))
        x = self.gat2(x, data.edge_index)
        return global_mean_pool(x, batch)

# ============================================================
# CLASSIFIER
# ============================================================
class Classifier(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.fc = nn.Linear(emb_dim, len(CHEXPERT_LABELS))

    def forward(self, z):
        return self.fc(z)


# ============================================================
# FULL MODEL
# ============================================================
class SSGL_CheXpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()
        self.gnn = GraphEncoder()
        self.cls = Classifier()

    def forward(self, images, augment=False):
        feat = self.cnn(images)
        graphs, _ = build_graph(feat) # Removed batch as it will be generated by Batch.from_data_list

        if augment:
            graphs = [graph_augment(g) for g in graphs]

        # Use Batch.from_data_list for proper graph batching
        data_batch = Batch.from_data_list(graphs).to(images.device)

        # Pass the batched data and its internal batch attribute to the GNN
        z = self.gnn(data_batch, data_batch.batch)
        logits = self.cls(z)

        return z, logits


# ============================================================
# SSL LOSS
# ============================================================
def contrastive_loss(z1, z2, temp=0.5):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.T) / temp
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(sim, labels)

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_logits):
    y_prob = torch.sigmoid(y_logits).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true_np = y_true.cpu().numpy().astype(int) # Explicitly cast y_true to int

    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred),
        "precision": precision_score(y_true_np, y_pred, average="macro"),
        "recall": recall_score(y_true_np, y_pred, average="macro"),
        "f1": f1_score(y_true_np, y_pred, average="macro"),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true_np, y_prob, average="macro")
    except ValueError:
        metrics["auc"] = np.nan

    return metrics

# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_epoch(model, loader, optimizer):
    model.train()
    total = 0

    # Wrap the loader with tqdm for a progress bar
    pbar = tqdm(loader, desc="Training", leave=False)

    for img, lbl in pbar:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()

        z1, logits = model(img, augment=False)
        z2, _ = model(img, augment=True)

        loss = nn.BCEWithLogitsLoss()(logits, lbl) \
               + SSL_WEIGHT * contrastive_loss(z1, z2)

        loss.backward()
        optimizer.step()
        total += loss.item()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': loss.item()})

    return total / len(loader)

def validate(model, loader):
    model.eval()
    total, preds, labels = 0, [], []

    # Wrap the loader with tqdm for a progress bar
    pbar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for img, lbl in pbar:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            _, logits = model(img)
            batch_loss = nn.BCEWithLogitsLoss()(logits, lbl).item()
            total += batch_loss
            preds.append(logits)
            labels.append(lbl)

            # Update progress bar with current loss
            pbar.set_postfix({'loss': batch_loss})

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    return total / len(loader), compute_metrics(labels, preds), preds, labels

# ============================================================
# ROC CURVES
# ============================================================
def plot_roc(y_true, y_logits):
    y_true = y_true.cpu().numpy()
    y_prob = torch.sigmoid(y_logits).cpu().numpy()

    plt.figure(figsize=(8,6))
    for i, name in enumerate(CHEXPERT_LABELS):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.legend()
    plt.title("Per-Class ROC Curves")
    plt.grid()
    plt.show()

# ============================================================
# TRAINING HISTORY PLOT
# ============================================================
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot 2: Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['f1'], label='F1-score')
    plt.plot(epochs, history['auc'], label='AUC')
    plt.plot(epochs, history['val_accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ============================================================
# ABLATION STUDY
# ============================================================
def ablation(train_loader, val_loader):
    results = []

    for name, ssl in [("CNN+GNN", False), ("CNN+GNN+SSL", True)]:
        model = SSGL_CheXpert().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for _ in range(5):
            train_epoch(model, train_loader, optimizer)

        _, metrics, _, _ = validate(model, val_loader)
        metrics["Model"] = name
        results.append(metrics)

    return pd.DataFrame(results)

# ============================================================
# MAIN
# ============================================================
def main():
    base_data_path = "/content/drive/MyDrive/Kaggel_direct_download/GNN_Training_Data"

    # Instantiate a single CheXpertDataset from the base_data_path
    # This dataset will contain all images from all class folders as specified by CHEXPERT_LABELS
    full_dataset = CheXpertDataset(image_root=base_data_path)

    # Limit the total number of samples to 1000 if the dataset is larger
    max_total_samples = 1000
    if len(full_dataset) > max_total_samples:
        # Create a list of indices and sample from them
        indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(indices, max_total_samples)
        # Create a subset of the full_dataset using the sampled indices
        limited_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
        print(f"Limited dataset to {len(limited_dataset)} total samples.")
    else:
        limited_dataset = full_dataset
        print(f"Dataset has {len(limited_dataset)} total samples, no limiting applied.")

    # Define the split ratios (e.g., 80% train, 20% validation) for the limited dataset
    train_size = int(0.8 * len(limited_dataset))
    val_size = len(limited_dataset) - train_size

    # Split the dataset into training and validation sets
    train_ds, val_ds = torch.utils.data.random_split(limited_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = SSGL_CheXpert().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {k: [] for k in ["train_loss","val_loss","f1","auc", "val_accuracy"]}

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, metrics, preds, labels = validate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["f1"].append(metrics["f1"])
        history["auc"].append(metrics["auc"])
        history["val_accuracy"].append(metrics["accuracy"])

        print(f"Epoch {epoch+1}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
              f"Accuracy={metrics['accuracy']:.4f}")

    plot_training_history(history)
    plot_roc(labels, preds)
    print(ablation(train_loader, val_loader))

if __name__ == "__main__":
    main()