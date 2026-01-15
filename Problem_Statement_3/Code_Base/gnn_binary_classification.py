import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)

from torch_geometric.nn import GATConv, global_mean_pool

class DummyChestXrayDataset(Dataset):
    def __init__(self, n=100):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)  # fake X-ray
        label = torch.randint(0, 2, (1,)).item()
        return image, label

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.encoder(x)  # [B, 2048, H, W]

def feature_map_to_nodes(fmap):
    B, C, H, W = fmap.shape
    nodes = fmap.view(B, C, H * W).permute(0, 2, 1)
    return nodes, H, W

def build_lung_graph(h, w):
    edges = []

    def idx(i, j): return i * w + j

    for i in range(h):
        for j in range(w):
            if i + 1 < h:
                edges += [[idx(i,j), idx(i+1,j)], [idx(i+1,j), idx(i,j)]]
            if j + 1 < w:
                edges += [[idx(i,j), idx(i,j+1)], [idx(i,j+1), idx(i,j)]]

            mirror_j = w - j - 1
            if mirror_j != j:
                edges += [[idx(i,j), idx(i,mirror_j)]]

    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def add_disease_node(x, edge_index):
    device = x.device
    n = x.size(0)

    disease_node = torch.zeros(1, x.size(1), device=device)
    x = torch.cat([x, disease_node], dim=0)

    disease_edges = []
    for i in range(n):
        disease_edges.append([i, n])
        disease_edges.append([n, i])

    disease_edges = torch.tensor(disease_edges, device=device).t()
    edge_index = torch.cat([edge_index, disease_edges], dim=1)

    return x, edge_index

class GATClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.gat1 = GATConv(in_dim, 256, heads=4, concat=False)
        self.gat2 = GATConv(256, 128, heads=4, concat=False)

        self.att_pool = nn.Linear(128, 1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch):
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x, _ = self.gat2(x, edge_index, return_attention_weights=True)

        weights = torch.sigmoid(self.att_pool(x))
        graph_feat = global_mean_pool(x * weights, batch)

        out = torch.sigmoid(self.classifier(graph_feat))
        return out, weights.squeeze()

class ChestXrayGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()
        self.gnn = GATClassifier(2048)

    def forward(self, images):
        fmap = self.cnn(images)
        nodes, H, W = feature_map_to_nodes(fmap)

        preds, atts = [], []

        for b in range(images.size(0)):
            x = nodes[b]
            edge_index = build_lung_graph(H, W).to(x.device)
            x, edge_index = add_disease_node(x, edge_index)

            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pred, att = self.gnn(x, edge_index, batch)

            preds.append(pred)
            atts.append(att)

        return torch.cat(preds), atts, H, W

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.float().to(device)

        preds, _, _, _ = model(imgs)
        loss = criterion(preds.squeeze(), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            p, _, _, _ = model(imgs)
            probs.extend(p.squeeze().cpu().numpy())
            labels.extend(lbls.cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "AUC": roc_auc_score(labels, probs)
    }

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, label=f"AUC={metrics['AUC']:.3f}")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()

    # PR Curve
    pr, rc, _ = precision_recall_curve(labels, probs)
    plt.plot(rc, pr)
    plt.title("Precision-Recall Curve")
    plt.show()

    return metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = DummyChestXrayDataset(200)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset, batch_size=4)

model = ChestXrayGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = FocalLoss()

# Train
for epoch in range(2):
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Evaluate
metrics = evaluate_model(model, test_loader, device)
print(metrics)


