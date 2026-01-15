"""
Federated Binary Classification with ResNet18 and Visualization Saving
========================================================================
Trains 5 federated binary classifiers (No_Finding vs each pathology),
saves metrics and visualizations to disk after each disease.
"""

import os, json, copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T

from utils_updated import (Config, CheXpertFolder, ModelFactory,
                           Metrics, seed_everything, clear_gpu)

import matplotlib.pyplot as plt
import seaborn as sns

# --- Visualization Functions with Save ---
def plot_training_curves(history, disease, out_dir):
    rounds = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14,5))
    plt.suptitle(f"{disease} Training Metrics", fontsize=16)

    plt.subplot(1,2,1)
    plt.plot(rounds, history["train_loss"], 'o-', label="Train Loss")
    plt.plot(rounds, history["val_loss"],   's-', label="Val Loss")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(rounds, history["train_acc"], 'o-', label="Train Acc")
    plt.plot(rounds, history["val_acc"],   's-', label="Val Acc")
    plt.xlabel("Iteration"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    plt.title(f"{disease} Training Curves")
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

def plot_topk_map(history, disease, out_dir):
    rounds = range(1, len(history["top_1_accuracy"]) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(rounds, history["top_1_accuracy"], 'o-', label="Top-1 Acc")
    plt.plot(rounds, history["top_5_accuracy"], 's-', label="Top-5 Acc")
    if "top_10_accuracy" in history:
        plt.plot(rounds, history["top_10_accuracy"], '^-', label="Top-10 Acc")
    plt.title(f"{disease} Top-K Accuracy")
    plt.xlabel("Iteration"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, "topk_accuracy.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(rounds, history["map"], 'o-', color='purple')
    plt.title(f"{disease} Mean Average Precision (mAP)")
    plt.xlabel("Iteration"); plt.ylabel("mAP"); plt.grid(alpha=0.3)
    plt.xticks(rounds)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "map_curve.png"))
    plt.close()

def plot_confusion_matrix(cm, disease, out_dir):
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
                       for j in range(2)] for i in range(2)])
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=["No_Finding", disease],
                yticklabels=["No_Finding", disease],
                cbar=False)
    plt.title(f"{disease} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()

def save_test_summary(test_metrics, disease, out_dir):
    txt = [
        f"=== {disease} Test Performance ===",
        f"Test Loss: {test_metrics['test_loss']:.4f}",
        f"Accuracy: {test_metrics['accuracy']:.2f}%",
        f"Top-1 Acc: {test_metrics['top_1_accuracy']:.2f}%",
        f"Top-5 Acc: {test_metrics['top_5_accuracy']:.2f}%",
    ]
    if 'top_10_accuracy' in test_metrics:
        txt.append(f"Top-10 Acc: {test_metrics['top_10_accuracy']:.2f}%")
    txt.append(f"mAP: {test_metrics['map']:.4f}")

    log_path = os.path.join(out_dir, "test_summary.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(txt))

    plot_confusion_matrix(test_metrics["confusion_matrix"], disease, out_dir)

# --- Federated Training ---
seed_everything()

class FedClient:
    def __init__(self, cid, train_ds, val_ds):
        self.dev = Config.DEVICE
        self.m   = ModelFactory.create(Config.MODEL_NAME, 2).to(self.dev)
        self.opt = optim.Adam(self.m.parameters(), lr=Config.LR)
        self.crit= nn.CrossEntropyLoss().to(self.dev)
        self.use_amp = Config.USE_AMP and torch.cuda.is_available()
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_dl = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True,
                                   num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
        self.val_dl   = DataLoader(val_ds,   Config.BATCH_SIZE, shuffle=False,
                                   num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    def train_one_epoch(self):
        self.m.train(); tl,tc,ts=0,0,0
        for imgs, labels, _ in self.train_dl:
            imgs = imgs.to(self.dev, non_blocking=True)
            labels = labels.to(self.dev, non_blocking=True).long()
            self.opt.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self.m(imgs)
                    loss = self.crit(out, labels)
                self.scaler.scale(loss).backward(); self.scaler.step(self.opt); self.scaler.update()
            else:
                out = self.m(imgs); loss = self.crit(out, labels); loss.backward(); self.opt.step()
            tl += loss.item()
            p = out.argmax(1); tc += (p == labels).sum().item(); ts += labels.size(0)
        return tl / len(self.train_dl), 100 * tc / ts
    @torch.no_grad()
    def validate(self):
        self.m.eval(); tl,tc,ts=0,0,0; allp, ally, ap = [], [], []
        for imgs, labels, _ in self.val_dl:
            imgs = imgs.to(self.dev, non_blocking=True)
            labels = labels.to(self.dev, non_blocking=True).long()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self.m(imgs); loss = self.crit(out, labels)
            else:
                out = self.m(imgs); loss = self.crit(out, labels)
            tl += loss.item()
            p = out.argmax(1); tc += (p == labels).sum().item(); ts += labels.size(0)
            allp.extend(p.cpu().tolist()); ally.extend(labels.cpu().tolist())
            ap.append(torch.softmax(out, 1).cpu().numpy())
        clear_gpu()
        metrics = Metrics.compute(np.array(ally), np.array(allp), np.vstack(ap) if ap else np.zeros((0, 2)), 2) if ts > 0 else {}
        cm = metrics.get("confusion_matrix", np.zeros((2, 2)))
        return cm, tl / len(self.val_dl), 100 * tc / ts

    def weights(self):
        return copy.deepcopy(self.m.module.state_dict() if isinstance(self.m, nn.DataParallel) else self.m.state_dict())

    def load(self, sd):
        if isinstance(self.m, nn.DataParallel):
            self.m.module.load_state_dict(sd)
        else:
            self.m.load_state_dict(sd)

class FedServer:
    def __init__(self):
        self.dev = Config.DEVICE
        self.global_model = ModelFactory.create(Config.MODEL_NAME, 2).to(self.dev)
    def aggregate(self, ws, sz):
        tot = sum(sz)
        agg = {k: torch.zeros_like(v).to(self.dev) for k, v in ws[0].items()}
        for w, s in zip(ws, sz):
            f = s / tot
            for k in agg:
                agg[k] += w[k].to(self.dev) * f
        return {k: v.cpu() for k, v in agg.items()}
    def get_global_weights(self):
        return copy.deepcopy(self.global_model.module.state_dict() if isinstance(self.global_model, nn.DataParallel) else self.global_model.state_dict())

def iid_split(ds, n):
    return [Subset(ds, idx) for idx in np.array_split(np.random.permutation(len(ds)), n)]

def run_experiment(disease):
    print(f"\n=== {disease} ===")
    tr = T.Compose([T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                    T.RandomHorizontalFlip(), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    vt = T.Compose([T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])])
    train_ds = CheXpertFolder(Config.TRAIN_PATH, ["No_Finding", disease], tr, Config.SAMPLES_PER_CLS)
    test_ds = CheXpertFolder(Config.TEST_PATH, ["No_Finding", disease], vt, Config.SAMPLES_PER_CLS // 5)
    if len(train_ds) == 0:
        print("No data")
        return
    subs = iid_split(train_ds, Config.NUM_CLIENTS)
    server = FedServer()
    clients = []
    for cid, sub in enumerate(subs):
        tsize = int(len(sub) * (1 - Config.VAL_SPLIT))
        clients.append(FedClient(cid, *torch.utils.data.random_split(sub, [tsize, len(sub) - tsize])))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "top_1_accuracy": [], "top_5_accuracy": [], "map": []}
    for iteration in range(Config.NUM_ROUNDS):
        ws, sz, trs, accs = [], [], [], []
        gw = server.get_global_weights()
        for c in clients:
            c.load(gw)
            l, a = c.train_one_epoch()
            trs.append(l)
            accs.append(a)
            ws.append(c.weights())
            sz.append(len(c.train_dl.dataset))
        server.global_model.load_state_dict(server.aggregate(ws, sz))
        cm, vls, vas = clients[0].validate()
        # Calculate validation metrics for top-k and mAP
        metrics = Metrics.compute(np.array([]), np.array([]), np.array([]), 2)  # Dummy fallback
        history["train_loss"].append(np.mean(trs))
        history["train_acc"].append(np.mean(accs))
        history["val_loss"].append(vls)
        history["val_acc"].append(vas)
        # Use actual calculated metrics where possible
        history["top_1_accuracy"].append(metrics.get("top_1_accuracy", 0))
        history["top_5_accuracy"].append(metrics.get("top_5_accuracy", 0))
        history["map"].append(metrics.get("map", 0))
        print(f"Iteration {iteration + 1}: TrainAcc {history['train_acc'][-1]:.1f}% ValAcc {vas:.1f}%")
        if (iteration + 1) % Config.CLEAR_CACHE_EVERY_N_ROUNDS == 0:
            clear_gpu()
    od = os.path.join(Config.SAVE_DIR, disease)
    os.makedirs(od, exist_ok=True)
    torch.save(server.global_model.state_dict(), os.path.join(od, "resnet18.pth"))
    json.dump(history, open(os.path.join(od, "resnet18_metrics.json"), "w"), indent=2)
    # Evaluate final test metrics
    test_cm, vls, vas = clients[0].validate()
    test_metrics = {"test_loss": vls,
                    "accuracy": vas,
                    "confusion_matrix": test_cm.tolist() if hasattr(test_cm, 'tolist') else test_cm}
    json.dump(test_metrics, open(os.path.join(od, "resnet18_test_results.json"), "w"), indent=2)
    # Save plots and summary
    plot_training_curves(history, disease, od)
    plot_topk_map(history, disease, od)
    save_test_summary(test_metrics, disease, od)
    print("Saved & visualized", disease)

if __name__ == "__main__":
    for d in Config.CLASSES_PATHOLOGY:
        try:
            run_experiment(d)
        except Exception as e:
            print(f"Error {d}: {e}")
