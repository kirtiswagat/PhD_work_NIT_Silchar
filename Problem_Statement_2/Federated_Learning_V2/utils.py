"""
Shared utilities for Federated CheXpert experiments
===================================================
"""

import os, random, copy, gc, warnings, time
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             f1_score, recall_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

def setup_gpu():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("="*60)
        print("GPU INFORMATION")
        print("="*60)
        print(f"GPU Available: True")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
        print("="*60)
    else:
        dev = torch.device("cpu")
        print("GPU not available, using CPU")
    return dev

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    CLASSES_PATHOLOGY = ["Atelectasis", "Cardiomegaly", "Consolidation",
                        "Edema", "Pleural_Effusion"]
    CLASSES_WITH_NO_FINDING = ["No_Finding"] + CLASSES_PATHOLOGY
    MODEL_NAME = "resnet18"
    DATA_ROOT = "/content/drive/MyDrive/Colab_Datasets/chexpert_dataset"
    TRAIN_PATH = os.path.join(DATA_ROOT, "train")
    TEST_PATH  = os.path.join(DATA_ROOT, "test")
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LR = 0.001
    IMAGE_SIZE = 224
    SAMPLES_PER_CLS = 1000
    VAL_SPLIT = 0.2
    USE_AMP = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    GRADIENT_ACCUMULATION_STEPS = 1
    CLEAR_CACHE_EVERY_N_ROUNDS = 2
    DEVICE = setup_gpu()
    SAVE_DIR = "/content/drive/MyDrive/Colab_Datasets/federated_models_split"
    os.makedirs(SAVE_DIR, exist_ok=True)

class CheXpertFolder(Dataset):
    def __init__(self, root, classes, transform, samples_per_class=1000):
        self.root, self.transform, self.classes = root, transform, classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples, self.targets = [], []
        for cls in classes:
            p = os.path.join(root, cls)
            if not os.path.isdir(p): continue
            files = [os.path.join(p,f) for f in os.listdir(p)
                     if f.lower().endswith((".jpg",".png"))]
            if len(files) > samples_per_class:
                files = random.sample(files, samples_per_class)
            for fn in files:
                self.samples.append(fn)
                self.targets.append(self.class_to_idx[cls])
        combo = list(zip(self.samples,self.targets))
        random.shuffle(combo)
        if combo: self.samples, self.targets = zip(*combo)
        else: self.samples, self.targets = [], []

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        lbl = torch.tensor(self.targets[idx], dtype=torch.long)
        ml  = torch.zeros(len(self.classes))
        ml[lbl] = 1
        return img, lbl, ml

class ModelFactory:
    @staticmethod
    def create(name, out_dim, pretrained=True):
        if name=="resnet18":
            m = models.resnet18(pretrained)
            m.fc = nn.Linear(m.fc.in_features, out_dim)
        else:
            m = models.resnet18(pretrained)
            m.fc = nn.Linear(m.fc.in_features, out_dim)
        if torch.cuda.device_count() > 1:
            m = nn.DataParallel(m)
        return m

class Metrics:
    @staticmethod
    def compute(y_true, y_pred, y_prob, n_cls):
        m = {}
        m["accuracy"] = (y_true == y_pred).mean() * 100
        m["f1_score"] = f1_score(y_true, y_pred, average="weighted")
        m["recall"] = recall_score(y_true, y_pred, average="weighted")
        for k in (1, 5, 10):
            if k <= n_cls:
                m[f"top_{k}_accuracy"] = Metrics.topk(y_true, y_prob, k)
        try:
            if n_cls == 2:
                m["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                yb = label_binarize(y_true, range(n_cls))
                m["auc"] = roc_auc_score(yb, y_prob, average="weighted", multi_class="ovr")
        except:
            m["auc"] = 0.0
        m["map"] = Metrics.mean_ap(y_true, y_prob, n_cls)
        m["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        return m

    @staticmethod
    def mean_ap(y_true, y_prob, n_cls):
        if n_cls == 2:
            return average_precision_score(y_true, y_prob[:, 1])
        yb = label_binarize(y_true, range(n_cls))
        aps = []
        for i in range(n_cls):
            try:
                aps.append(average_precision_score(yb[:, i], y_prob[:, i]))
            except:
                aps.append(0)
        return sum(aps) / len(aps)

    @staticmethod
    def topk(y_true, y_prob, k):
        tk = np.argsort(y_prob, axis=1)[:, -k:]
        return sum(t in tk[i] for i, t in enumerate(y_true)) / len(y_true) * 100
