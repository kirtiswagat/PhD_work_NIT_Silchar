# Mount Google Drive to access datasets
from google.colab import drive
drive.mount('/content/drive')

"""
Federated Learning for Multi-Class Disease Classification from Chest X-rays
============================================================================
Fixed version with proper dtype handling for CrossEntropyLoss
"""

import os
import random
import copy
import gc
import warnings
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from PIL import Image

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, 
    recall_score, precision_score, average_precision_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
class Config:
    """Configuration class for federated learning experiments"""
    
    # Class definitions
    CLASSES_PATHOLOGY = ["Atelectasis", "Cardiomegaly", "Consolidation", 
                        "Edema", "Pleural_Effusion"]
    CLASSES_WITH_NO_FINDING = ["No_Finding"] + CLASSES_PATHOLOGY
    
    # Data paths
    DATA_ROOT = "/content/drive/MyDrive/Colab_Datasets/chexpert_dataset"
    TRAIN_PATH = os.path.join(DATA_ROOT, "train")
    TEST_PATH = os.path.join(DATA_ROOT, "test")
    
    # Federated learning parameters
    NUM_CLIENTS = 5
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    IMAGE_SIZE = 224
    TRAIN_SAMPLES_PER_CLASS = 500
    TEST_SAMPLES_PER_CLASS = 200
    VAL_SPLIT = 0.2
    
    # GPU optimization
    USE_AMP = True  # Automatic Mixed Precision
    NUM_WORKERS = 2  # Reduced for stability
    PIN_MEMORY = True
    GRADIENT_ACCUMULATION_STEPS = 2
    CLEAR_CACHE_EVERY_N_ROUNDS = 3
    
    # Model settings
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Save directories
    SAVE_DIR = "/content/drive/MyDrive/Colab_Datasets/federated_results"
    BINARY_MODELS_DIR = os.path.join(SAVE_DIR, "binary_models")
    MULTI_CLASS_DIR = os.path.join(SAVE_DIR, "multi_class_model")
    VISUALIZATIONS_DIR = os.path.join(SAVE_DIR, "visualizations")
    
    # Device configuration
    DEVICE = None  # Will be set in setup_environment()


# ==================== Environment Setup ====================
def setup_environment():
    """Setup GPU environment and create necessary directories"""
    
    # GPU setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False
        
        print("=" * 80)
        print("GPU CONFIGURATION")
        print("=" * 80)
        print(f"GPU Available: True")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB")
        print(f"GPU Capability: {gpu_props.major}.{gpu_props.minor}")
        print("=" * 80)
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    Config.DEVICE = device
    
    # Create directories
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.BINARY_MODELS_DIR, exist_ok=True)
    os.makedirs(Config.MULTI_CLASS_DIR, exist_ok=True)
    os.makedirs(Config.VISUALIZATIONS_DIR, exist_ok=True)
    
    for disease in Config.CLASSES_PATHOLOGY:
        os.makedirs(os.path.join(Config.VISUALIZATIONS_DIR, disease), exist_ok=True)
    
    return device

def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==================== Dataset ====================
class CheXpertDataset(Dataset):
    """Custom dataset for CheXpert chest X-ray images"""
    
    def __init__(self, root: str, classes: List[str], transform=None, 
                 samples_per_class: int = 1000, is_multi_class: bool = False):
        self.root = root
        self.transform = transform
        self.classes = classes
        self.is_multi_class = is_multi_class
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.samples = []
        self.targets = []
        
        # Load samples
        for cls in classes:
            class_dir = os.path.join(root, cls)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample if necessary
            if len(files) > samples_per_class:
                files = random.sample(files, samples_per_class)
            
            for filepath in files:
                self.samples.append(filepath)
                self.targets.append(self.class_to_idx[cls])
        
        # Shuffle samples
        if self.samples:
            combined = list(zip(self.samples, self.targets))
            random.shuffle(combined)
            self.samples, self.targets = zip(*combined)
            self.samples = list(self.samples)
            self.targets = list(self.targets)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load and transform image
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        
        # Ensure target is integer and create label tensor
        target_idx = int(target)
        label = target_idx  # Return as integer, will be converted to tensor in collate_fn
        
        # Multi-label representation for metrics computation
        multi_label = torch.zeros(len(self.classes), dtype=torch.float32)
        multi_label[target_idx] = 1.0
        
        return img, label, multi_label

# Custom collate function to ensure proper label dtype
def custom_collate_fn(batch):
    """Custom collate function to ensure labels are LongTensor"""
    images, labels, multi_labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Convert labels to LongTensor
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Stack multi-labels
    multi_labels = torch.stack(multi_labels, 0)
    
    return images, labels, multi_labels

# ==================== Model Factory ====================
class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int, pretrained: bool = True):
        """Create a model with specified architecture"""
        
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            in_features = model.fc.in_features
            # Simple linear layer without dropout for stability
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Initialize the final layer properly
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        
        # Use DataParallel if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        return model

# ==================== Metrics Calculator ====================
class MetricsCalculator:
    """Calculate comprehensive metrics for classification"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray, num_classes: int, 
                       class_names: List[str] = None) -> Dict:
        """Compute comprehensive metrics"""
        
        metrics = {}
        
        # Handle empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0,
                'recall': 0.0, 'top_1_accuracy': 0.0, 'top_5_accuracy': 0.0,
                'map': 0.0, 'auc': 0.0, 'confusion_matrix': [[0]]
            }
        
        # Basic metrics
        metrics['accuracy'] = (y_true == y_pred).mean() * 100
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Top-K accuracy
        for k in [1, 5, 10]:
            if k <= num_classes:
                metrics[f'top_{k}_accuracy'] = MetricsCalculator._top_k_accuracy(y_true, y_prob, k)
        
        # AUC and mAP
        try:
            if num_classes == 2:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['map'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                y_bin = label_binarize(y_true, classes=range(num_classes))
                metrics['auc'] = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')
                metrics['map'] = MetricsCalculator._mean_average_precision(y_true, y_prob, num_classes)
        except Exception as e:
            print(f"Warning: Could not compute AUC/mAP: {e}")
            metrics['auc'] = 0.0
            metrics['map'] = 0.0
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Per-class metrics if class names provided
        if class_names:
            try:
                report = classification_report(y_true, y_pred, target_names=class_names, 
                                             output_dict=True, zero_division=0)
                metrics['per_class_metrics'] = report
            except:
                pass
        
        return metrics
    
    @staticmethod
    def _top_k_accuracy(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy"""
        if len(y_true) == 0:
            return 0.0
        top_k = np.argsort(y_prob, axis=1)[:, -k:]
        correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
        return (correct / len(y_true)) * 100
    
    @staticmethod
    def _mean_average_precision(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
        """Calculate mean average precision"""
        y_bin = label_binarize(y_true, classes=range(num_classes))
        aps = []
        for i in range(num_classes):
            try:
                ap = average_precision_score(y_bin[:, i], y_prob[:, i])
                aps.append(ap)
            except:
                aps.append(0.0)
        return np.mean(aps) if aps else 0.0

# ==================== Federated Client ====================
class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: int, train_dataset, val_dataset, num_classes: int = 2):
        self.client_id = client_id
        self.device = Config.DEVICE
        self.num_classes = num_classes
        
        # Model and optimizer
        self.model = ModelFactory.create_model(Config.MODEL_NAME, num_classes, Config.PRETRAINED)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LR, 
                                   weight_decay=Config.WEIGHT_DECAY)
        
        # Use CrossEntropyLoss with label smoothing for stability
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        
        # Mixed precision training
        self.use_amp = Config.USE_AMP and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Data loaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            collate_fn=custom_collate_fn, drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            collate_fn=custom_collate_fn, drop_last=False
        )
    
    def train_epochs(self, num_epochs: int) -> Tuple[float, float]:
        """Train for specified number of epochs"""
        
        total_loss = 0
        total_acc = 0
        
        for epoch in range(num_epochs):
            loss, acc = self._train_one_epoch()
            total_loss += loss
            total_acc += acc
        
        return total_loss / num_epochs, total_acc / num_epochs
    
    def _train_one_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    # Ensure outputs and labels have compatible shapes
                    if outputs.shape[0] != labels.shape[0]:
                        min_batch = min(outputs.shape[0], labels.shape[0])
                        outputs = outputs[:min_batch]
                        labels = labels[:min_batch]
                    loss = self.criterion(outputs, labels)
                
                # Gradient accumulation
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                # Ensure outputs and labels have compatible shapes
                if outputs.shape[0] != labels.shape[0]:
                    min_batch = min(outputs.shape[0], labels.shape[0])
                    outputs = outputs[:min_batch]
                    labels = labels[:min_batch]
                loss = self.criterion(outputs, labels)
                
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Calculate accuracy
            running_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Handle case where no batches were processed
        if total == 0:
            return 0.0, 0.0
        
        accuracy = 100.0 * correct / total
        avg_loss = running_loss / max(len(self.train_loader), 1)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, Dict]:
        """Evaluate model on validation set"""
        
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        for images, labels, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    # Ensure compatible shapes
                    if outputs.shape[0] != labels.shape[0]:
                        min_batch = min(outputs.shape[0], labels.shape[0])
                        outputs = outputs[:min_batch]
                        labels = labels[:min_batch]
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                # Ensure compatible shapes
                if outputs.shape[0] != labels.shape[0]:
                    min_batch = min(outputs.shape[0], labels.shape[0])
                    outputs = outputs[:min_batch]
                    labels = labels[:min_batch]
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            # Use torch.softmax instead of F.softmax
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        # Calculate metrics
        if len(all_probs) > 0:
            all_probs = np.vstack(all_probs)
            metrics = MetricsCalculator.compute_metrics(
                np.array(all_labels), np.array(all_predictions), 
                all_probs, self.num_classes
            )
        else:
            metrics = {
                'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0,
                'recall': 0.0, 'top_1_accuracy': 0.0, 'top_5_accuracy': 0.0,
                'map': 0.0, 'auc': 0.0, 'confusion_matrix': [[0]]
            }
        
        avg_loss = running_loss / max(len(self.val_loader), 1)
        accuracy = metrics['accuracy']
        
        return avg_loss, accuracy, metrics
    
    def get_model_weights(self) -> Dict:
        """Get model weights"""
        if isinstance(self.model, nn.DataParallel):
            return copy.deepcopy(self.model.module.state_dict())
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_weights(self, weights: Dict):
        """Set model weights"""
        try:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")


# ==================== Federated Server ====================
class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, num_classes: int = 2):
        self.device = Config.DEVICE
        self.num_classes = num_classes
        self.global_model = ModelFactory.create_model(Config.MODEL_NAME, num_classes, Config.PRETRAINED)
        self.global_model = self.global_model.to(self.device)
    
    def aggregate_weights(self, client_weights: List[Dict], client_sizes: List[int]) -> Dict:
        """Aggregate client weights using FedAvg with robust dtype handling"""
        
        total_size = sum(client_sizes)
        if total_size == 0:
            return self.get_global_weights()
        
        # Use first client's weights as template
        aggregated_weights = {}
        
        # Process each parameter separately with proper dtype handling
        for key in client_weights[0].keys():
            # Get the first weight tensor to determine dtype and shape
            first_weight = client_weights[0][key]
            
            # Check if this is a BatchNorm running stat (usually integers)
            is_bn_stat = 'num_batches_tracked' in key
            
            if is_bn_stat:
                # For batch norm tracking, just use the first client's value
                aggregated_weights[key] = first_weight.cpu()
            else:
                # For regular weights, perform weighted averaging
                # Initialize accumulator with float32 for precision
                accumulator = torch.zeros_like(first_weight, dtype=torch.float32).to(self.device)
                
                # Accumulate weighted contributions
                for weights, size in zip(client_weights, client_sizes):
                    weight_factor = float(size) / float(total_size)
                    contribution = weights[key].to(self.device).to(torch.float32) * weight_factor
                    accumulator = accumulator + contribution
                
                # Convert back to original dtype
                if first_weight.dtype in [torch.float16, torch.bfloat16]:
                    aggregated_weights[key] = accumulator.to(first_weight.dtype).cpu()
                elif first_weight.dtype in [torch.int32, torch.int64, torch.long]:
                    aggregated_weights[key] = accumulator.round().to(first_weight.dtype).cpu()
                else:
                    aggregated_weights[key] = accumulator.to(first_weight.dtype).cpu()
        
        return aggregated_weights
    
    def get_global_weights(self) -> Dict:
        """Get global model weights"""
        if isinstance(self.global_model, nn.DataParallel):
            return copy.deepcopy(self.global_model.module.state_dict())
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_global_weights(self, weights: Dict):
        """Set global model weights"""
        try:
            if isinstance(self.global_model, nn.DataParallel):
                self.global_model.module.load_state_dict(weights)
            else:
                self.global_model.load_state_dict(weights)
        except Exception as e:
            print(f"Warning: Could not set global weights: {e}")

# ==================== Visualization ====================
class Visualizer:
    """Create and save visualizations"""
    
    @staticmethod
    def plot_training_curves(history: Dict, disease: str, save_dir: str):
        """Plot training and validation curves"""
        
        iterations = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(iterations, history['train_loss'], 'o-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(iterations, history['val_loss'], 's-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Round', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title(f'{disease} - Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(iterations, history['train_acc'], 'o-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(iterations, history['val_acc'], 's-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Round', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title(f'{disease} - Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(iterations, history['f1_score'], 'o-', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Round', fontsize=12)
        axes[1, 0].set_ylabel('F1 Score', fontsize=12)
        axes[1, 0].set_title(f'{disease} - F1 Score', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # mAP
        axes[1, 1].plot(iterations, history['map'], 'o-', color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Round', fontsize=12)
        axes[1, 1].set_ylabel('mAP', fontsize=12)
        axes[1, 1].set_title(f'{disease} - Mean Average Precision', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{disease} Training Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_top_k_accuracy(history: Dict, disease: str, save_dir: str):
        """Plot top-k accuracy curves"""
        
        iterations = range(1, len(history['top_1_accuracy']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, history['top_1_accuracy'], 'o-', label='Top-1', linewidth=2)
        plt.plot(iterations, history['top_5_accuracy'], 's-', label='Top-5', linewidth=2)
        if 'top_10_accuracy' in history:
            plt.plot(iterations, history['top_10_accuracy'], '^-', label='Top-10', linewidth=2)
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'{disease} - Top-K Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'top_k_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_dir: str, title: str = "Confusion Matrix"):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Create annotations
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

# ==================== Binary Classification Training ====================
def train_binary_classifier(disease: str) -> Dict:
    """Train binary classifier for No_Finding vs disease"""
    
    print(f"\n{'='*80}")
    print(f"Training Binary Classifier: No_Finding vs {disease}")
    print(f"{'='*80}")
    
    # Data transforms
    train_transform = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    classes = ["No_Finding", disease]
    train_dataset = CheXpertDataset(
        Config.TRAIN_PATH, classes, train_transform, 
        Config.TRAIN_SAMPLES_PER_CLASS, is_multi_class=False
    )
    
    test_dataset = CheXpertDataset(
        Config.TEST_PATH, classes, val_transform,
        Config.TEST_SAMPLES_PER_CLASS, is_multi_class=False
    )
    
    if len(train_dataset) == 0:
        print(f"No data found for {disease}")
        return None
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Split data for federated clients (IID split)
    client_datasets = []
    indices = np.random.permutation(len(train_dataset))
    splits = np.array_split(indices, Config.NUM_CLIENTS)
    
    for split in splits:
        client_dataset = Subset(train_dataset, split)
        # Further split into train and validation
        val_size = int(len(client_dataset) * Config.VAL_SPLIT)
        train_size = len(client_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            client_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        client_datasets.append((train_subset, val_subset))
    
    # Initialize server and clients
    server = FederatedServer(num_classes=2)
    clients = []
    
    for client_id, (train_subset, val_subset) in enumerate(client_datasets):
        client = FederatedClient(client_id, train_subset, val_subset, num_classes=2)
        clients.append(client)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'f1_score': [], 'precision': [], 'recall': [],
        'top_1_accuracy': [], 'top_5_accuracy': [], 
        'map': [], 'auc': []
    }
    
    # Federated training
    print(f"\nStarting Federated Training for {disease}")
    print("-" * 40)
    
    best_val_acc = 0
    best_weights = None
    
    for round_num in range(Config.NUM_ROUNDS):
        round_train_losses = []
        round_train_accs = []
        client_weights = []
        client_sizes = []
        
        # Get global weights
        global_weights = server.get_global_weights()
        
        # Train each client
        for client in clients:
            # Set global weights
            client.set_model_weights(global_weights)
            
            # Local training
            train_loss, train_acc = client.train_epochs(Config.LOCAL_EPOCHS)
            round_train_losses.append(train_loss)
            round_train_accs.append(train_acc)
            
            # Get updated weights
            client_weights.append(client.get_model_weights())
            client_sizes.append(len(client.train_loader.dataset))
        
        # Aggregate weights
        aggregated_weights = server.aggregate_weights(client_weights, client_sizes)
        server.set_global_weights(aggregated_weights)
        
        # Evaluate on first client's validation set
        clients[0].set_model_weights(aggregated_weights)
        val_loss, val_acc, metrics = clients[0].evaluate()
        
        # Update history
        history['train_loss'].append(np.mean(round_train_losses))
        history['train_acc'].append(np.mean(round_train_accs))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['f1_score'].append(metrics['f1_score'])
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])
        history['top_1_accuracy'].append(metrics['top_1_accuracy'])
        history['top_5_accuracy'].append(metrics.get('top_5_accuracy', 0))
        history['map'].append(metrics['map'])
        history['auc'].append(metrics['auc'])
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(aggregated_weights)
        
        print(f"Round {round_num + 1}/{Config.NUM_ROUNDS}: "
              f"Train Acc: {history['train_acc'][-1]:.2f}%, "
              f"Val Acc: {val_acc:.2f}%, "
              f"F1: {metrics['f1_score']:.3f}, "
              f"mAP: {metrics['map']:.3f}")
        
        # Clear GPU cache periodically
        if (round_num + 1) % Config.CLEAR_CACHE_EVERY_N_ROUNDS == 0:
            clear_gpu_cache()
    
    # Use best weights for final evaluation
    if best_weights is not None:
        server.set_global_weights(best_weights)
    
    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
        collate_fn=custom_collate_fn, drop_last=False
    )
    
    # Use first client for evaluation (with best weights)
    clients[0].set_model_weights(server.get_global_weights())
    clients[0].val_loader = test_loader  # Temporarily replace val_loader
    test_loss, test_acc, test_metrics = clients[0].evaluate()
    
    # Save model and results
    save_dir = os.path.join(Config.BINARY_MODELS_DIR, disease)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(save_dir, 'model.pth')
    torch.save(server.get_global_weights(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    results = {
        'disease': disease,
        'training_history': history,
        'test_metrics': test_metrics,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': best_val_acc
    }
    
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    viz_dir = os.path.join(Config.VISUALIZATIONS_DIR, disease)
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        Visualizer.plot_training_curves(history, disease, viz_dir)
        Visualizer.plot_top_k_accuracy(history, disease, viz_dir)
        Visualizer.plot_confusion_matrix(
            np.array(test_metrics['confusion_matrix']), 
            ["No_Finding", disease], viz_dir,
            f"{disease} - Test Confusion Matrix"
        )
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Save test summary
    summary_path = os.path.join(viz_dir, 'test_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"{disease} Test Performance\n")
        f.write(f"{'='*60}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"F1 Score: {test_metrics['f1_score']:.3f}\n")
        f.write(f"Precision: {test_metrics['precision']:.3f}\n")
        f.write(f"Recall: {test_metrics['recall']:.3f}\n")
        f.write(f"Top-1 Accuracy: {test_metrics['top_1_accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {test_metrics.get('top_5_accuracy', 0):.2f}%\n")
        f.write(f"mAP: {test_metrics['map']:.3f}\n")
        f.write(f"AUC: {test_metrics['auc']:.3f}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    
    print(f"Results saved for {disease}")
    print("-" * 40)
    
    # Clean up
    del clients
    del server
    clear_gpu_cache()
    
    return results

# ==================== Multi-Class Model Aggregation ====================
class MultiClassAggregator:
    """Aggregate binary models into multi-class classifier"""
    
    def __init__(self, binary_results: Dict[str, Dict]):
        self.binary_results = binary_results
        self.device = Config.DEVICE
        self.num_classes = len(Config.CLASSES_WITH_NO_FINDING)
        
    def create_multiclass_model(self) -> nn.Module:
        """Create multi-class model from binary classifiers"""
        
        print(f"\n{'='*80}")
        print("Creating Multi-Class Model from Binary Classifiers")
        print(f"{'='*80}")
        
        # Create base model
        multiclass_model = ModelFactory.create_model(
            Config.MODEL_NAME, self.num_classes, Config.PRETRAINED
        )
        multiclass_model = multiclass_model.to(self.device)
        
        # Load and aggregate binary model weights
        aggregated_weights = self._aggregate_binary_weights()
        
        if aggregated_weights:
            # Initialize multi-class model with aggregated weights
            multiclass_model = self._initialize_multiclass_weights(multiclass_model, aggregated_weights)
        
        return multiclass_model
    
    def _aggregate_binary_weights(self) -> Dict:
        """Aggregate weights from binary models"""
        
        aggregated_weights = {}
        valid_models = 0
        
        for disease, results in self.binary_results.items():
            if results is None:
                continue
                
            # Load binary model weights
            model_path = os.path.join(Config.BINARY_MODELS_DIR, disease, 'model.pth')
            if not os.path.exists(model_path):
                print(f"Warning: Model not found for {disease}")
                continue
                
            try:
                weights = torch.load(model_path, map_location=self.device)
                valid_models += 1
                
                # Aggregate weights (except final layer)
                for key, value in weights.items():
                    if 'fc' not in key:  # Don't aggregate final layer
                        if key not in aggregated_weights:
                            aggregated_weights[key] = value.clone()
                        else:
                            aggregated_weights[key] += value
            except Exception as e:
                print(f"Warning: Could not load model for {disease}: {e}")
        
        # Average the weights
        if valid_models > 0:
            for key in aggregated_weights.keys():
                aggregated_weights[key] = aggregated_weights[key] / valid_models
        
        return aggregated_weights
    
    def _initialize_multiclass_weights(self, model: nn.Module, aggregated_weights: Dict) -> nn.Module:
        """Initialize multi-class model with aggregated weights"""
        
        # Get current model state dict
        model_state = model.state_dict()
        
        # Update with aggregated weights (except final layer)
        for key in model_state.keys():
            if 'fc' not in key and key in aggregated_weights:
                model_state[key] = aggregated_weights[key]
        
        # Load updated weights
        model.load_state_dict(model_state)
        
        return model
    
    def fine_tune_multiclass(self, model: nn.Module, num_epochs: int = 10) -> Tuple[nn.Module, Dict]:
        """Fine-tune the multi-class model"""
        
        print("\nFine-tuning Multi-Class Model...")
        print("-" * 40)
        
        # Data transforms
        train_transform = T.Compose([
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = T.Compose([
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create multi-class datasets
        train_dataset = CheXpertDataset(
            Config.TRAIN_PATH, Config.CLASSES_WITH_NO_FINDING,
            train_transform, Config.TRAIN_SAMPLES_PER_CLASS, is_multi_class=True
        )
        
        test_dataset = CheXpertDataset(
            Config.TEST_PATH, Config.CLASSES_WITH_NO_FINDING,
            val_transform, Config.TEST_SAMPLES_PER_CLASS, is_multi_class=True
        )
        
        if len(train_dataset) == 0:
            print("No data found for multi-class training")
            return model, {}
        
        # Split train into train/val
        val_size = int(len(train_dataset) * Config.VAL_SPLIT)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            collate_fn=custom_collate_fn, drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset, batch_size=Config.BATCH_SIZE, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            collate_fn=custom_collate_fn, drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY,
            collate_fn=custom_collate_fn, drop_last=False
        )
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=Config.LR * 0.5, weight_decay=Config.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Mixed precision
        use_amp = Config.USE_AMP and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'f1_score': [], 'precision': [], 'recall': [],
            'top_1_accuracy': [], 'top_5_accuracy': [], 'top_10_accuracy': [],
            'map': [], 'auc': []
        }
        
        best_val_acc = 0
        best_model_weights = None
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for images, labels, _ in train_bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': train_loss / (train_bar.n + 1),
                    'acc': 100. * train_correct / train_total
                })
            
            # Validation phase
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_labels = []
            all_val_probs = []
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for images, labels, _ in val_bar:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    # Use torch.softmax instead of F.softmax
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_probs.append(probs.cpu().numpy())
            
            # Calculate metrics
            if len(all_val_probs) > 0:
                all_val_probs = np.vstack(all_val_probs)
                val_metrics = MetricsCalculator.compute_metrics(
                    np.array(all_val_labels), np.array(all_val_preds),
                    all_val_probs, self.num_classes, Config.CLASSES_WITH_NO_FINDING
                )
            else:
                val_metrics = {'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0,
                              'top_1_accuracy': 0, 'top_5_accuracy': 0, 'top_10_accuracy': 0,
                              'map': 0, 'auc': 0}
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100. * train_correct / train_total)
            history['val_loss'].append(val_loss / len(val_loader))
            history['val_acc'].append(val_metrics['accuracy'])
            history['f1_score'].append(val_metrics['f1_score'])
            history['precision'].append(val_metrics['precision'])
            history['recall'].append(val_metrics['recall'])
            history['top_1_accuracy'].append(val_metrics['top_1_accuracy'])
            history['top_5_accuracy'].append(val_metrics['top_5_accuracy'])
            history['top_10_accuracy'].append(val_metrics.get('top_10_accuracy', 0))
            history['map'].append(val_metrics['map'])
            history['auc'].append(val_metrics['auc'])
            
            print(f"\nEpoch {epoch+1}: "
                  f"Train Acc: {history['train_acc'][-1]:.2f}%, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                  f"F1: {val_metrics['f1_score']:.3f}, "
                  f"mAP: {val_metrics['map']:.3f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_weights = copy.deepcopy(model.state_dict())
            
            scheduler.step()
            
            # Clear cache
            if (epoch + 1) % 2 == 0:
                clear_gpu_cache()
        
        # Load best model
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        
        # Test evaluation
        print("\nEvaluating on test set...")
        model.eval()
        test_loss = 0
        all_test_preds = []
        all_test_labels = []
        all_test_probs = []
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc='Testing')
            for images, labels, _ in test_bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                # Use torch.softmax instead of F.softmax
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
                all_test_probs.append(probs.cpu().numpy())
        
        # Calculate test metrics
        if len(all_test_probs) > 0:
            all_test_probs = np.vstack(all_test_probs)
            test_metrics = MetricsCalculator.compute_metrics(
                np.array(all_test_labels), np.array(all_test_preds),
                all_test_probs, self.num_classes, Config.CLASSES_WITH_NO_FINDING
            )
        else:
            test_metrics = {'accuracy': 0, 'confusion_matrix': [[0]]}
        
        results = {
            'training_history': history,
            'test_metrics': test_metrics,
            'test_loss': test_loss / max(len(test_loader), 1),
            'best_val_acc': best_val_acc
        }
        
        return model, results
    

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    
    try:
        # Setup environment
        setup_environment()
        seed_everything(42)
        
        print("\n" + "="*80)
        print("FEDERATED LEARNING FOR MULTI-CLASS DISEASE CLASSIFICATION")
        print("="*80)
        print(f"Classes: {Config.CLASSES_WITH_NO_FINDING}")
        print(f"Training samples per class: {Config.TRAIN_SAMPLES_PER_CLASS}")
        print(f"Test samples per class: {Config.TEST_SAMPLES_PER_CLASS}")
        print(f"Number of clients: {Config.NUM_CLIENTS}")
        print(f"Number of rounds: {Config.NUM_ROUNDS}")
        print("="*80)
        
        # Step 1: Train binary classifiers
        binary_results = {}
        
        for disease in Config.CLASSES_PATHOLOGY:
            try:
                results = train_binary_classifier(disease)
                binary_results[disease] = results
                clear_gpu_cache()
            except Exception as e:
                print(f"Error training {disease}: {str(e)}")
                import traceback
                traceback.print_exc()
                binary_results[disease] = None
        
        # Step 2: Create and fine-tune multi-class model
        if any(r is not None for r in binary_results.values()):
            aggregator = MultiClassAggregator(binary_results)
            multiclass_model = aggregator.create_multiclass_model()
            multiclass_model, multiclass_results = aggregator.fine_tune_multiclass(multiclass_model, num_epochs=15)
            
            # Step 3: Save multi-class model and results
            print("\nSaving Multi-Class Model...")
            
            # Save model
            model_path = os.path.join(Config.MULTI_CLASS_DIR, 'multiclass_model.pth')
            if isinstance(multiclass_model, nn.DataParallel):
                torch.save(multiclass_model.module.state_dict(), model_path)
            else:
                torch.save(multiclass_model.state_dict(), model_path)
            print(f"Multi-class model saved to {model_path}")
            
            # Save results
            results_path = os.path.join(Config.MULTI_CLASS_DIR, 'multiclass_results.json')
            with open(results_path, 'w') as f:
                json.dump(multiclass_results, f, indent=2)
            
            # Create final visualizations
            viz_dir = Config.MULTI_CLASS_DIR
            try:
                Visualizer.plot_training_curves(multiclass_results['training_history'], 'Multi-Class', viz_dir)
                Visualizer.plot_top_k_accuracy(multiclass_results['training_history'], 'Multi-Class', viz_dir)
                Visualizer.plot_confusion_matrix(
                    np.array(multiclass_results['test_metrics']['confusion_matrix']),
                    Config.CLASSES_WITH_NO_FINDING, viz_dir,
                    "Multi-Class Confusion Matrix"
                )
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
            
            # Print final summary
            print("\n" + "="*80)
            print("FINAL MULTI-CLASS MODEL PERFORMANCE")
            print("="*80)
            test_metrics = multiclass_results['test_metrics']
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"F1 Score: {test_metrics['f1_score']:.3f}")
            print(f"Precision: {test_metrics['precision']:.3f}")
            print(f"Recall: {test_metrics['recall']:.3f}")
            print(f"Top-1 Accuracy: {test_metrics['top_1_accuracy']:.2f}%")
            print(f"Top-5 Accuracy: {test_metrics['top_5_accuracy']:.2f}%")
            print(f"Top-10 Accuracy: {test_metrics.get('top_10_accuracy', 0):.2f}%")
            print(f"mAP: {test_metrics['map']:.3f}")
            print(f"AUC: {test_metrics['auc']:.3f}")
            print("="*80)
            
            # Per-class performance
            if 'per_class_metrics' in test_metrics:
                print("\nPer-Class Performance:")
                print("-"*40)
                for class_name in Config.CLASSES_WITH_NO_FINDING:
                    if class_name in test_metrics['per_class_metrics']:
                        class_metrics = test_metrics['per_class_metrics'][class_name]
                        print(f"{class_name:20} - "
                              f"Precision: {class_metrics['precision']:.3f}, "
                              f"Recall: {class_metrics['recall']:.3f}, "
                              f"F1: {class_metrics['f1-score']:.3f}")
        else:
            print("\nNo binary classifiers were successfully trained. Cannot create multi-class model.")
        
        print("\n✓ Training Complete!")
        print(f"All results saved to: {Config.SAVE_DIR}")
        
    except Exception as e:
        print(f"\nFatal error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()