"""
Federated Learning Pipeline for CheXpert Dataset
=================================================
This script implements a comprehensive federated learning pipeline with:
- 5 Client nodes with IID distribution
- Per-class and multi-class training
- Support for ResNet18, DenseNet121, and EfficientNet
- Extensive metrics tracking including MAP
- Dataset organized in class folders
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score,
                           recall_score, roc_curve, average_precision_score,
                           precision_recall_curve)
from sklearn.preprocessing import label_binarize
import copy
import random
from collections import defaultdict
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

def setup_gpu():
    """
    Setup GPU for training and print GPU information
    """
    if torch.cuda.is_available():
        print("="*60)
        print("GPU INFORMATION")
        print("="*60)
        print(f"GPU Available: {torch.cuda.is_available()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Clear cache
        torch.cuda.empty_cache()

        # Set GPU memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Enable cudNN autotuner for better performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Print initial memory usage
        print(f"Initial GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Initial GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print("="*60)

        return torch.device("cuda")
    else:
        print("GPU not available, using CPU")
        return torch.device("cpu")
    
def print_gpu_memory():
    """
    Print current GPU memory usage
    """
    if torch.cuda.is_available():
        print(f"GPU Memory: Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB, "
              f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

def clear_gpu_memory():
    """
    Clear GPU memory cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
class Config:
    # Paths
    TRAIN_PATH = "/content/drive/MyDrive/Colab_Datasets/chexpert_dataset/train"
    TEST_PATH = "/content/drive/MyDrive/Colab_Datasets/chexpert_dataset/test"

    # Classes
    CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural_Effusion"]
    NUM_CLASSES = len(CLASSES)

    # Federated Learning Settings
    NUM_CLIENTS = 5
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 5

    # Training Settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 224
    SAMPLES_PER_CLASS = 1000
    VAL_SPLIT = 0.2
    NUM_WORKERS = 2  # For DataLoader
    PIN_MEMORY = True  # For faster GPU transfer

    # Mixed Precision Training
    USE_AMP = True  # Automatic Mixed Precision
    GRADIENT_ACCUMULATION_STEPS = 1  # For larger effective batch size

    # # Device
    # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Device - Will be set by setup_gpu()
    DEVICE = None

    # Model Saves
    SAVE_DIR = "/content/drive/MyDrive/Colab_Datasets/federated_models"

    # Memory Management
    CLEAR_CACHE_EVERY_N_ROUNDS = 2

# Initialize GPU
Config.DEVICE = setup_gpu()

# Custom Dataset Class for folder-based structure
class CheXpertFolderDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, samples_per_class=1000):
        """
        Custom dataset for folder-organized CheXpert data
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.samples_per_class = samples_per_class
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.samples = []
        self.targets = []

        self._load_samples()

    def _load_samples(self):
        """
        Load samples from class folders with retry mechanism for OSError
        """
        import time

        max_retries = 5
        retry_delay = 5  # seconds

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found. Skipping class {class_name}")
                continue

            image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
            image_files = []

            # Add retry logic for os.listdir
            for attempt in range(max_retries):
                try:
                    all_files = os.listdir(class_dir)
                    image_files = [os.path.join(class_dir, f) for f in all_files if f.lower().endswith(image_extensions)]
                    break # Break if successful
                except OSError as e:
                    print(f"OSError reading directory {class_dir}: {e}. Attempt {attempt + 1}/{max_retries}.")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to read directory {class_dir} after {max_retries} attempts.")
                        # Optionally, raise the exception or skip the class
                        # raise e
                        image_files = [] # Skip if persistent error
                        break


            # Limit samples per class
            if len(image_files) > self.samples_per_class:
                image_files = random.sample(image_files, self.samples_per_class)

            # Add samples
            if image_files: # Only add samples if image_files is not empty after retries
                class_idx = self.class_to_idx[class_name]
                for img_path in image_files:
                    self.samples.append(img_path)
                    self.targets.append(class_idx)


        # Shuffle samples
        combined = list(zip(self.samples, self.targets))
        random.shuffle(combined)
        self.samples, self.targets = zip(*combined) if combined else ([], [])

        print(f"Loaded {len(self.samples)} total samples from {self.root_dir}")
        for class_name in self.classes:
            count = sum(1 for i, t in enumerate(self.targets) if t == self.class_to_idx[class_name])
            print(f"  {class_name}: {count} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.targets[idx]

        # Create multi-label representation (for compatibility)
        multi_label = torch.zeros(len(self.classes))
        multi_label[label] = 1.0

        return image, label, multi_label
    

# Model Factory
class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes, pretrained=True):
        """
        Create model based on name
        """
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_name == "efficientnet":
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Move model to GPU and enable DataParallel if multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for {model_name}")
            model = nn.DataParallel(model)

        return model
    
# Metrics Calculator with MAP
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_probs, num_classes):
        """
        Calculate comprehensive metrics including MAP
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred) * 100

        # F1 Score
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')

        # Recall
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')

        # Top-K Accuracy
        for k in [1, 5, 10]:
            if k <= num_classes:
                metrics[f'top_{k}_accuracy'] = MetricsCalculator.top_k_accuracy(y_true, y_probs, k)

        # AUC (for multi-class)
        if num_classes > 2:
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            try:
                metrics['auc'] = roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])

        # MAP (Mean Average Precision)
        metrics['map'] = MetricsCalculator.calculate_map(y_true, y_probs, num_classes)

        # Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return metrics

    @staticmethod
    def calculate_map(y_true, y_probs, num_classes):
        """
        Calculate Mean Average Precision
        """
        if num_classes == 2:
            # Binary classification
            return average_precision_score(y_true, y_probs[:, 1])
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(num_classes))

            # Calculate AP for each class
            ap_scores = []
            for i in range(num_classes):
                try:
                    ap = average_precision_score(y_true_bin[:, i], y_probs[:, i])
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)

            # Return mean of AP scores
            return np.mean(ap_scores)

    @staticmethod
    def top_k_accuracy(y_true, y_probs, k):
        """
        Calculate top-k accuracy
        """
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_preds[i]:
                correct += 1
        return (correct / len(y_true)) * 100

# Federated Learning Client
class FederatedClient:
    def __init__(self, client_id, train_data, val_data, model_name, num_classes, device):
        self.client_id = client_id
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.model_name = model_name
        self.model = ModelFactory.create_model(model_name, num_classes).to(device)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.metrics = defaultdict(list)

        # Initialize AMP (Automatic Mixed Precision) for faster training
        self.use_amp = Config.USE_AMP and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

   
    def train_epoch(self):
        """
        Train for one epoch with GPU optimization and mixed precision
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_loader = DataLoader(
            self.train_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            if self.use_amp:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Scale loss and backward
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Regular training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Clear cache periodically to prevent memory overflow
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

   
    def validate(self):
        """
        Validate model with GPU optimization
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []

        val_loader = DataLoader(
            self.val_data,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            persistent_workers=True if Config.NUM_WORKERS > 0 else False
        )

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0
        all_probs = np.vstack(all_probs) if all_probs else np.array([])

        # Clear GPU cache after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, accuracy, all_predictions, all_labels, all_probs

    
    def get_model_weights(self):
        """
        Get model weights for aggregation
        """
        # Handle DataParallel wrapper
        if isinstance(self.model, nn.DataParallel):
            return copy.deepcopy(self.model.module.state_dict())
        else:
            return copy.deepcopy(self.model.state_dict())

    
    def set_model_weights(self, weights):
        """
        Set model weights from aggregated weights
        """
        # Handle DataParallel wrapper
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights)

# Federated Learning Server
class FederatedServer:
    def __init__(self, model_name, num_classes, device):
        self.global_model = ModelFactory.create_model(model_name, num_classes).to(device)
        self.device = device
        self.round_metrics = defaultdict(list)

    
    def aggregate_weights(self, client_weights, client_sizes):
        """
        FedAvg aggregation with GPU tensor operations
        """
        total_size = sum(client_sizes)

        # Initialize aggregated weights
        aggregated_weights = {}

        # Get first client's weights as template
        for key in client_weights[0].keys():
            # Move to GPU for faster aggregation
            aggregated_weights[key] = torch.zeros_like(client_weights[0][key]).to(self.device)

        # Weighted average using GPU operations
        for i, client_weight in enumerate(client_weights):
            weight = client_sizes[i] / total_size
            for key in client_weight.keys():
                aggregated_weights[key] = aggregated_weights[key] + weight * client_weight[key].to(self.device)

        # Move back to CPU for storage
        for key in aggregated_weights.keys():
            aggregated_weights[key] = aggregated_weights[key].cpu()

        return aggregated_weights


   
    def update_global_model(self, aggregated_weights):
        """
        Update global model with aggregated weights
        """
        # Handle DataParallel wrapper
        if isinstance(self.global_model, nn.DataParallel):
            self.global_model.module.load_state_dict(aggregated_weights)
        else:
            self.global_model.load_state_dict(aggregated_weights)

    def get_global_weights(self):
        """
        Get global model weights
        """
        # Handle DataParallel wrapper
        if isinstance(self.global_model, nn.DataParallel):
            return copy.deepcopy(self.global_model.module.state_dict())
        else:
            return copy.deepcopy(self.global_model.state_dict())


# Visualization Class
class Visualizer:
    @staticmethod
    def plot_training_curves(metrics_dict, title="Training Curves"):
        """
        Plot training and validation curves including MAP
        """
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        fig.suptitle(title, fontsize=16)

        # Plot Loss
        if 'train_loss' in metrics_dict and 'val_loss' in metrics_dict:
            axes[0, 0].plot(metrics_dict['train_loss'], label='Train Loss', marker='o')
            axes[0, 0].plot(metrics_dict['val_loss'], label='Val Loss', marker='s')
            axes[0, 0].set_xlabel('Epoch/Round')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss Curve')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot Accuracy
        if 'train_acc' in metrics_dict and 'val_acc' in metrics_dict:
            axes[0, 1].plot(metrics_dict['train_acc'], label='Train Acc', marker='o')
            axes[0, 1].plot(metrics_dict['val_acc'], label='Val Acc', marker='s')
            axes[0, 1].set_xlabel('Epoch/Round')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Accuracy Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot F1 Score
        if 'f1_score' in metrics_dict:
            axes[0, 2].plot(metrics_dict['f1_score'], marker='o', color='green')
            axes[0, 2].set_xlabel('Epoch/Round')
            axes[0, 2].set_ylabel('F1 Score')
            axes[0, 2].set_title('F1 Score Curve')
            axes[0, 2].grid(True, alpha=0.3)

        # Plot MAP
        if 'map' in metrics_dict:
            axes[0, 3].plot(metrics_dict['map'], marker='o', color='purple')
            axes[0, 3].set_xlabel('Epoch/Round')
            axes[0, 3].set_ylabel('MAP')
            axes[0, 3].set_title('Mean Average Precision Curve')
            axes[0, 3].grid(True, alpha=0.3)

        # Plot AUC
        if 'auc' in metrics_dict:
            axes[1, 0].plot(metrics_dict['auc'], marker='o', color='orange')
            axes[1, 0].set_xlabel('Epoch/Round')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].set_title('AUC Curve')
            axes[1, 0].grid(True, alpha=0.3)

        # Plot Top-K Accuracies
        top_k_data = []
        for k in [1, 5, 10]:
            key = f'top_{k}_accuracy'
            if key in metrics_dict and metrics_dict[key]:
                top_k_data.append((k, metrics_dict[key][-1]))

        if top_k_data:
            k_values, acc_values = zip(*top_k_data)
            bars = axes[1, 1].bar(k_values, acc_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[1, 1].set_xlabel('K')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].set_title('Top-K Accuracy')
            axes[1, 1].set_xticks(k_values)
            axes[1, 1].grid(True, axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, val in zip(bars, acc_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                              f'{val:.1f}%', ha='center', va='bottom')

        # Plot Recall
        if 'recall' in metrics_dict:
            axes[1, 2].plot(metrics_dict['recall'], marker='o', color='red')
            axes[1, 2].set_xlabel('Epoch/Round')
            axes[1, 2].set_ylabel('Recall')
            axes[1, 2].set_title('Recall Curve')
            axes[1, 2].grid(True, alpha=0.3)

        # Summary Stats Box
        if metrics_dict:
            summary_text = "Final Metrics:\n"
            if 'val_acc' in metrics_dict and metrics_dict['val_acc']:
                summary_text += f"Acc: {metrics_dict['val_acc'][-1]:.2f}%\n"
            if 'f1_score' in metrics_dict and metrics_dict['f1_score']:
                summary_text += f"F1: {metrics_dict['f1_score'][-1]:.3f}\n"
            if 'map' in metrics_dict and metrics_dict['map']:
                summary_text += f"MAP: {metrics_dict['map'][-1]:.3f}\n"
            if 'auc' in metrics_dict and metrics_dict['auc']:
                summary_text += f"AUC: {metrics_dict['auc'][-1]:.3f}"

            axes[1, 3].text(0.5, 0.5, summary_text, ha='center', va='center',
                          fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 3].set_xlim(0, 1)
            axes[1, 3].set_ylim(0, 1)
            axes[1, 3].axis('off')
            axes[1, 3].set_title('Summary')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
        """
        Plot confusion matrix
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create annotation text with both raw counts and percentages
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'

        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14)

        return fig

    @staticmethod
    def plot_per_class_metrics(per_class_metrics, classes):
        """
        Plot per-class performance metrics including MAP
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16)

        metrics_to_plot = ['accuracy', 'f1_score', 'recall', 'auc', 'map']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            values = [per_class_metrics[cls].get(metric, 0) for cls in classes]
            bars = ax.bar(classes, values)

            # Color bars with gradient
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').upper(), fontsize=10)
            ax.set_title(f'Per-Class {metric.replace("_", " ").title()}', fontsize=12)
            ax.set_ylim([0, 1.1 if metric != 'accuracy' else 110])
            ax.grid(True, axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        # Hide the last subplot if we have odd number of metrics
        if len(metrics_to_plot) % 3 != 0:
            axes[-1, -1].axis('off')

        plt.tight_layout()
        return fig


# Main Federated Learning Pipeline
class FederatedLearningPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.results = defaultdict(lambda: defaultdict(dict))

        # Create save directory
        os.makedirs(config.SAVE_DIR, exist_ok=True)

        # Data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_data(self):
        """
        Load and prepare dataset from folder structure
        """
        print("Loading CheXpert dataset from folders...")

        # Load train dataset
        self.train_dataset = CheXpertFolderDataset(
            root_dir=self.config.TRAIN_PATH,
            classes=self.config.CLASSES,
            transform=self.train_transform,
            samples_per_class=self.config.SAMPLES_PER_CLASS
        )

        # Load test dataset
        self.test_dataset = CheXpertFolderDataset(
            root_dir=self.config.TEST_PATH,
            classes=self.config.CLASSES,
            transform=self.val_transform,
            samples_per_class=self.config.SAMPLES_PER_CLASS // 5  # Smaller test set
        )

        print(f"\nDataset Summary:")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def create_federated_datasets(self, dataset, num_clients):
        """
        Split dataset into federated clients (IID)
        """
        total_size = len(dataset)
        indices = list(range(total_size))
        np.random.shuffle(indices)

        # Split indices for each client
        split_size = total_size // num_clients
        client_indices = []

        for i in range(num_clients):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < num_clients - 1 else total_size
            client_indices.append(indices[start_idx:end_idx])

        # Create subsets for each client
        client_datasets = []
        for idx_list in client_indices:
            client_datasets.append(Subset(dataset, idx_list))

        return client_datasets

    def split_train_val(self, dataset, val_split=0.2):
        """
        Split dataset into train and validation
        """
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return train_subset, val_subset

    def create_binary_dataset(self, dataset, target_class_idx):
        """
        Create binary dataset for a specific class (one-vs-all)
        """
        binary_samples = []
        binary_targets = []

        for i in range(len(dataset)):
            _, label, _ = dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()

            # Include samples from target class as positive (1)
            if label == target_class_idx:
                binary_samples.append(i)
                binary_targets.append(1)
            # Include samples from other classes as negative (0)
            else:
                binary_samples.append(i)
                binary_targets.append(0)

        # Balance the dataset
        pos_indices = [i for i, t in enumerate(binary_targets) if t == 1]
        neg_indices = [i for i, t in enumerate(binary_targets) if t == 0]

        min_samples = min(len(pos_indices), len(neg_indices))
        balanced_indices = pos_indices[:min_samples] + neg_indices[:min_samples]
        np.random.shuffle(balanced_indices)

        # Create subset with balanced samples
        final_indices = [binary_samples[i] for i in balanced_indices]

        # Create a wrapper to return binary labels
        class BinaryWrapper(Dataset):
            def __init__(self, dataset, indices, target_class):
                self.dataset = dataset
                self.indices = indices
                self.target_class = target_class

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                image, label, multi = self.dataset[real_idx]
                # Convert to binary: 1 if target class, 0 otherwise
                binary_label = 1 if label == self.target_class else 0
                return image, binary_label, multi

        return BinaryWrapper(dataset, final_indices, target_class_idx)


    def train_per_class_federated(self, model_name, class_idx, class_name):
        """
        Train federated learning for a single class (binary classification)
        """
        print(f"\n{'='*50}")
        print(f"Training {model_name} for class: {class_name}")
        print(f"{'='*50}")

        # Create binary dataset for this class
        binary_train_dataset = self.create_binary_dataset(self.train_dataset, class_idx)
        binary_test_dataset = self.create_binary_dataset(self.test_dataset, class_idx)

        # Check if there are enough samples for training
        if len(binary_train_dataset) == 0:
            print(f"Skipping training for {class_name}: Not enough samples.")
            # Store empty metrics to avoid errors later
            self.results[model_name][class_name] = {
                'training_metrics': {},
                'test_metrics': {},
                'confusion_matrix': np.zeros((2, 2))
            }
            return {}, {}


        # Create federated datasets
        client_datasets = self.create_federated_datasets(binary_train_dataset, self.config.NUM_CLIENTS)

        # Initialize server and clients
        server = FederatedServer(model_name, 2, self.device)  # Binary classification
        clients = []

        for i, client_data in enumerate(client_datasets):
            train_data, val_data = self.split_train_val(client_data, self.config.VAL_SPLIT)
            client = FederatedClient(i, train_data, val_data, model_name, 2, self.device)
            clients.append(client)

        # Training metrics storage
        metrics = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'f1_score': [], 'auc': [], 'recall': [], 'map': [],
            'top_1_accuracy': [], 'top_5_accuracy': []
        }

        # Federated training rounds
        for round_num in range(self.config.NUM_ROUNDS):
            print(f"\nRound {round_num + 1}/{self.config.NUM_ROUNDS}")

            # Distribute global weights to clients
            global_weights = server.get_global_weights()
            for client in clients:
                client.set_model_weights(global_weights)

            # Local training
            client_weights = []
            client_sizes = []
            round_train_loss = []
            round_train_acc = []

            for client in clients:
                # Train for local epochs
                for epoch in range(self.config.LOCAL_EPOCHS):
                    loss, acc = client.train_epoch()
                    round_train_loss.append(loss)
                    round_train_acc.append(acc)

                # Get client weights
                client_weights.append(client.get_model_weights())
                client_sizes.append(len(client.train_data))

            # Aggregate weights
            aggregated_weights = server.aggregate_weights(client_weights, client_sizes)
            server.update_global_model(aggregated_weights)

            # Validation on aggregated model
            val_losses = []
            val_accs = []
            all_preds = []
            all_labels = []
            all_probs = []

            for client in clients:
                client.set_model_weights(aggregated_weights)
                val_loss, val_acc, preds, labels, probs = client.validate()
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if len(preds) > 0:
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    all_probs.append(probs)

            # Calculate metrics if we have predictions
            if all_preds:
                all_probs = np.vstack(all_probs)
                round_metrics = MetricsCalculator.calculate_metrics(
                    np.array(all_labels), np.array(all_preds), all_probs, 2
                )

                # Store metrics
                metrics['train_loss'].append(np.mean(round_train_loss))
                metrics['train_acc'].append(np.mean(round_train_acc))
                metrics['val_loss'].append(np.mean(val_losses))
                metrics['val_acc'].append(np.mean(val_accs))
                metrics['f1_score'].append(round_metrics['f1_score'])
                metrics['auc'].append(round_metrics['auc'])
                metrics['recall'].append(round_metrics['recall'])
                metrics['map'].append(round_metrics['map'])
                metrics['top_1_accuracy'].append(round_metrics['top_1_accuracy'])

                print(f"Avg Train Loss: {metrics['train_loss'][-1]:.4f}, "
                      f"Avg Train Acc: {metrics['train_acc'][-1]:.2f}%")
                print(f"Avg Val Loss: {metrics['val_loss'][-1]:.4f}, "
                      f"Avg Val Acc: {metrics['val_acc'][-1]:.2f}%")
                print(f"F1 Score: {metrics['f1_score'][-1]:.4f}, "
                      f"AUC: {metrics['auc'][-1]:.4f}, "
                      f"MAP: {metrics['map'][-1]:.4f}")

        # Test evaluation
        test_metrics = self.evaluate_on_test(server.global_model, binary_test_dataset, 2)

        # Save model
        model_path = os.path.join(self.config.SAVE_DIR, f"{model_name}_{class_name}_model.pth")
        torch.save(server.global_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Store results
        self.results[model_name][class_name] = {
            'training_metrics': metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': test_metrics['confusion_matrix']
        }

        return metrics, test_metrics

    def train_multiclass_federated(self, model_name):
        """
        Train federated learning for multi-class classification
        """
        print(f"\n{'='*50}")
        print(f"Training {model_name} for Multi-Class Classification")
        print(f"{'='*50}")

        # Create federated datasets
        client_datasets = self.create_federated_datasets(self.train_dataset, self.config.NUM_CLIENTS)

        # Initialize server and clients
        server = FederatedServer(model_name, self.config.NUM_CLASSES, self.device)
        clients = []

        for i, client_data in enumerate(client_datasets):
            train_data, val_data = self.split_train_val(client_data, self.config.VAL_SPLIT)
            client = FederatedClient(i, train_data, val_data, model_name,
                                   self.config.NUM_CLASSES, self.device)
            clients.append(client)

        # Training metrics storage
        metrics = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
            'f1_score': [], 'auc': [], 'recall': [], 'map': [],
            'top_1_accuracy': [], 'top_5_accuracy': [], 'top_10_accuracy': []
        }

        # Federated training rounds
        for round_num in range(self.config.NUM_ROUNDS):
            print(f"\nRound {round_num + 1}/{self.config.NUM_ROUNDS}")

            # Distribute global weights to clients
            global_weights = server.get_global_weights()
            for client in clients:
                client.set_model_weights(global_weights)

            # Local training
            client_weights = []
            client_sizes = []
            round_train_loss = []
            round_train_acc = []

            for client in clients:
                # Train for local epochs
                for epoch in range(self.config.LOCAL_EPOCHS):
                    loss, acc = client.train_epoch()
                    round_train_loss.append(loss)
                    round_train_acc.append(acc)

                # Get client weights
                client_weights.append(client.get_model_weights())
                client_sizes.append(len(client.train_data))

            # Aggregate weights
            aggregated_weights = server.aggregate_weights(client_weights, client_sizes)
            server.update_global_model(aggregated_weights)

            # Validation on aggregated model
            val_losses = []
            val_accs = []
            all_preds = []
            all_labels = []
            all_probs = []

            for client in clients:
                client.set_model_weights(aggregated_weights)
                val_loss, val_acc, preds, labels, probs = client.validate()
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if len(preds) > 0:
                    all_preds.extend(preds)
                    all_labels.extend(labels)
                    all_probs.append(probs)

            # Calculate metrics if we have predictions
            if all_preds:
                all_probs = np.vstack(all_probs)
                round_metrics = MetricsCalculator.calculate_metrics(
                    np.array(all_labels), np.array(all_preds), all_probs, self.config.NUM_CLASSES
                )

                # Store metrics
                metrics['train_loss'].append(np.mean(round_train_loss))
                metrics['train_acc'].append(np.mean(round_train_acc))
                metrics['val_loss'].append(np.mean(val_losses))
                metrics['val_acc'].append(np.mean(val_accs))
                metrics['f1_score'].append(round_metrics['f1_score'])
                metrics['auc'].append(round_metrics['auc'])
                metrics['recall'].append(round_metrics['recall'])
                metrics['map'].append(round_metrics['map'])
                metrics['top_1_accuracy'].append(round_metrics['top_1_accuracy'])
                metrics['top_5_accuracy'].append(round_metrics.get('top_5_accuracy', 0))

                print(f"Avg Train Loss: {metrics['train_loss'][-1]:.4f}, "
                      f"Avg Train Acc: {metrics['train_acc'][-1]:.2f}%")
                print(f"Avg Val Loss: {metrics['val_loss'][-1]:.4f}, "
                      f"Avg Val Acc: {metrics['val_acc'][-1]:.2f}%")
                print(f"F1 Score: {metrics['f1_score'][-1]:.4f}, "
                      f"AUC: {metrics['auc'][-1]:.4f}, "
                      f"MAP: {metrics['map'][-1]:.4f}")

        # Test evaluation
        test_metrics = self.evaluate_on_test(server.global_model, self.test_dataset,
                                            self.config.NUM_CLASSES)

        # Save model
        model_path = os.path.join(self.config.SAVE_DIR, f"{model_name}_multiclass_model.pth")
        torch.save(server.global_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Store results
        self.results[model_name]['multiclass'] = {
            'training_metrics': metrics,
            'test_metrics': test_metrics,
            'confusion_matrix': test_metrics['confusion_matrix']
        }

        return metrics, test_metrics

    def evaluate_on_test(self, model, test_dataset, num_classes):
        """
        Evaluate model on test dataset
        """
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())

        if all_probs:
            all_probs = np.vstack(all_probs)

            # Calculate comprehensive metrics
            test_metrics = MetricsCalculator.calculate_metrics(
                np.array(all_labels), np.array(all_preds), all_probs, num_classes
            )
            test_metrics['test_loss'] = total_loss / len(test_loader)

            print(f"\nTest Results:")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")
            print(f"Test AUC: {test_metrics['auc']:.4f}")
            print(f"Test MAP: {test_metrics['map']:.4f}")
            print(f"Test Recall: {test_metrics['recall']:.4f}")
        else:
            test_metrics = {
                'test_loss': 0, 'accuracy': 0, 'f1_score': 0,
                'auc': 0, 'map': 0, 'recall': 0,
                'confusion_matrix': np.zeros((num_classes, num_classes))
            }

        return test_metrics


    def visualize_results(self, model_name):
        """
        Create comprehensive visualizations for results
        """
        print(f"\n{'='*50}")
        print(f"Generating Visualizations for {model_name}")
        print(f"{'='*50}")

        # Create figure directory
        fig_dir = os.path.join(self.config.SAVE_DIR, f"{model_name}_figures")
        os.makedirs(fig_dir, exist_ok=True)

        # 1. Per-Class Training Curves
        for class_name in self.config.CLASSES:
            if class_name in self.results[model_name] and self.results[model_name][class_name].get('training_metrics'):
                metrics = self.results[model_name][class_name]['training_metrics']
                fig = Visualizer.plot_training_curves(
                    metrics, f"{model_name} - {class_name} Training Curves"
                )
                fig.savefig(os.path.join(fig_dir, f"{class_name}_training_curves.png"),
                           dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Confusion Matrix for class
                cm = self.results[model_name][class_name]['confusion_matrix']
                fig = Visualizer.plot_confusion_matrix(
                    cm, ['Negative', 'Positive'],
                    f"{model_name} - {class_name} Confusion Matrix"
                )
                fig.savefig(os.path.join(fig_dir, f"{class_name}_confusion_matrix.png"),
                           dpi=100, bbox_inches='tight')
                plt.close(fig)

        # 2. Multi-Class Results
        if 'multiclass' in self.results[model_name] and self.results[model_name]['multiclass'].get('training_metrics'):
            metrics = self.results[model_name]['multiclass']['training_metrics']
            fig = Visualizer.plot_training_curves(
                metrics, f"{model_name} - Multi-Class Training Curves"
            )
            fig.savefig(os.path.join(fig_dir, "multiclass_training_curves.png"),
                       dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Multi-class Confusion Matrix
            cm = self.results[model_name]['multiclass']['confusion_matrix']
            fig = Visualizer.plot_confusion_matrix(
                cm, self.config.CLASSES,
                f"{model_name} - Multi-Class Confusion Matrix"
            )
            fig.savefig(os.path.join(fig_dir, "multiclass_confusion_matrix.png"),
                       dpi=100, bbox_inches='tight')
            plt.close(fig)

        # 3. Comparative Per-Class Metrics
        per_class_metrics = {}
        for class_name in self.config.CLASSES:
            if class_name in self.results[model_name] and self.results[model_name][class_name].get('test_metrics'):
                test_metrics = self.results[model_name][class_name]['test_metrics']
                per_class_metrics[class_name] = test_metrics

        if per_class_metrics:
            # Filter out classes that were skipped due to no samples
            classes_with_metrics = [cls for cls in self.config.CLASSES if cls in per_class_metrics]
            fig = Visualizer.plot_per_class_metrics(per_class_metrics, classes_with_metrics)
            fig.savefig(os.path.join(fig_dir, "per_class_comparison.png"),
                       dpi=100, bbox_inches='tight')
            plt.close(fig)

        print(f"Visualizations saved to {fig_dir}")

    def generate_summary_report(self):
        """
        Generate comprehensive summary report with all metrics
        """
        print(f"\n{'='*60}")
        print("FEDERATED LEARNING SUMMARY REPORT")
        print(f"{'='*60}")

        summary = []
        summary.append("="*60)
        summary.append("FEDERATED LEARNING EXPERIMENT RESULTS")
        summary.append("="*60)
        summary.append(f"\nExperiment Configuration:")
        summary.append(f"  - Number of Clients: {self.config.NUM_CLIENTS}")
        summary.append(f"  - Number of Rounds: {self.config.NUM_ROUNDS}")
        summary.append(f"  - Local Epochs: {self.config.LOCAL_EPOCHS}")
        summary.append(f"  - Batch Size: {self.config.BATCH_SIZE}")
        summary.append(f"  - Learning Rate: {self.config.LEARNING_RATE}")
        summary.append(f"  - Samples per Class: {self.config.SAMPLES_PER_CLASS}")
        summary.append(f"  - Device: {self.config.DEVICE}")

        # Model comparison table
        summary.append("\n" + "="*60)
        summary.append("MODEL PERFORMANCE COMPARISON")
        summary.append("="*60)

        for model_name in self.results.keys():
            summary.append(f"\n### Model: {model_name.upper()} ###\n")

            # Per-Class Results
            summary.append("Per-Class Binary Classification Results:")
            summary.append("-" * 40)

            for class_name in self.config.CLASSES:
                if class_name in self.results[model_name] and self.results[model_name][class_name].get('test_metrics'):
                    test_metrics = self.results[model_name][class_name]['test_metrics']
                    summary.append(f"\n  {class_name}:")
                    summary.append(f"    - Accuracy:  {test_metrics['accuracy']:.2f}%")
                    summary.append(f"    - F1 Score:  {test_metrics['f1_score']:.4f}")
                    summary.append(f"    - AUC:       {test_metrics['auc']:.4f}")
                    summary.append(f"    - MAP:       {test_metrics['map']:.4f}")
                    summary.append(f"    - Recall:    {test_metrics['recall']:.4f}")
                elif class_name in self.config.CLASSES:
                     summary.append(f"\n  {class_name}:")
                     summary.append(f"    - Skipped due to insufficient samples.")


            # Multi-Class Results
            if 'multiclass' in self.results[model_name] and self.results[model_name]['multiclass'].get('test_metrics'):
                test_metrics = self.results[model_name]['multiclass']['test_metrics']
                summary.append(f"\n  Multi-Class Classification Results:")
                summary.append("  " + "-" * 38)
                summary.append(f"    - Accuracy:       {test_metrics['accuracy']:.2f}%")
                summary.append(f"    - F1 Score:       {test_metrics['f1_score']:.4f}")
                summary.append(f"    - AUC:            {test_metrics['auc']:.4f}")
                summary.append(f"    - MAP:            {test_metrics['map']:.4f}")
                summary.append(f"    - Recall:         {test_metrics['recall']:.4f}")
                summary.append(f"    - Top-1 Accuracy: {test_metrics['top_1_accuracy']:.2f}%")
                if 'top_5_accuracy' in test_metrics:
                    summary.append(f"    - Top-5 Accuracy: {test_metrics['top_5_accuracy']:.2f}%")

        # Best performing model
        summary.append("\n" + "="*60)
        summary.append("BEST PERFORMING MODELS")
        summary.append("="*60)

        # Find best model for each metric
        best_models = self._find_best_models()
        for metric, (model, value) in best_models.items():
            summary.append(f"  {metric}: {model} ({value:.4f})")

        # Print and save report
        report_text = '\n'.join(summary)
        print(report_text)

        # Save to file
        report_path = os.path.join(self.config.SAVE_DIR, "federated_learning_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)

        # Save as CSV for easier analysis
        self._save_results_csv()

        print(f"\nReport saved to {report_path}")

        return report_text


    def _find_best_models(self):
        """
        Find best performing models for each metric
        """
        best_models = {}
        metrics_to_check = ['accuracy', 'f1_score', 'auc', 'map', 'recall']

        for metric in metrics_to_check:
            best_value = -1
            best_model = None

            for model_name in self.results.keys():
                if 'multiclass' in self.results[model_name] and self.results[model_name]['multiclass'].get('test_metrics'):
                    test_metrics = self.results[model_name]['multiclass']['test_metrics']
                    if metric in test_metrics:
                        value = test_metrics[metric]
                        if metric == 'accuracy':
                            value = value / 100  # Convert percentage to decimal
                        if value > best_value:
                            best_value = value
                            best_model = model_name

            if best_model:
                if metric == 'accuracy':
                    best_value = best_value * 100  # Convert back to percentage
                best_models[metric.upper()] = (best_model, best_value)

        return best_models


    def _save_results_csv(self):
        """
        Save results as CSV for easier analysis
        """
        # Prepare data for CSV
        rows = []

        for model_name in self.results.keys():
            # Per-class results
            for class_name in self.config.CLASSES:
                if class_name in self.results[model_name] and self.results[model_name][class_name].get('test_metrics'):
                    test_metrics = self.results[model_name][class_name]['test_metrics']
                    row = {
                        'Model': model_name,
                        'Type': 'Binary',
                        'Class': class_name,
                        'Accuracy': test_metrics['accuracy'],
                        'F1_Score': test_metrics['f1_score'],
                        'AUC': test_metrics['auc'],
                        'MAP': test_metrics['map'],
                        'Recall': test_metrics['recall']
                    }
                    rows.append(row)
                elif class_name in self.config.CLASSES:
                    row = {
                        'Model': model_name,
                        'Type': 'Binary',
                        'Class': class_name,
                        'Accuracy': 'Skipped',
                        'F1_Score': 'Skipped',
                        'AUC': 'Skipped',
                        'MAP': 'Skipped',
                        'Recall': 'Skipped'
                    }
                    rows.append(row)


            # Multi-class results
            if 'multiclass' in self.results[model_name] and self.results[model_name]['multiclass'].get('test_metrics'):
                test_metrics = self.results[model_name]['multiclass']['test_metrics']
                row = {
                    'Model': model_name,
                    'Type': 'Multi-class',
                    'Class': 'All',
                    'Accuracy': test_metrics['accuracy'],
                    'F1_Score': test_metrics['f1_score'],
                    'AUC': test_metrics['auc'],
                    'MAP': test_metrics['map'],
                    'Recall': test_metrics['recall']
                }
                rows.append(row)

        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.config.SAVE_DIR, "federated_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Results CSV saved to {csv_path}")


    def run_complete_pipeline(self):
        """
        Run the complete federated learning pipeline
        """
        print(f"\n{'='*70}")
        print("STARTING FEDERATED LEARNING PIPELINE FOR CHEXPERT DATASET")
        print(f"{'='*70}")

        # Load data
        self.load_data()

        # Models to train
        models = ["resnet18", "densenet121", "efficientnet"]

        for model_name in models:
            print(f"\n\n{'#'*60}")
            print(f"TRAINING MODEL: {model_name.upper()}")
            print(f"{'#'*60}")

            # Train per-class models
            for i, class_name in enumerate(self.config.CLASSES):
                try:
                    self.train_per_class_federated(model_name, i, class_name)
                except Exception as e:
                    print(f"Error training {model_name} for {class_name}: {str(e)}")
                    # Ensure an empty entry exists in results for skipped classes
                    if class_name not in self.results[model_name]:
                         self.results[model_name][class_name] = {
                            'training_metrics': {},
                            'test_metrics': {},
                            'confusion_matrix': np.zeros((2, 2))
                        }

                    continue

            # Train multi-class model
            try:
                self.train_multiclass_federated(model_name)
            except Exception as e:
                print(f"Error training multi-class {model_name}: {str(e)}")
                continue

            # Generate visualizations
            self.visualize_results(model_name)

        # Generate summary report
        self.generate_summary_report()

        print(f"\n{'='*70}")
        print("FEDERATED LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")

# Main execution
def main():
    """
    Main function to run the federated learning pipeline
    """
    # Initialize configuration
    config = Config()

    print("Federated Learning Pipeline for CheXpert Dataset")
    print(f"Configuration:")
    print(f"  - Classes: {config.CLASSES}")
    print(f"  - Clients: {config.NUM_CLIENTS}")
    print(f"  - Rounds: {config.NUM_ROUNDS}")
    print(f"  - Device: {config.DEVICE}")

    # Initialize pipeline
    pipeline = FederatedLearningPipeline(config)

    # Run complete pipeline
    pipeline.run_complete_pipeline()

    return pipeline


if __name__ == "__main__":
    # Mount Google Drive first
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        print("Not running in Google Colab or Drive already mounted")

    # Install required packages if needed
    import subprocess
    import sys

    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    # Check and install required packages
    required_packages = ['torch', 'torchvision', 'scikit-learn', 'matplotlib', 'seaborn', 'pandas', 'numpy']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)

    # Run the pipeline
    pipeline = main()

    print("\nPipeline execution completed!")
    print(f"All models and results saved to: {Config.SAVE_DIR}")