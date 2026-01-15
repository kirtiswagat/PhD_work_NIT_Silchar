import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model Definition
class CustomXception(nn.Module):
    def __init__(self, num_classes):
        super(CustomXception, self).__init__()
        
        # Load pretrained Xception
        self.base_model = models.xception(pretrained=True)
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last 4 layers
        for param in list(self.base_model.parameters())[-4:]:
            param.requires_grad = True
        
        # Modify classifier
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# Training function with metrics
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(train_loader), predictions, true_labels

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(val_loader), predictions, true_labels

# Calculate metrics
def calculate_metrics(true_labels, predictions):
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    f2 = fbeta_score(true_labels, predictions, beta=2, average='weighted')
    
    return precision, recall, f1, f2

# Plot metrics
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

# Plot ROC curves
def plot_roc_curves(true_labels, pred_probs, num_classes):
    plt.figure(figsize=(10, 8))
    
    # Binarize labels
    true_labels_bin = label_binarize(true_labels, classes=range(num_classes))
    
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (class {i}) (area = {roc_auc:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

# Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_precision, train_recall, train_f1, train_f2 = calculate_metrics(train_labels, train_preds)
        
        # Validation
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_precision, val_recall, val_f1, val_f2 = calculate_metrics(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_precision)  # Using precision as accuracy
        val_accs.append(val_precision)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model

# Main execution
def main():
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CustomImageDataset(train_dir, transform=train_transform)
    val_dataset = CustomImageDataset(validation_dir, transform=val_transform)
    test_dataset = CustomImageDataset(test_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = CustomXception(num_classes=len(train_dataset.classes)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # Train model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)
    test_precision, test_recall, test_f1, test_f2 = calculate_metrics(test_labels, test_preds)
    
    print("\nTest Set Metrics:")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"F2 Score: {test_f2:.4f}")
    
    # Plot ROC curves for test set
    with torch.no_grad():
        all_probs = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = torch.softmax(model(inputs), dim=1)
            all_probs.extend(outputs.cpu().numpy())
    
    plot_roc_curves(test_labels, np.array(all_probs), len(train_dataset.classes))

if __name__ == "__main__":
    main()