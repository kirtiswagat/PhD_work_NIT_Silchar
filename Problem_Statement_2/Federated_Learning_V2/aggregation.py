"""
Aggregation of Binary-Class Models - ResNet18 Only
==================================================

Loads each class-specific binary ResNet18 checkpoint, averages the backbone
weights (FedAvg across diseases), and builds a **single five-output
multi-label classifier**.

The aggregated model performs multi-class classification across all pathologies
using only ResNet18 architecture.
"""

import os, copy, json
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils_updated import (Config, CheXpertFolder, ModelFactory, seed_everything, 
                          clear_gpu, Metrics)

# Configuration - Using only ResNet18
AGGREGATION_EPOCHS = 5  # Number of fine-tuning epochs for aggregated model

seed_everything()

def load_and_aggregate_models():
    """
    Load binary classification checkpoints and aggregate backbone weights
    """
    print("="*60)
    print("LOADING BINARY CLASSIFICATION MODELS - ResNet18")
    print("="*60)
    
    state_dicts = []
    successful_diseases = []
    
    # Load checkpoints for each disease
    for disease in Config.CLASSES_PATHOLOGY:
        ckpt_path = os.path.join(Config.SAVE_DIR, disease, f"{Config.MODEL_NAME}.pth")
        
        if not os.path.isfile(ckpt_path):
            print(f"⚠ Missing checkpoint for {disease}: {ckpt_path}")
            continue
        
        print(f"✓ Loading {disease} model...")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dicts.append(state_dict)
        successful_diseases.append(disease)
    
    if not state_dicts:
        raise FileNotFoundError("No binary classification checkpoints found!")
    
    print(f"Successfully loaded {len(state_dicts)} models: {successful_diseases}")
    
    # FedAvg aggregation of backbone weights
    print("\nAggregating backbone weights using FedAvg...")
    avg_state = {}
    
    # Get all keys from first model
    all_keys = set(state_dicts[0].keys())
    
    # Find backbone keys (exclude final classification layers)
    backbone_keys = []
    head_keys = []
    
    for key in all_keys:
        if key.startswith(("fc.", "classifier.", "module.fc.", "module.classifier.")):
            head_keys.append(key)
        else:
            backbone_keys.append(key)
    
    print(f"Backbone parameters: {len(backbone_keys)}")
    print(f"Head parameters (will be reinitialized): {len(head_keys)}")
    
    # Average backbone weights
    for key in backbone_keys:
        # Check if all models have this key
        if all(key in sd for sd in state_dicts):
            avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
        else:
            print(f"⚠ Key {key} not found in all models, skipping...")
    
    print(f"✓ Aggregated {len(avg_state)} backbone parameters")
    return avg_state, successful_diseases

def create_aggregated_model(avg_backbone_weights):
    """
    Create multi-class model with aggregated backbone weights
    """
    print("\nCreating aggregated multi-class ResNet18 model...")
    
    # Create model with 5 outputs (for 5 pathologies)
    model = ModelFactory.create(Config.MODEL_NAME, len(Config.CLASSES_PATHOLOGY), pretrained=False)
    
    # Load current state dict
    current_state = model.state_dict()
    
    # Update backbone weights with aggregated weights
    for key in avg_backbone_weights:
        if key in current_state:
            current_state[key] = avg_backbone_weights[key]
    
    # Load the updated state dict
    model.load_state_dict(current_state)
    model = model.to(Config.DEVICE)
    
    print(f"✓ Created ResNet18 model with aggregated backbone weights")
    return model

def evaluate_model(model, test_ds):
    """Evaluate the aggregated model on test data"""
    print("\n--- Evaluating Aggregated ResNet18 Model ---")
    
    test_dl = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, 
                        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y, _ in test_dl:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(torch.softmax(outputs, 1).cpu().numpy())
    
    if all_preds:
        all_probs = np.vstack(all_probs)
        test_metrics = Metrics.compute(np.array(all_labels), np.array(all_preds), 
                                     all_probs, len(Config.CLASSES_PATHOLOGY))
        test_metrics["test_loss"] = test_loss / len(test_dl)
        
        print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1: {test_metrics['f1_score']:.4f}")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test mAP: {test_metrics['map']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        
        return test_metrics
    else:
        return {}

def fine_tune_aggregated_model(model):
    """
    Fine-tune the aggregated model on multi-class data (optional)
    """
    print(f"\nFine-tuning aggregated ResNet18 model for {AGGREGATION_EPOCHS} epochs...")
    
    # Prepare datasets
    train_tr = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tr = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load multi-class datasets
    train_ds = CheXpertFolder(Config.TRAIN_PATH, Config.CLASSES_PATHOLOGY, 
                             train_tr, Config.SAMPLES_PER_CLS)
    val_ds = CheXpertFolder(Config.TEST_PATH, Config.CLASSES_PATHOLOGY, 
                           val_tr, Config.SAMPLES_PER_CLS // 5)
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("⚠ No multi-class data available, skipping fine-tuning")
        return {}, val_ds
    
    # Data loaders
    train_dl = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, 
                         num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_dl = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, 
                       num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
    
    training_history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "f1_score": [], "auc": [], "map": [], "recall": [],
        "top_1_accuracy": [], "top_5_accuracy": []
    }
    
    # Fine-tuning loop
    for epoch in range(AGGREGATION_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y, _ in train_dl:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_dl)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for x, y, _ in val_dl:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.append(torch.softmax(outputs, 1).cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dl)
        
        # Calculate comprehensive metrics
        if all_preds:
            all_probs = np.vstack(all_probs)
            metrics = Metrics.compute(np.array(all_labels), np.array(all_preds), 
                                    all_probs, len(Config.CLASSES_PATHOLOGY))
            
            # Store metrics
            training_history["train_loss"].append(avg_train_loss)
            training_history["train_acc"].append(train_acc)
            training_history["val_loss"].append(avg_val_loss)
            training_history["val_acc"].append(metrics["accuracy"])
            training_history["f1_score"].append(metrics["f1_score"])
            training_history["auc"].append(metrics["auc"])
            training_history["map"].append(metrics["map"])
            training_history["recall"].append(metrics["recall"])
            training_history["top_1_accuracy"].append(metrics.get("top_1_accuracy", 0))
            training_history["top_5_accuracy"].append(metrics.get("top_5_accuracy", 0))
            
            print(f"Epoch {epoch+1}/{AGGREGATION_EPOCHS}:")
            print(f"  Train: Loss {avg_train_loss:.4f}, Acc {train_acc:.2f}%")
            print(f"  Val: Loss {avg_val_loss:.4f}, Acc {metrics['accuracy']:.2f}%")
            print(f"  Metrics: F1 {metrics['f1_score']:.3f}, AUC {metrics['auc']:.3f}, mAP {metrics['map']:.3f}")
        
        clear_gpu()
    
    return training_history, val_ds

def save_aggregated_model(model, training_history, test_metrics, successful_diseases):
    """
    Save the final aggregated model and results
    """
    print("\nSaving aggregated ResNet18 model...")
    
    # Create aggregated results directory
    agg_dir = os.path.join(Config.SAVE_DIR, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(agg_dir, f"{Config.MODEL_NAME}_aggregated_multilabel.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save training history
    if training_history:
        history_path = os.path.join(agg_dir, f"{Config.MODEL_NAME}_aggregation_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")
    
    # Save test metrics
    if test_metrics:
        test_path = os.path.join(agg_dir, f"{Config.MODEL_NAME}_test_results.json")
        test_results = {k: v.tolist() if isinstance(v, np.ndarray) else float(v) 
                       for k, v in test_metrics.items() 
                       if k != "confusion_matrix"}
        if "confusion_matrix" in test_metrics:
            test_results["confusion_matrix"] = test_metrics["confusion_matrix"].tolist()
        
        with open(test_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"✓ Test results saved to: {test_path}")
    
    # Save aggregation info
    info = {
        "backbone": Config.MODEL_NAME,
        "aggregated_diseases": successful_diseases,
        "num_models_aggregated": len(successful_diseases),
        "output_classes": Config.CLASSES_PATHOLOGY,
        "aggregation_epochs": AGGREGATION_EPOCHS,
        "final_test_accuracy": test_metrics.get("accuracy", 0) if test_metrics else 0,
        "final_test_map": test_metrics.get("map", 0) if test_metrics else 0
    }
    
    info_path = os.path.join(agg_dir, "aggregation_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"✓ Aggregation info saved to: {info_path}")
    
    return model_path

def main():
    """
    Main aggregation pipeline
    """
    print("="*70)
    print("FEDERATED MODEL AGGREGATION - ResNet18 Only")
    print("="*70)
    print(f"Backbone: {Config.MODEL_NAME}")
    print(f"Target classes: {Config.CLASSES_PATHOLOGY}")
    print(f"Save directory: {Config.SAVE_DIR}")
    print("="*70)
    
    try:
        # Step 1: Load and aggregate binary models
        avg_backbone_weights, successful_diseases = load_and_aggregate_models()
        
        # Step 2: Create aggregated multi-class model
        aggregated_model = create_aggregated_model(avg_backbone_weights)
        
        # Step 3: Fine-tune aggregated model (optional)
        training_history, test_ds = fine_tune_aggregated_model(aggregated_model)
        
        # Step 4: Final test evaluation
        test_metrics = evaluate_model(aggregated_model, test_ds)
        
        # Step 5: Save final model
        final_model_path = save_aggregated_model(aggregated_model, training_history, 
                                               test_metrics, successful_diseases)
        
        print("\n" + "="*70)
        print("AGGREGATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"✅ Final aggregated ResNet18 model: {final_model_path}")
        print(f"✅ Aggregated {len(successful_diseases)} disease models")
        print(f"✅ Model can classify: {Config.CLASSES_PATHOLOGY}")
        
        if test_metrics:
            final_acc = test_metrics.get("accuracy", 0)
            final_map = test_metrics.get("map", 0)
            print(f"✅ Final test performance: Acc {final_acc:.2f}%, mAP {final_map:.3f}")
        
        if training_history and "val_acc" in training_history and training_history["val_acc"]:
            final_train_acc = training_history["val_acc"][-1]
            final_train_map = training_history["map"][-1]
            print(f"✅ Final fine-tuning performance: Acc {final_train_acc:.2f}%, mAP {final_train_map:.3f}")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during aggregation: {str(e)}")
        raise

if __name__ == "__main__":
    # Mount Google Drive (if in Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        print("Not running in Google Colab or Drive already mounted")
    
    main()
