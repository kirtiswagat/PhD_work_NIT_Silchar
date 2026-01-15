import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from tensorflow.keras.regularizers import l2

# Enhanced Parameters
img_height, img_width = 299, 299
batch_size = 16  # Reduced batch size for better generalization
num_classes = len(next(os.walk(dataset_dir))[1])

# Improved Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=(0.7, 1.3),
    fill_mode='reflect',  # Better fill mode
    validation_split=0.2  # Added validation split
)

# Validation and Test Data Generator
# Using minimal augmentation for validation and test
test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.xception.preprocess_input
)

# Training Generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Validation Generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Test Generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important for maintaining order in predictions
)

# Print dataset information
print("\nDataset Information:")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Number of classes: {num_classes}")
print("\nClass mapping:")
for class_name, class_idx in train_generator.class_indices.items():
    print(f"{class_name}: {class_idx}")



'''
# Separate validation augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.keras.applications.xception.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
'''
# Function to calculate steps per epoch
def calculate_steps_per_epoch(generator, batch_size):
    return generator.samples // batch_size + (1 if generator.samples % batch_size != 0 else 0)

# Calculate steps for each phase
train_steps = calculate_steps_per_epoch(train_generator, batch_size)
val_steps = calculate_steps_per_epoch(validation_generator, batch_size)
test_steps = calculate_steps_per_epoch(test_generator, batch_size)

print("\nSteps per epoch:")
print(f"Training steps: {train_steps}")
print(f"Validation steps: {val_steps}")
print(f"Test steps: {test_steps}")



# Model Architecture Improvements
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Progressive Layer Unfreezing
for layer in base_model.layers:
    layer.trainable = False

# Create a more robust model architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)

# First Dense Block
x = Dense(2048, kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dropout(0.4)(x)

# Second Dense Block
x = Dense(1024, kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dropout(0.4)(x)

# Third Dense Block
x = Dense(512, kernel_regularizer=l2(0.01))(x)
x = BatchNormalization()(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dropout(0.3)(x)

# Output layer
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Learning Rate Schedule
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

# Improved Optimizer
optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

def compute_class_weights(y_train):
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        y_train: Array of training labels (class indices)
        
    Returns:
        Dictionary mapping class indices to class weights
    """
    # Count samples in each class
    class_counts = np.bincount(y_train)
    
    # Calculate total number of samples
    total_samples = np.sum(class_counts)
    
    # Calculate number of classes
    n_classes = len(class_counts)
    
    # Compute weights for each class
    weights = total_samples / (n_classes * class_counts)
    
    # Normalize weights to sum to n_classes
    weights = weights * n_classes / np.sum(weights)
    
    # Create dictionary mapping class indices to weights
    class_weights = {i: weight for i, weight in enumerate(weights)}
    
    # Print class distribution and weights for verification
    print("\nClass Distribution and Weights:")
    for class_idx, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"Class {class_idx}:")
        print(f"  Count: {count}")
        print(f"  Weight: {weight:.4f}")
        print(f"  Percentage: {(count/total_samples)*100:.2f}%")
    
    return class_weights

# Compile with class weights
class_weights = compute_class_weights(train_generator.classes)  # You'll need to implement this
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Enhanced Callbacks
callbacks = [
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Training with progressive unfreezing
def train_with_unfreezing(model, epochs_per_stage=10):
    # Stage 1: Train only top layers
    print("\nStage 1: Training top layers")
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs_per_stage,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights
        verbose=1
    )
    
    # Stage 2: Unfreeze last 30 layers
    print("\nStage 2: Fine-tuning last 30 layers")
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs_per_stage,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights
        verbose=1
    )
    
    # Stage 3: Unfreeze all layers
    print("\nStage 3: Fine-tuning all layers")
    for layer in model.layers:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=1e-6),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history3 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs_per_stage,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights
        verbose=1
    )
    
    return history1, history2, history3


# Train the model with progressive unfreezing
histories = train_with_unfreezing(model)


'''
# Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')

# Calculate F1 and F2 scores
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

f1 = f1_score(y_true, y_pred_classes, average='weighted')
f2 = fbeta_score(y_true, y_pred_classes, beta=2, average='weighted')

print(f'F1 Score: {f1:.4f}')
print(f'F2 Score: {f2:.4f}')
'''
# Model evaluation function
def evaluate_model(model, test_generator, test_steps):
    """
    Evaluate the model and print detailed metrics
    """
    # Get predictions
    y_pred = model.predict(test_generator, steps=test_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    f2 = fbeta_score(y_true, y_pred_classes, beta=2, average='weighted')

    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")

    # Per-class metrics
    print("\nPer-class metrics:")
    for class_name, class_idx in test_generator.class_indices.items():
        class_precision = precision_score(y_true, y_pred_classes, labels=[class_idx], average=None)[0]
        class_recall = recall_score(y_true, y_pred_classes, labels=[class_idx], average=None)[0]
        class_f1 = f1_score(y_true, y_pred_classes, labels=[class_idx], average=None)[0]
        
        print(f"\nClass: {class_name}")
        print(f"Precision: {class_precision:.4f}")
        print(f"Recall: {class_recall:.4f}")
        print(f"F1 Score: {class_f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'y_pred': y_pred,
        'y_true': y_true
    }
# After training is complete
print("\nEvaluating model on test set...")
evaluation_results = evaluate_model(model, test_generator, test_steps)

# Get predictions for specific cases
predictions = model.predict(test_generator)

print("Evaluation Result:",evaluation_results)


import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def evaluate_model_auc(model, test_generator, num_classes):
    """
    Evaluate model performance using AUC metrics
    """
    # Predict probabilities
    y_pred_proba = model.predict(test_generator)
    y_true = test_generator.classes
    
    # Binarize the true labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Print AUC scores for each class
    print("\nAUC Scores per Class:")
    class_names = list(test_generator.class_indices.keys())
    for i in range(num_classes):
        print(f"{class_names[i]}: {roc_auc[i]:.4f}")
    
    print(f"\nMicro-average AUC: {roc_auc['micro']:.4f}")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves for each class
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, num_classes))
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save the plot
    plt.savefig('roc_curves.png')
    plt.close()
    
    return roc_auc

# Example usage:
def train_and_evaluate_model():
    # Your existing model training code here
    
    # After training, evaluate using AUC
    print("\nEvaluating model performance using AUC metrics...")
    auc_scores = evaluate_model_auc(model, test_generator, num_classes)
    
    # Create a summary table of results
    results_df = pd.DataFrame({
        'Class': list(test_generator.class_indices.keys()),
        'AUC Score': [auc_scores[i] for i in range(num_classes)]
    })
    
    print("\nResults Summary:")
    print(results_df.to_string(index=False))
    
    return auc_scores, results_df

def plot_auc_comparison(auc_scores, class_names):
    """
    Create a bar plot comparing AUC scores across different categories
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    categories = range(len(class_names))
    plt.bar(categories, [auc_scores[i] for i in range(len(class_names))], 
            color='skyblue', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate([auc_scores[i] for i in range(len(class_names))]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Customize plot
    plt.xlabel('Different Categories of Sound')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores Across Different Sound Categories')
    plt.xticks(categories, class_names, rotation=45)
    plt.ylim(0.5, 1.0)  # Set y-axis range from 0.5 to 1.0
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.3, label='Fair (0.7)')
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='Good (0.8)')
    plt.axhline(y=0.9, color='b', linestyle='--', alpha=0.3, label='Excellent (0.9)')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('auc_comparison.png')
    plt.close()

# Example usage:
def analyze_model_performance():
    # Train and evaluate model
    auc_scores, results_df = train_and_evaluate_model()
    
    # Plot AUC comparison
    plot_auc_comparison(auc_scores, list(test_generator.class_indices.keys()))
    
    # Calculate and print additional statistics
    mean_auc = np.mean([auc_scores[i] for i in range(len(test_generator.class_indices))])
    std_auc = np.std([auc_scores[i] for i in range(len(test_generator.class_indices))])
    
    print("\nOverall AUC Statistics:")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Standard Deviation: {std_auc:.4f}")
    print(f"95% Confidence Interval: [{mean_auc - 1.96*std_auc:.4f}, {mean_auc + 1.96*std_auc:.4f}]")


import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

def get_metrics_from_roc(y_true, y_pred_proba, class_names):
    """
    Extract Recall, Precision, and Accuracy from ROC curve data
    using optimal threshold selection.
    """
    metrics_per_class = {}
    
    # Binarize the labels for multi-class
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    for i, class_name in enumerate(class_names):
        # Get ROC curve points
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        
        # Get Precision-Recall curve points
        precision, recall, pr_thresholds = precision_recall_curve(y_true_bin[:, i], 
                                                                y_pred_proba[:, i])
        
        # Calculate optimal threshold using Youden's J statistic
        # J = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate metrics at optimal threshold
        y_pred_class = (y_pred_proba[:, i] >= optimal_threshold).astype(int)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        TP = np.sum((y_pred_class == 1) & (y_true_bin[:, i] == 1))
        FP = np.sum((y_pred_class == 1) & (y_true_bin[:, i] == 0))
        TN = np.sum((y_pred_class == 0) & (y_true_bin[:, i] == 0))
        FN = np.sum((y_pred_class == 0) & (y_true_bin[:, i] == 1))
        
        # Calculate metrics
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[class_name] = {
            'Recall/Sensitivity': recall,
            'Precision': precision,
            'Accuracy': accuracy,
            'F1-Score': f1_score,
            'Optimal Threshold': optimal_threshold,
            'AUC': auc(fpr, tpr)
        }
    
    return metrics_per_class

def print_metrics_table(metrics_dict):
    """
    Print metrics in a formatted table
    """
    # Print header
    print("\nMetrics for each class at optimal threshold:")
    print("-" * 80)
    print(f"{'Class':<15} {'Recall':>10} {'Precision':>10} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>10}")
    print("-" * 80)
    
    # Print metrics for each class
    for class_name, metrics in metrics_dict.items():
        print(f"{class_name:<15} "
              f"{metrics['Recall/Sensitivity']:>10.3f} "
              f"{metrics['Precision']:>10.3f} "
              f"{metrics['Accuracy']:>10.3f} "
              f"{metrics['F1-Score']:>10.3f} "
              f"{metrics['AUC']:>10.3f}")
    print("-" * 80)

# Example usage
def analyze_model_metrics(model, test_generator):
    # Get predictions
    y_pred_proba = model.predict(test_generator)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    
    # Calculate metrics
    metrics = get_metrics_from_roc(y_true, y_pred_proba, class_names)
    
    # Print results
    print_metrics_table(metrics)
    
    # Calculate and print average metrics
    avg_metrics = {
        'Recall/Sensitivity': np.mean([m['Recall/Sensitivity'] for m in metrics.values()]),
        'Precision': np.mean([m['Precision'] for m in metrics.values()]),
        'Accuracy': np.mean([m['Accuracy'] for m in metrics.values()]),
        'F1-Score': np.mean([m['F1-Score'] for m in metrics.values()]),
        'AUC': np.mean([m['AUC'] for m in metrics.values()])
    }
    
    print("\nAverage Metrics Across All Classes:")
    print(f"Average Recall: {avg_metrics['Recall/Sensitivity']:.3f}")
    print(f"Average Precision: {avg_metrics['Precision']:.3f}")
    print(f"Average Accuracy: {avg_metrics['Accuracy']:.3f}")
    print(f"Average F1-Score: {avg_metrics['F1-Score']:.3f}")
    print(f"Average AUC: {avg_metrics['AUC']:.3f}")
    
    return metrics, avg_metrics

# After training your model
metrics, avg_metrics = analyze_model_metrics(model, test_generator)

# To look at specific class metrics
print(f"Metrics for class C-1:", metrics['C-1'])

'''



'''