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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import math

class LRFinderWithROC(Callback):
    def __init__(self, validation_data, start_lr=1e-7, end_lr=10, n_steps=100, beta=0.98, num_classes=5):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.n_steps = n_steps
        self.beta = beta
        self.validation_data = validation_data
        self.num_classes = num_classes
        
        # Storage for metrics
        self.lrs = []
        self.losses = []
        self.smoothed_losses = []
        self.roc_data = {}  # Store ROC data for different LR checkpoints
        self.lr_checkpoints = []  # Store specific LRs where we compute ROC
        
        self.step_size = (end_lr - start_lr) / n_steps
        self.best_loss = float('inf')
        
    def on_train_begin(self, logs={}):
        self.step = 0
        K = tf.keras.backend
        K.set_value(self.model.optimizer.learning_rate, self.start_lr)
        
        # Define LR checkpoints for ROC analysis (e.g., every order of magnitude)
        self.lr_checkpoints = np.logspace(
            np.log10(self.start_lr), 
            np.log10(self.end_lr), 
            5  # Number of checkpoints
        )
        
    def compute_roc(self, learning_rate):
        """Compute ROC curves for current model state"""
        # Get predictions for validation data
        y_pred = self.model.predict(self.validation_data)
        y_true = self.validation_data.classes
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
        
    def on_batch_end(self, batch, logs={}):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        loss = logs.get('loss')
        
        self.lrs.append(lr)
        self.losses.append(loss)
        
        # Update smoothed loss
        if self.step == 0:
            self.smoothed_losses.append(loss)
        else:
            smooth_loss = self.beta * self.smoothed_losses[-1] + (1-self.beta) * loss
            self.smoothed_losses.append(smooth_loss)
        
        # Check if current LR is close to any checkpoint
        for checkpoint_lr in self.lr_checkpoints:
            if abs(lr - checkpoint_lr) < self.step_size/2:
                print(f"\nComputing ROC for learning rate: {lr:.2e}")
                self.roc_data[lr] = self.compute_roc(lr)
        
        # Stop if loss explodes
        if self.step > 0 and loss > 4 * self.smoothed_losses[-2]:
            self.model.stop_training = True
            return
            
        # Update learning rate
        if self.step < self.n_steps:
            lr += self.step_size
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            
        self.step += 1

    def plot_loss(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.lrs[10:-5], self.smoothed_losses[10:-5])
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs Loss')
        plt.grid(True)
        plt.show()
        
    def plot_roc_curves(self):
        """Plot ROC curves for different learning rates"""
        n_classes = self.num_classes
        n_lrs = len(self.roc_data)
        
        # Create subplots for each class
        fig, axes = plt.subplots(n_classes, 1, figsize=(15, 5*n_classes))
        fig.suptitle('ROC Curves for Different Learning Rates by Class')
        
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            
            for lr, roc_data in self.roc_data.items():
                fpr = roc_data['fpr'][class_idx]
                tpr = roc_data['tpr'][class_idx]
                roc_auc = roc_data['roc_auc'][class_idx]
                
                ax.plot(
                    fpr, 
                    tpr, 
                    label=f'LR={lr:.2e} (AUC={roc_auc:.2f})'
                )
                
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Class {class_idx}')
            ax.legend(loc="lower right")
            ax.grid(True)
            
        plt.tight_layout()
        plt.show()
        
    def suggest_lr(self):
        # Find the learning rate with minimum loss
        min_grad_idx = np.gradient(np.array(self.smoothed_losses[10:])).argmin() + 10
        return self.lrs[min_grad_idx]

# Your existing setup code remains the same
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

# Define paths
base_dir = '/home/user/GPU_CSE/Kirti/Dataset_for_Code_Development/Dataset_for_5_classes_by_Sir/XceptionNet'
dataset_dir = '/home/user/GPU_CSE/Kirti/Dataset_for_Code_Development/Dataset_for_5_classes_by_Sir/clean_train_partition'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Parameters
img_height, img_width = 299, 299
batch_size = 32
num_classes = len(next(os.walk(dataset_dir))[1])

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.8, 1.2)
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    
    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(2048, activation='relu', name='Dense_1024')(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(1024, activation='relu', name='Dense_512')(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(512, activation='relu', name='Dense_64')(top_model)
    predictions = Dense(num_classes, activation='softmax', name='Output_layer')(top_model)
    
    return Model(inputs=base_model.input, outputs=predictions)

# First, find the optimal learning rate with ROC analysis
print("Finding optimal learning rate and computing ROC curves...")
model = create_model()
model.compile(optimizer=Adam(learning_rate=1e-7),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Create and use the LR finder with ROC analysis
lr_finder = LRFinderWithROC(
    validation_data=validation_generator,
    start_lr=1e-7,
    end_lr=1,
    n_steps=100,
    num_classes=num_classes
)

# Run the learning rate finder
history = model.fit(
    train_generator,
    steps_per_epoch=50,  # Limit steps for LR finder
    epochs=1,
    callbacks=[lr_finder],
    verbose=1
)

# Plot the results
print("\nPlotting Learning Rate vs Loss curve...")
lr_finder.plot_loss()

print("\nPlotting ROC curves for different learning rates...")
lr_finder.plot_roc_curves()

suggested_lr = lr_finder.suggest_lr()
print(f"\nSuggested learning rate: {suggested_lr:.2e}")

# Train the final model with the found learning rate
print("\nTraining final model with optimal learning rate...")
model = create_model()
model.compile(optimizer=Adam(learning_rate=suggested_lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for the actual training
checkpoint = ModelCheckpoint('Architecture_no_7.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the optimal learning rate
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)

# Plot final training history
epochs = range(len(history.history['accuracy']))
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(epochs, history.history['accuracy'], 'go-', label='Training Accuracy')
ax[0].plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
ax[0].set_title('Training and Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(epochs, history.history['loss'], 'go-', label='Training Loss')
ax[1].plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss')
ax[1].set_title('Training and Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')
ax[1].legend()

plt.show()