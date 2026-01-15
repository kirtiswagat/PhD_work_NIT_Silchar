from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)
from tqdm.notebook import tqdm # Import tqdm for progress bars

import tensorflow as tf
import os

print("TensorFlow Version: ", tf.__version__)
print("CUDA built: ", tf.test.is_built_with_cuda())
print("Available GPUs: ")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"  {gpu}")
    print("GPU is available and being used by TensorFlow.")
else:
    print("No GPU devices found. TensorFlow is running on CPU.")


# =====================================================
# 1. Dual Modal Encoder
# =====================================================
class DualModalEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, dropout_rate=0.3):
        super().__init__()

        self.xray_encoder = EfficientNetB0(
            include_top=False, weights='imagenet',
            input_shape=input_shape, pooling='avg'
        )

        self.ct_encoder = EfficientNetB0(
            include_top=False, weights='imagenet',
            input_shape=input_shape, pooling='avg'
        )

        self.xray_feat = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu')
        ])

        self.ct_feat = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu')
        ])

        self.fusion = tf.keras.Sequential([
            layers.Concatenate(),
            layers.Dense(feature_dim * 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(feature_dim, activation='relu')
        ])

    def call(self, inputs, training=None):
        xray, ct = inputs
        xray_f = self.xray_feat(self.xray_encoder(xray, training=training))
        ct_f = self.ct_feat(self.ct_encoder(ct, training=training))
        fused = self.fusion([xray_f, ct_f])
        return fused

# =====================================================
# 2. Build Multi-Task Model
# =====================================================
def build_classifier(encoder, num_disease_classes, num_cancer_classes):
    xray_in = layers.Input(shape=(224,224,3))
    ct_in = layers.Input(shape=(224,224,3))

    features = encoder([xray_in, ct_in])

    # disease_out now configured for single-label classification (e.g., GNN data)
    disease_out = layers.Dense(
        num_disease_classes, activation='softmax', name='disease_out'
    )(features)

    cancer_out = layers.Dense(
        num_cancer_classes, activation='softmax', name='cancer_out'
    )(features)

    return Model([xray_in, ct_in], [disease_out, cancer_out])

# =====================================================
# 3. Image Loader
# =====================================================
def load_and_preprocess_image(path, image_size=(224,224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    return img / 255.0

# =====================================================
# 4. Dataset Creation (General for multi-label CheXpert-like data)
# =====================================================
def create_multi_label_tf_dataset_from_folder(
    base_dir, class_names, batch_size=16, max_samples_per_class=None
):
    all_paths, all_labels = [], []
    label_map = {c: i for i, c in enumerate(class_names)}

    print(f"Scanning files in {base_dir}...")
    # Use tqdm to show progress during file scanning
    for root, _, files in tqdm(os.walk(base_dir), desc=f"Scanning {base_dir}"):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, base_dir).split(os.sep)

                vec = np.zeros(len(class_names))
                has_relevant_label = False
                for p in rel:
                    if p in label_map:
                        vec[label_map[p]] = 1
                        has_relevant_label = True
                if not has_relevant_label: # If no relevant labels found, skip
                    continue

                all_paths.append(full)
                all_labels.append(vec)
    print(f"Found {len(all_paths)} files in {base_dir}.")

    # --- Implement class-wise sampling if max_samples_per_class is provided ---
    paths = all_paths
    labels = all_labels
    if max_samples_per_class is not None:
        selected_paths_set = set()
        class_positive_counts = {i: 0 for i in range(len(class_names))}

        combined = list(zip(all_paths, all_labels))
        np.random.shuffle(combined)

        final_paths = []
        final_labels = []

        for path, label_vec in combined:
            add_file = False
            for i, val in enumerate(label_vec):
                if val == 1 and class_positive_counts[i] < max_samples_per_class:
                    add_file = True
                    break

            if add_file and path not in selected_paths_set:
                final_paths.append(path)
                final_labels.append(label_vec)
                selected_paths_set.add(path)

                for i, val in enumerate(label_vec):
                    if val == 1:
                        class_positive_counts[i] += 1

            # Check if all classes have reached their limit (or if all files have been processed)
            if all(count >= max_samples_per_class for count in class_positive_counts.values()) or len(selected_paths_set) == len(all_paths):
                break

        paths = final_paths
        labels = final_labels
        print(f"After sampling, using {len(paths)} files for {base_dir}.")
        for i, class_name in enumerate(class_names):
            print(f"  Class '{class_name}': {class_positive_counts[i]} positive samples.")

    def gen():
        for p,l in tqdm(zip(paths, labels), total=len(paths), desc="Loading multi-label images"):
            yield load_and_preprocess_image(p), l

    output_signature_label = tf.TensorSpec((len(class_names),), tf.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((224,224,3), tf.float32),
            output_signature_label
        )
    )

    return ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# =====================================================
# Helper for Single-Label TF Dataset Creation (General)
# =====================================================
def _create_single_label_tf_dataset_from_paths_labels(filepaths, labels, batch_size=32):
    print(f"Preparing single-label TF dataset from {len(filepaths)} files...")
    def gen():
        for p,l in tqdm(zip(filepaths, labels), total=len(filepaths), desc="Loading images"):
            yield load_and_preprocess_image(p), l

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((224,224,3), tf.float32),
            tf.TensorSpec((), tf.int32) # Single integer label
        )
    )
    # Ensure shuffle works even with empty datasets
    # Added `drop_remainder=True` to ensure all batches have uniform size
    return ds.shuffle(len(filepaths) if len(filepaths) > 0 else 1).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def create_paired_dataset(xray_ds, ct_ds):
    return tf.data.Dataset.zip((xray_ds, ct_ds)).map(
        lambda x, y: ((x[0], y[0]), (x[1], y[1])) # Changed from [x[0], y[0]] to (x[0], y[0])
    )

# =====================================================
# 5. CLASS-BALANCED LOSS (ITEM-3)
# =====================================================
def compute_class_weights(y):
    pos = np.sum(y, axis=0)
    neg = y.shape[0] - pos
    return neg / (pos + 1e-6)


class WeightedBinaryCE(tf.keras.losses.Loss):
    def __init__(self, weights):
        super().__init__()
        self.w = tf.constant(weights, tf.float32)

    def call(self, y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * self.w + (1 - y_true)
        return tf.reduce_mean(bce * weight)


# =====================================================
# 6. TRAINING
# =====================================================
def train_model(model, train_ds, val_ds):
    # Now, disease_out is also single-label, so we use sparse_categorical_crossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            'disease_out': 'sparse_categorical_crossentropy',
            'cancer_out': 'sparse_categorical_crossentropy'
        },
        metrics={
            'disease_out': ['accuracy'], # AUC is less relevant for single-label classification with sparse labels
            'cancer_out': ['accuracy']
        }
    )
    # Calculate how many batches are in one full pass
    steps = len(gnn_train_paths) // batch_size
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

# =====================================================
# 7. EVALUATION (ITEM-4 & 5)
# =====================================================
def evaluate_disease(model, dataset, class_names):
    y_true_labels = [] # To store integer true labels for disease
    y_pred_probs = []  # To store predicted probabilities for disease

    # Loop through the paired dataset
    for (xray_batch, ct_batch), (disease_labels_batch, cancer_labels_batch) in tqdm(dataset, desc="Evaluating disease predictions"):
        # Predict both outputs from the model
        disease_preds, cancer_preds = model.predict([xray_batch, ct_batch])

        y_true_labels.append(disease_labels_batch.numpy())
        y_pred_probs.append(disease_preds)

    # Concatenate all batches
    y_true_labels = np.concatenate(y_true_labels, axis=0) # Shape: (total_samples,)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)   # Shape: (total_samples, num_disease_classes)

    # For single-label multi-class classification, get the predicted class index
    y_pred_classes = np.argmax(y_pred_probs, axis=1) # Shape: (total_samples,)

    print("\nDisease Classification Metrics:")
    print("Classification Report:")
    # target_names are derived from gnn_classes
    print(classification_report(y_true_labels, y_pred_classes, target_names=class_names, zero_division=0))

    print("\nF1 Scores (Disease):")
    print("Micro:", f1_score(y_true_labels, y_pred_classes, average='micro', zero_division=0))
    print("Macro:", f1_score(y_true_labels, y_pred_classes, average='macro', zero_division=0))
    print("Weighted:", f1_score(y_true_labels, y_pred_classes, average='weighted', zero_division=0))

    print("\nPer-Class AUC (Disease, One-vs-Rest):")
    # Convert y_true_labels to one-hot encoding for AUC calculation
    num_disease_classes = len(class_names)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_labels, num_classes=num_disease_classes)

    if y_true_one_hot.shape[1] != y_pred_probs.shape[1]:
        # Handle cases where `to_categorical` might produce fewer columns if a class is entirely absent
        # (e.g., if a class is not present in the validation set, to_categorical might produce fewer columns).
        # Ensure y_true_one_hot has the full `num_disease_classes` columns.
        temp_y_true_one_hot = np.zeros((y_true_labels.shape[0], num_disease_classes))
        for i, val in enumerate(y_true_labels):
            if val < num_disease_classes: # Ensure index is within bounds
                temp_y_true_one_hot[i, val] = 1
        y_true_one_hot = temp_y_true_one_hot

    if y_true_one_hot.shape[1] == y_pred_probs.shape[1]:
        for i, c in enumerate(class_names):
            if len(np.unique(y_true_one_hot[:, i])) > 1: # Check if there are both positive and negative samples for this class
                try:
                    auc_score = roc_auc_score(y_true_one_hot[:, i], y_pred_probs[:, i])
                    print(f"  {c}: {auc_score:.4f}")
                except ValueError as e:
                    print(f"  {c}: AUC error - {e}")
            else:
                print(f"  {c}: AUC not applicable (only one class present in true labels or prediction for this class)")
    else:
        print("Warning: Cannot compute per-class AUC due to class count mismatch between true labels (one-hot) and predicted probabilities.")

    print("\nOptimal Thresholds (Not directly applicable for single-label multiclass with softmax output. Showing F1 score per class based on predicted label):")
    for i, c in enumerate(class_names):
        # Calculate F1 score for each class by treating it as a binary (one vs rest) problem
        # using the overall predicted classes
        y_true_binary = (y_true_labels == i).astype(int)
        y_pred_binary = (y_pred_classes == i).astype(int)
        class_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        print(f"  {c}: F1 score for class {i}: {class_f1:.4f}")

    return None # No specific thresholds to return for single-label multiclass


# =====================================================
# Helper function to load and split single-label data by class folder
# =====================================================
def _load_single_label_class_data(base_dir, train_count_per_class, val_count_per_class):
    # Filter out .ipynb_checkpoints and other hidden directories
    all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    class_names = sorted([d for d in all_subdirs if not d.startswith('.')])  # Exclude hidden directories
    label_map = {c: i for i, c in enumerate(class_names)}

    all_train_paths, all_train_labels = [], []
    all_val_paths, all_val_labels = [], []

    print(f"\nLoading and splitting data from {base_dir}...")
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(base_dir, class_name)
        images_in_class = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        np.random.shuffle(images_in_class)

        num_train = min(train_count_per_class, len(images_in_class))
        remaining_for_val = len(images_in_class) - num_train
        num_val = min(val_count_per_class, remaining_for_val)

        train_files = images_in_class[:num_train]
        val_files = images_in_class[num_train: num_train + num_val]

        label_id = label_map[class_name]
        all_train_paths.extend(train_files)
        all_train_labels.extend([label_id] * len(train_files))
        all_val_paths.extend(val_files)
        all_val_labels.extend([label_id] * len(val_files))

        print(
            f"  Class '{class_name}': {len(train_files)} train, {len(val_files)} val (out of {len(images_in_class)} total).")

    print(f"Total files for GNN dataset: {len(all_train_paths)} train, {len(all_val_paths)} val.")
    return (all_train_paths, all_train_labels), (all_val_paths, all_val_labels), class_names
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Plot Disease Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['disease_out_loss'], label='Disease Training Loss')
    plt.plot(history.history['val_disease_out_loss'], label='Disease Validation Loss')
    plt.title('Disease Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Cancer Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['cancer_out_loss'], label='Cancer Training Loss')
    plt.plot(history.history['val_cancer_out_loss'], label='Cancer Validation Loss')
    plt.title('Cancer Output Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))

    # Plot Disease Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['disease_out_accuracy'], label='Disease Training Accuracy')
    plt.plot(history.history['val_disease_out_accuracy'], label='Disease Validation Accuracy')
    plt.title('Disease Output Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Cancer Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['cancer_out_accuracy'], label='Cancer Training Accuracy')
    plt.plot(history.history['val_cancer_out_accuracy'], label='Cancer Validation Accuracy')
    plt.title('Cancer Output Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plotting AUC-ROC Curves for Disease Classification
from sklearn.metrics import roc_curve, auc

def plot_multiclass_roc(y_true_one_hot, y_pred_probs, class_names):
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Disease Output')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Displaying Confusion Matrix for Disease Classification
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true_labels, y_pred_classes, class_names):
    cm = confusion_matrix(y_true_labels, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Disease Output')
    plt.show()



# =====================================================
# 8. MAIN
# =====================================================
def main():
    # Define classes for the new GNN dataset (will be inferred from folders)
    gnn_data_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    ct_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset'

    # --- GNN Dataset Loading (replacing X-ray part) ---
    (gnn_train_paths, gnn_train_labels), (gnn_val_paths, gnn_val_labels), gnn_classes = \
        _load_single_label_class_data(gnn_data_dir, train_count_per_class=100, val_count_per_class=50)

    gnn_train_ds = _create_single_label_tf_dataset_from_paths_labels(gnn_train_paths, gnn_train_labels)
    gnn_val_ds = _create_single_label_tf_dataset_from_paths_labels(gnn_val_paths, gnn_val_labels)

    # No class weights needed for GNN data if using sparse_categorical_crossentropy

    # --- CT Dataset Loading ---
    ct_classes = ['Bengin', 'Malignant', 'Normal']  # Corrected 'benign' to 'Bengin'
    ct_files_all, ct_labels_all = [], []
    collected_ct_counts = {c: 0 for c in ct_classes}  # To track counts after sampling per class
    print("\nCollecting CT files and applying per-class sampling...")
    for i, c in enumerate(ct_classes):
        class_specific_files = [os.path.join(ct_dir, c, f) for f in os.listdir(os.path.join(ct_dir, c)) if
                                f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Randomly sample up to 500 files for this class
        if len(class_specific_files) > 500:
            sampled_files = np.random.choice(class_specific_files, 500, replace=False)
        else:
            sampled_files = class_specific_files

        ct_files_all.extend(sampled_files)
        ct_labels_all.extend([i] * len(sampled_files))
        collected_ct_counts[c] = len(sampled_files)
        print(f"  Collected {len(sampled_files)} files for CT class '{c}'.")

    ct_files, ct_labels = np.array(ct_files_all), np.array(ct_labels_all)
    print(f"Total CT files for training/validation after initial sampling: {len(ct_files)}")
    print("CT dataset distribution after initial sampling:")
    for c_name, count in collected_ct_counts.items():
        print(f"  Class '{c_name}': {count} samples.")

    tr_f, va_f, tr_l, va_l = train_test_split(
        ct_files, ct_labels, stratify=ct_labels, test_size=0.3, random_state=42
        # Added random_state for reproducibility
    )
    print(f"\nCT Train Set: {len(tr_f)} files, CT Validation Set: {len(va_f)} files.")

    ct_train_ds = _create_single_label_tf_dataset_from_paths_labels(tr_f, tr_l)
    ct_val_ds = _create_single_label_tf_dataset_from_paths_labels(va_f, va_l)

    # --- Create Paired Dataset ---
    train_ds = create_paired_dataset(gnn_train_ds, ct_train_ds).repeat()  # Add .repeat() for training dataset
    val_ds = create_paired_dataset(gnn_val_ds, ct_val_ds)

    # --- Build and Train Model ---
    model = build_classifier(
        DualModalEncoder(),
        len(gnn_classes),  # num_disease_classes now from GNN classes
        len(ct_classes)
    )

    history = train_model(
        model,
        train_ds,
        val_ds
    )

    # Call plotting function for training history
    plot_training_history(history)

    # Note: evaluate_disease is still designed for multi-label (sigmoid output).
    # For the GNN data, which is now single-label, these metrics might not be ideal.
    # You might want to update evaluate_disease for single-label classification metrics if needed.
    # For demonstration, we'll pass the GNN class names to it, but interpret results carefully.

    # Capture evaluation results for plotting
    eval_results = evaluate_disease(model, val_ds, gnn_classes)

    # Retrieve necessary data for plotting AUC-ROC and Confusion Matrix
    # Note: evaluate_disease now returns y_true_labels, y_pred_probs, y_pred_classes, y_true_one_hot
    # I'll need to modify evaluate_disease to return these values explicitly
    # For now, let's assume they are globally accessible or refactor evaluate_disease.

    # To make these available, I will refactor evaluate_disease to return the necessary items
    # and then call these plotting functions here. For this step, I'll update main.

    # Re-running evaluation to capture outputs for plotting
    y_true_labels = []  # To store integer true labels for disease
    y_pred_probs = []  # To store predicted probabilities for disease

    # Loop through the paired dataset for evaluation data
    for (xray_batch, ct_batch), (disease_labels_batch, cancer_labels_batch) in tqdm(val_ds,
                                                                                    desc="Collecting evaluation data for plotting"):
        disease_preds, _ = model.predict([xray_batch, ct_batch])
        y_true_labels.append(disease_labels_batch.numpy())
        y_pred_probs.append(disease_preds)

    y_true_labels = np.concatenate(y_true_labels, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    num_disease_classes = len(gnn_classes)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_labels, num_classes=num_disease_classes)

    # Call plotting functions
    plot_multiclass_roc(y_true_one_hot, y_pred_probs, gnn_classes)
    plot_confusion_matrix(y_true_labels, y_pred_classes, gnn_classes)


# =====================================================
# Helper function to load and split single-label data by class folder
# =====================================================
def _load_single_label_class_data(base_dir, train_count_per_class, val_count_per_class):
    # Filter out .ipynb_checkpoints and other hidden directories
    all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    class_names = sorted([d for d in all_subdirs if not d.startswith('.')]) # Exclude hidden directories
    label_map = {c: i for i, c in enumerate(class_names)}

    all_train_paths, all_train_labels = [], []
    all_val_paths, all_val_labels = [], []

    print(f"\nLoading and splitting data from {base_dir}...")
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(base_dir, class_name)
        images_in_class = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        np.random.shuffle(images_in_class)

        num_train = min(train_count_per_class, len(images_in_class))
        remaining_for_val = len(images_in_class) - num_train
        num_val = min(val_count_per_class, remaining_for_val)

        train_files = images_in_class[:num_train]
        val_files = images_in_class[num_train : num_train + num_val]

        label_id = label_map[class_name]
        all_train_paths.extend(train_files)
        all_train_labels.extend([label_id] * len(train_files))
        all_val_paths.extend(val_files)
        all_val_labels.extend([label_id] * len(val_files))

        print(f"  Class '{class_name}': {len(train_files)} train, {len(val_files)} val (out of {len(images_in_class)} total).")

    print(f"Total files for GNN dataset: {len(all_train_paths)} train, {len(all_val_paths)} val.")
    return (all_train_paths, all_train_labels), (all_val_paths, all_val_labels), class_names


# =====================================================
# Helper function to load and split single-label data by class folder
# =====================================================
def _load_single_label_class_data(base_dir, train_count_per_class, val_count_per_class):
    # Filter out .ipynb_checkpoints and other hidden directories
    all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    class_names = sorted([d for d in all_subdirs if not d.startswith('.')])  # Exclude hidden directories
    label_map = {c: i for i, c in enumerate(class_names)}

    all_train_paths, all_train_labels = [], []
    all_val_paths, all_val_labels = [], []

    print(f"\nLoading and splitting data from {base_dir}...")
    for class_name in tqdm(class_names, desc="Processing classes"):
        class_path = os.path.join(base_dir, class_name)
        images_in_class = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        np.random.shuffle(images_in_class)

        num_train = min(train_count_per_class, len(images_in_class))
        remaining_for_val = len(images_in_class) - num_train
        num_val = min(val_count_per_class, remaining_for_val)

        train_files = images_in_class[:num_train]
        val_files = images_in_class[num_train: num_train + num_val]

        label_id = label_map[class_name]
        all_train_paths.extend(train_files)
        all_train_labels.extend([label_id] * len(train_files))
        all_val_paths.extend(val_files)
        all_val_labels.extend([label_id] * len(val_files))

        print(
            f"  Class '{class_name}': {len(train_files)} train, {len(val_files)} val (out of {len(images_in_class)} total).")

    print(f"Total files for GNN dataset: {len(all_train_paths)} train, {len(all_val_paths)} val.")
    return (all_train_paths, all_train_labels), (all_val_paths, all_val_labels), class_names


# =====================================================
# 8. MAIN
# =====================================================
def main():
    # Define classes for the new GNN dataset (will be inferred from folders)
    gnn_data_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    ct_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset'

    # --- GNN Dataset Loading (replacing X-ray part) ---
    (gnn_train_paths, gnn_train_labels), (gnn_val_paths, gnn_val_labels), gnn_classes = \
        _load_single_label_class_data(gnn_data_dir, train_count_per_class=100, val_count_per_class=50)

    gnn_train_ds = _create_single_label_tf_dataset_from_paths_labels(gnn_train_paths, gnn_train_labels)
    gnn_val_ds = _create_single_label_tf_dataset_from_paths_labels(gnn_val_paths, gnn_val_labels)

    # No class weights needed for GNN data if using sparse_categorical_crossentropy

    # --- CT Dataset Loading ---
    ct_classes = ['Bengin', 'Malignant', 'Normal']  # Corrected 'benign' to 'Bengin'
    ct_files_all, ct_labels_all = [], []
    collected_ct_counts = {c: 0 for c in ct_classes}  # To track counts after sampling per class
    print("\nCollecting CT files and applying per-class sampling...")
    for i, c in enumerate(ct_classes):
        class_specific_files = [os.path.join(ct_dir, c, f) for f in os.listdir(os.path.join(ct_dir, c)) if
                                f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Randomly sample up to 500 files for this class
        if len(class_specific_files) > 500:
            sampled_files = np.random.choice(class_specific_files, 500, replace=False)
        else:
            sampled_files = class_specific_files

        ct_files_all.extend(sampled_files)
        ct_labels_all.extend([i] * len(sampled_files))
        collected_ct_counts[c] = len(sampled_files)
        print(f"  Collected {len(sampled_files)} files for CT class '{c}'.")

    ct_files, ct_labels = np.array(ct_files_all), np.array(ct_labels_all)
    print(f"Total CT files for training/validation after initial sampling: {len(ct_files)}")
    print("CT dataset distribution after initial sampling:")
    for c_name, count in collected_ct_counts.items():
        print(f"  Class '{c_name}': {count} samples.")

    tr_f, va_f, tr_l, va_l = train_test_split(
        ct_files, ct_labels, stratify=ct_labels, test_size=0.3, random_state=42
        # Added random_state for reproducibility
    )
    print(f"\nCT Train Set: {len(tr_f)} files, CT Validation Set: {len(va_f)} files.")

    ct_train_ds = _create_single_label_tf_dataset_from_paths_labels(tr_f, tr_l)
    ct_val_ds = _create_single_label_tf_dataset_from_paths_labels(va_f, va_l)

    # --- Create Paired Dataset ---
    train_ds = create_paired_dataset(gnn_train_ds, ct_train_ds).repeat()  # Add .repeat() for training dataset
    val_ds = create_paired_dataset(gnn_val_ds, ct_val_ds)

    # --- Build and Train Model ---
    model = build_classifier(
        DualModalEncoder(),
        len(gnn_classes),  # num_disease_classes now from GNN classes
        len(ct_classes)
    )

    history = train_model(
        model,
        train_ds,
        val_ds
    )

    # Call plotting function for training history
    plot_training_history(history)

    # Note: evaluate_disease is still designed for multi-label (sigmoid output).
    # For the GNN data, which is now single-label, these metrics might not be ideal.
    # You might want to update evaluate_disease for single-label classification metrics if needed.
    # For demonstration, we'll pass the GNN class names to it, but interpret results carefully.

    # Capture evaluation results for plotting
    eval_results = evaluate_disease(model, val_ds, gnn_classes)

    # Retrieve necessary data for plotting AUC-ROC and Confusion Matrix
    # Note: evaluate_disease now returns y_true_labels, y_pred_probs, y_pred_classes, y_true_one_hot
    # I'll need to modify evaluate_disease to return these values explicitly
    # For now, let's assume they are globally accessible or refactor evaluate_disease.

    # To make these available, I will refactor evaluate_disease to return the necessary items
    # and then call these plotting functions here. For this step, I'll update main.

    # Re-running evaluation to capture outputs for plotting
    y_true_labels = []  # To store integer true labels for disease
    y_pred_probs = []  # To store predicted probabilities for disease

    # Loop through the paired dataset for evaluation data
    for (xray_batch, ct_batch), (disease_labels_batch, cancer_labels_batch) in tqdm(val_ds,
                                                                                    desc="Collecting evaluation data for plotting"):
        disease_preds, _ = model.predict([xray_batch, ct_batch])
        y_true_labels.append(disease_labels_batch.numpy())
        y_pred_probs.append(disease_preds)

    y_true_labels = np.concatenate(y_true_labels, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    num_disease_classes = len(gnn_classes)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_labels, num_classes=num_disease_classes)

    # Call plotting functions
    plot_multiclass_roc(y_true_one_hot, y_pred_probs, gnn_classes)
    plot_confusion_matrix(y_true_labels, y_pred_classes, gnn_classes)

if __name__ == "__main__":
  main()