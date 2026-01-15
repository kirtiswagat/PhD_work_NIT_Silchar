####################################
#🔹 STEP 0: Mount Google Drive
####################################

from google.colab import drive

# 1. SETUP & DRIVE MOUNTING
drive.mount('/content/drive')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow detected the following GPUs: {gpus}")
    print("Training should be using GPU.")
else:
    print("No GPU devices found. Training will use CPU.")

####################################
#📦 STEP 1 — Imports
####################################

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib.ticker import MultipleLocator
import time
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sns
####################################
#📁 STEP 2 — Paths (YOUR PATHS)
####################################

train_csv = "/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/train.csv"
val_csv   = "/content/drive/MyDrive/Kaggel_direct_download/CheXpert-v1.0-small/valid.csv"

train_img_root = "/content/drive/MyDrive/Kaggel_direct_download"
val_img_root   = "/content/drive/MyDrive/Kaggel_direct_download"

###########################################
#📁 STEP 3 — Disease Labels (Lung-Focused)
###########################################

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Lung Opacity",
    "Pneumonia",
    "Pneumothorax"
]

NUM_DISEASES = len(DISEASES)
IMG_SIZE = 224
EMBED_DIM = 256


###########################################
# 📁 STEP 4 — CSV Loader
###########################################

def load_chexpert_csv(csv_path, img_root, num_images_per_class=None):
    df = pd.read_csv(csv_path)

    # Filter for '1' or '0' labels, ignoring -1 (uncertain) and -2 (not mentioned)
    # And handle 'U-zero' (uncertain as zero) and 'U-one' (uncertain as one)

    # Convert -1 to 0 (uncertain to negative) for binary classification consistency
    df = df.replace(-1, 0)

    if num_images_per_class is not None:
        sampled_dfs = []
        for disease in DISEASES:
            # Get positive examples for the current disease
            positive_examples = df[df[disease] == 1]
            # Sample up to num_images_per_class, or take all if fewer exist
            if len(positive_examples) > num_images_per_class:
                sampled_dfs.append(positive_examples.sample(n=num_images_per_class, random_state=42))
            else:
                sampled_dfs.append(positive_examples)

            # Get negative examples for the current disease
            negative_examples = df[df[disease] == 0]
            # Sample an equal number of negative examples (if available) to maintain balance
            if len(negative_examples) > num_images_per_class:
                sampled_dfs.append(negative_examples.sample(n=num_images_per_class, random_state=42))
            else:
                sampled_dfs.append(negative_examples)

        # Combine and remove duplicates, as some images might be positive/negative for multiple diseases
        df = pd.concat(sampled_dfs).drop_duplicates().reset_index(drop=True)

    image_paths = df["Path"].apply(lambda x: os.path.join(img_root, x)).values
    labels = df[DISEASES].fillna(-2).values.astype(np.float32)

    # Convert -1 back to 0 for consistency if not handled earlier, or adjust as per dataset interpretation
    # Here, after sampling, if any -1s were still present, this would convert them.
    labels[labels == -1] = 0

    return image_paths, labels


###########################################
# 📁 STEP 5 — Image Loader
###########################################

def load_image(path):
    path = path.decode('utf-8')  # Decode path from bytes to string

    if not os.path.exists(path):
        # tf.print(f"Warning: Image file not found at path: {path}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    img = cv2.imread(path)
    if img is None:
        # tf.print(f"Warning: cv2.imread failed to load image at path: {path}. Returning black image.")
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.astype(np.float32)

###########################################
#📁 STEP 6 — tf.data Pipeline
###########################################



def build_dataset(image_paths, labels, batch_size, shuffle=False):
    def _parse(img_path, label):
        img = tf.numpy_function(load_image, [img_path], tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###########################################
#📁 STEP 7 — Load Dataset
###########################################


train_imgs, train_labels = load_chexpert_csv(train_csv, train_img_root, num_images_per_class=1000)
val_imgs, val_labels     = load_chexpert_csv(val_csv, val_img_root, num_images_per_class=1000)

train_ds = build_dataset(train_imgs, train_labels, batch_size=16, shuffle=True)
val_ds   = build_dataset(val_imgs, val_labels, batch_size=16)

###########################################
#📁 STEP 8 — Disease Graph (Adjacency)
###########################################

def build_adjacency():
    adj = np.zeros((NUM_DISEASES, NUM_DISEASES), dtype=np.float32)

    edges = [
        (5, 6), (5, 2), (6, 2),
        (3, 4), (0, 5),
        (7, 5), (1, 3)
    ]

    for i, j in edges:
        adj[i, j] = adj[j, i] = 1

    adj += np.eye(NUM_DISEASES)
    return tf.constant(adj, dtype=tf.float32)

ADJ = build_adjacency()

###########################################
#📁 STEP 9 — Disease Graph GAT
###########################################
class DiseaseGraphGAT(layers.Layer):
    def __init__(self, n, d):
        super().__init__()
        self.emb = self.add_weight((n, d), initializer="glorot_uniform", trainable=True)
        self.attn = self.add_weight((2*d, 1), initializer="glorot_uniform", trainable=True)

    def call(self, adj):
        n = tf.shape(self.emb)[0]
        hi = tf.tile(self.emb[:, None, :], [1, n, 1])
        hj = tf.tile(self.emb[None, :, :], [n, 1, 1])

        e = tf.nn.leaky_relu(tf.matmul(tf.concat([hi, hj], -1), self.attn))[:, :, 0]
        e = e + tf.where(adj > 0, 0.0, -1e9)

        alpha = tf.nn.softmax(e, axis=1)
        return tf.matmul(alpha, self.emb)

###########################################
#📁 STEP 10 — Image Encoder
###########################################

def build_image_encoder():
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(EMBED_DIM, activation="relu")(x)
    return models.Model(base.input, x)

###########################################
#📁 STEP 11 — Full Model
###########################################


def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    img_feat = build_image_encoder()(inp)

    disease_emb = DiseaseGraphGAT(NUM_DISEASES, EMBED_DIM)(ADJ)
    # Wrap tf.matmul in a Lambda layer to correctly integrate it into the Keras Functional API
    logits = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([img_feat, disease_emb])
    out = layers.Activation('sigmoid')(logits)

    return models.Model(inp, out)

model = build_model()

model.summary()

###########################################
# 📁 STEP 12 — CheXpert Masking + HBCE
###########################################



def chexpert_mask(y):
    mask = tf.cast(y >= 0, tf.float32)
    y = tf.clip_by_value(y, 0, 1)
    return y, mask


HIERARCHY = [(5, 6), (5, 2), (6, 2), (3, 4)]


class HBCE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y, mask = chexpert_mask(y_true)

        # Use K.binary_crossentropy for explicit element-wise loss calculation
        # This ensures bce has shape (batch_size, NUM_DISEASES)
        bce = K.binary_crossentropy(y, y_pred)

        bce = tf.reduce_sum(bce * mask) / (tf.reduce_sum(mask) + 1e-7)  # Add small epsilon to prevent division by zero

        penalty = 0
        for p, c in HIERARCHY:
            penalty += tf.reduce_mean(tf.nn.relu(y_pred[:, c] - y_pred[:, p]))

        return bce + 0.3 * penalty

###########################################
#📁 STEP 13 — Compile
###########################################

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=HBCE(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(
            name="auc",
            multi_label=True,
            num_labels=NUM_DISEASES
        )
    ]
)

###########################################
#📁 STEP 14 — Train
###########################################

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")

#################################
# Predictions
#################################
y_true, y_pred = [], []

for x,y in val_ds:
    y_true.append(y.numpy())
    y_pred.append(model.predict(x))

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

###########################################
# Evaluation on Training and Validation
# Loss & Accuracy
##########################################

def plot_training_curves(history):
    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(18, 5))

    # 🔹 Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # 🔹 Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # 🔹 AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history.history["auc"], label="Train AUC")
    plt.plot(epochs, history.history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Training vs Validation AUC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_training_curves(history)

###############################################
# Disease wise Confusion Matrix
###############################################

def plot_confusion_matrices(y_true, y_pred, disease_names, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)

    for i, disease in enumerate(disease_names):
        valid = y_true[:, i] >= 0
        if np.sum(valid) == 0:
            continue

        cm = confusion_matrix(
            y_true[valid, i],
            y_pred_bin[valid, i]
        )

        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"]
        )
        plt.title(f"Normalized Confusion Matrix – {disease}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()


plot_confusion_matrices(y_true, y_pred, DISEASES)

################################################
# Training and Validation AUC
################################################

plt.plot(history.history["auc"], label="train AUC")
plt.plot(history.history["val_auc"], label="val AUC")
plt.legend(); plt.grid(); plt.show()


#####################################################
# AUC-ROC Curve
#####################################################

def plot_roc_curves_fixed(y_true, y_pred, disease_names):
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(disease_names):
        valid = y_true[:, i] >= 0
        if np.sum(valid) == 0:
            continue

        fpr, tpr, _ = roc_curve(
            y_true[valid, i],
            y_pred[valid, i]
        )
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            linewidth=2,
            label=f"{name} (AUC={roc_auc:.2f})"
        )

    # Diagonal reference
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    # Axis limits
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # Labels
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves (CheXpert Lung Diseases)", fontsize=14)

    # Major ticks
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))

    # Minor ticks
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.05))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(0.05))

    # Grid
    plt.grid(which="major", linestyle="-", linewidth=0.7, alpha=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.6)

    # Axis spines (corners)
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(True)
        plt.gca().spines[spine].set_linewidth(1.2)

    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.show()


plot_roc_curves_fixed(y_true, y_pred, DISEASES)

###############################################
# Classification Report with Fixed Thresholding
# value = 0.5
###############################################

for i,d in enumerate(DISEASES):
    m = y_true[:,i] >= 0
    print(d)
    print(classification_report(
        y_true[m,i], (y_pred[m,i] > 0.5).astype(int),
        zero_division=0
    ))


#######################################
# Hierarchy Violation Calculation
#######################################


def hierarchy_violation(y_pred):
    v=[]
    for p,c in HIERARCHY:
        v.append(np.mean(y_pred[:,c] > y_pred[:,p]))
    return np.mean(v)

print("Hierarchy Violation Rate:", hierarchy_violation(y_pred))
