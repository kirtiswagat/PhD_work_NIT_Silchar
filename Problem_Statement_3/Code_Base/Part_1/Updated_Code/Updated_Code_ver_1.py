# ============================================================
# PHASE-1: UNPAIRED X-RAY + CT (CORRECT BASELINE)
# ============================================================

# -------------------------------
# 1. IMPORTS
# -------------------------------
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize


# -------------------------------
# 2. FOCAL LOSSES
# -------------------------------
class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        loss = -y_true * tf.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred) \
               - (1 - y_true) * tf.pow(y_pred, self.gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        loss = -y_true * tf.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# -------------------------------
# 3. IMAGE LOADER
# -------------------------------
def load_img(path):
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        return img / 255.0
    except Exception as e:
        tf.print(f"ERROR loading image {path}: {e}")
        # Return a placeholder tensor of zeros if an image fails to load
        return tf.zeros((224, 224, 3), dtype=tf.float32)


# -------------------------------
# 4. DATASET BUILDERS
# -------------------------------
def build_xray_dataset(img_paths, labels, batch=16):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(lambda x, y: (load_img(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_ct_dataset(img_paths, labels, num_classes, batch=16):
    labels = tf.one_hot(labels, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(lambda x, y: (load_img(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# -------------------------------
# 5. MODEL (PHASE-1 CORRECT)
# -------------------------------
def build_phase1_model(num_xray_labels, num_ct_classes):
    encoder = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    x_in = layers.Input((224, 224, 3), name="xray_input")
    c_in = layers.Input((224, 224, 3), name="ct_input")

    x_feat = encoder(x_in)
    c_feat = encoder(c_in)

    x_feat = layers.Dense(512, activation="relu")(x_feat)
    c_feat = layers.Dense(512, activation="relu")(c_feat)

    x_out = layers.Dense(
        num_xray_labels,
        activation="sigmoid",
        name="xray_out"
    )(x_feat)

    c_out = layers.Dense(
        num_ct_classes,
        activation="softmax",
        name="ct_out"
    )(c_feat)

    return Model(inputs=[x_in, c_in], outputs=[x_out, c_out])

# -------------------------------
# 6. EVALUATION UTILITIES
# -------------------------------
def plot_pr_multilabel(y_true, y_pred, class_names, title):
    plt.figure(figsize=(7,6))
    for i, cls in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        ap = average_precision_score(y_true[:, i], y_pred[:, i])
        plt.plot(r, p, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_pr_multiclass(
    y_true,
    y_pred,
    class_names,
    title="Multi-Class Precision–Recall Curve",
    show_micro=False
):
    """
    y_true: shape (N,) integer labels
    y_pred: shape (N, C) softmax probabilities
    """

    n_classes = len(class_names)

    # One-hot encode ground truth
    y_true_oh = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(7,6))

    # Per-class PR curves
    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_oh[:, i], y_pred[:, i]
        )
        ap = average_precision_score(
            y_true_oh[:, i], y_pred[:, i]
        )
        plt.plot(
            recall, precision,
            label=f"{cls} (AP={ap:.2f})"
        )

    # Optional micro-average
    if show_micro:
        precision_micro, recall_micro, _ = precision_recall_curve(
            y_true_oh.ravel(), y_pred.ravel()
        )
        ap_micro = average_precision_score(
            y_true_oh, y_pred, average="micro"
        )
        plt.plot(
            recall_micro,
            precision_micro,
            linestyle="--",
            linewidth=2,
            label=f"Micro-average (AP={ap_micro:.2f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def diagnosis_report_multilabel(y_true, y_pred, class_names):
    y_pred_bin = (y_pred > 0.5).astype(int)
    report = classification_report(
        y_true, y_pred_bin,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    print("Report for Multi Label")
    print(report)
    return pd.DataFrame(report).transpose()[["precision","recall","f1-score","support"]]


def diagnosis_report_multiclass(y_true, y_pred, class_names):
    y_pred_cls = np.argmax(y_pred, axis=1)
    report = classification_report(
        y_true, y_pred_cls,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    print("Report for Multi Class")
    print(report)
    return pd.DataFrame(report).transpose()[["precision","recall","f1-score","support"]]


def plot_accuracy(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['xray_out_accuracy'], label="Train")
    plt.plot(history.history['val_xray_out_accuracy'], label="Val")
    plt.title("X-ray Accuracy")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history.history['ct_out_accuracy'], label="Train")
    plt.plot(history.history['val_ct_out_accuracy'], label="Val")
    plt.title("CT Accuracy")
    plt.legend(); plt.grid(True)

    plt.show()


def plot_loss(history):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['xray_out_loss'], label="Train")
    plt.plot(history.history['val_xray_out_loss'], label="Val")
    plt.title("X-ray Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history.history['ct_out_loss'], label="Train")
    plt.plot(history.history['val_ct_out_loss'], label="Val")
    plt.title("CT Loss")
    plt.legend(); plt.grid(True)

    plt.show()


# -------------------------------
# 7. DATA LOADING HELPERS
# -------------------------------
def load_xray_data(root_dir, class_names, max_per_class=1000):
    imgs, labels = [], []
    print("\n📂 Loading X-ray data")

    for i, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        files = files[:max_per_class]

        print(f"  ▶ {cls}: {len(files)} images")

        for f in tqdm(files, desc=f"    Loading {cls}", leave=False):
            imgs.append(os.path.join(cls_dir, f))
            y = np.zeros(len(class_names))
            y[i] = 1
            labels.append(y)

    print(f"✅ Total X-ray images loaded: {len(imgs)}")
    return np.array(imgs), np.array(labels)


def load_ct_data(root_dir, class_names, max_per_class=700):
    imgs, labels = [], []
    print("\n📂 Loading CT data")

    for i, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        files = files[:max_per_class]

        print(f"  ▶ {cls}: {len(files)} images")

        for f in tqdm(files, desc=f"    Loading {cls}", leave=False):
            imgs.append(os.path.join(cls_dir, f))
            labels.append(i)

    print(f"✅ Total CT images loaded: {len(imgs)}")
    return np.array(imgs), np.array(labels)


class EpochLogger(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\n📌 Epoch {epoch+1}: "
            f"X-ray Acc={logs['xray_out_accuracy']:.3f}, "
            f"CT Acc={logs['ct_out_accuracy']:.3f}, "
            f"Val X-ray Acc={logs['val_xray_out_accuracy']:.3f}, "
            f"Val CT Acc={logs['val_ct_out_accuracy']:.3f}"
        )

def load_and_prepare_data():
    X_RAY_DIR = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    CT_DIR    = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset/Lung_Cancer_LIDC_Y_Net/LIDC_Y-Net'

    xray_class_names = ['Pleural_Effusion', 'Edema', 'Lung_Opacity']
    ct_class_names   = ['Benign', 'Malignant', 'Normal']

    # Load raw data
    xray_imgs, xray_labels = load_xray_data(X_RAY_DIR, xray_class_names)
    ct_imgs, ct_labels     = load_ct_data(CT_DIR, ct_class_names)

    # Split
    x_tr, x_va, yx_tr, yx_va = train_test_split(
        xray_imgs, xray_labels, test_size=0.2, random_state=42
    )
    c_tr, c_va, yc_tr, yc_va = train_test_split(
        ct_imgs, ct_labels, test_size=0.2, random_state=42
    )

    print("\n📊 Dataset Summary")
    print(f"  X-ray Train: {len(x_tr)} | Val: {len(x_va)}")
    print(f"  CT    Train: {len(c_tr)} | Val: {len(c_va)}")

    # Build tf.data datasets
    train_xray_ds = build_xray_dataset(x_tr, yx_tr)
    val_xray_ds   = build_xray_dataset(x_va, yx_va)

    train_ct_ds = build_ct_dataset(c_tr, yc_tr, len(ct_class_names))
    val_ct_ds   = build_ct_dataset(c_va, yc_va, len(ct_class_names))

    train_ds = tf.data.Dataset.zip((train_xray_ds, train_ct_ds))
    val_ds   = tf.data.Dataset.zip((val_xray_ds, val_ct_ds))

    train_ds = train_ds.map(
        lambda x, c: (
            {"xray_input": x[0], "ct_input": c[0]},
            {"xray_out": x[1], "ct_out": c[1]}
        )
    )

    val_ds = val_ds.map(
        lambda x, c: (
            {"xray_input": x[0], "ct_input": c[0]},
            {"xray_out": x[1], "ct_out": c[1]}
        )
    )

    return train_ds, val_ds, xray_class_names, ct_class_names


def build_and_compile_model(xray_class_names, ct_class_names):
    model = build_phase1_model(
        len(xray_class_names),
        len(ct_class_names)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            "xray_out": BinaryFocalLoss(),
            "ct_out": CategoricalFocalLoss()
        },
        metrics={
            "xray_out": "accuracy",
            "ct_out": "accuracy"
        }
    )

    model.summary()
    return model


def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
            EpochLogger()
        ]
    )
    return history


def run_validation(model, val_ds):
    print("\n🔍 Running validation & prediction")

    xray_true, xray_pred = [], []
    ct_true, ct_pred = [], []

    val_steps = tf.data.experimental.cardinality(val_ds).numpy()

    for inputs, targets in tqdm(val_ds, desc="Validating", total=val_steps):

        # ✅ CALL MODEL DIRECTLY (NOT model.predict)
        preds = model(
            {
                "xray_input": inputs["xray_input"],
                "ct_input": inputs["ct_input"]
            },
            training=False
        )

        xray_pred.append(preds[0].numpy())
        ct_pred.append(preds[1].numpy())

        xray_true.append(targets["xray_out"].numpy())
        ct_true.append(targets["ct_out"].numpy())

    print("✅ Validation complete")

    xray_true = np.vstack(xray_true)
    xray_pred = np.vstack(xray_pred)

    ct_true = np.argmax(np.vstack(ct_true), axis=1)
    ct_pred = np.vstack(ct_pred)

    return xray_true, xray_pred, ct_true, ct_pred


##########################################################
##  ✅ X-ray Confusion Matrices (Multi-label)############
##########################################################



def plot_xray_confusion_matrices(y_true, y_pred, class_names, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(int)

    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 4))

    if n_classes == 1:
        axes = [axes]

    for i, cls in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Neg", "Pos"],
            yticklabels=["Neg", "Pos"],
            ax=axes[i]
        )
        axes[i].set_title(f"X-ray Confusion Matrix\n{cls}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    plt.tight_layout()
    plt.show()


##########################################################
##  ✅ CT Confusion Matrix (Multi-class)#################
##########################################################

def plot_ct_confusion_matrix(y_true, y_pred, class_names):
    y_pred_cls = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred_cls)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("CT (Cancer) Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


###################################################
#####✅ Accuracy-Augmented Diagnosis Reports######
#####         X-Ray                           #####
###################################################
def diagnosis_report_multilabel_with_accuracy(y_true, y_pred, class_names, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(int)

    report = classification_report(
        y_true, y_pred_bin,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report).transpose()

    # Add overall accuracy (mean label accuracy)
    acc = (y_true == y_pred_bin).mean()
    df.loc["accuracy"] = [acc, acc, acc, y_true.shape[0]]

    return df[["precision", "recall", "f1-score", "support"]]


###################################################
#####✅ Accuracy-Augmented Diagnosis Reports######
#####         CT                           ########
###################################################
def diagnosis_report_multiclass_with_accuracy(y_true, y_pred, class_names):
    y_pred_cls = np.argmax(y_pred, axis=1)

    report = classification_report(
        y_true, y_pred_cls,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report).transpose()

    acc = accuracy_score(y_true, y_pred_cls)
    df.loc["accuracy"] = [acc, acc, acc, len(y_true)]

    return df[["precision", "recall", "f1-score", "support"]]


def run_reports_and_plots(
    xray_true, xray_pred,
    ct_true, ct_pred,
    xray_class_names, ct_class_names,
    history
):

    # =======================
    # X-RAY REPORT
    # =======================
    print("\nX-RAY DIAGNOSIS REPORT (WITH ACCURACY)")
    display(
        diagnosis_report_multilabel_with_accuracy(
            xray_true, xray_pred, xray_class_names
        )
    )

    plot_xray_confusion_matrices(
        xray_true, xray_pred, xray_class_names
    )

    plot_pr_multilabel(
        xray_true, xray_pred,
        xray_class_names,
        "X-ray Precision–Recall Curve"
    )

    plot_pr_multiclass(
              ct_true,
              ct_pred,
              ct_class_names,
              title="CT (Cancer) Precision–Recall Curve"
          )


    # =======================
    # CT REPORT
    # =======================
    print("\nCT (CANCER) DIAGNOSIS REPORT (WITH ACCURACY)")
    display(
        diagnosis_report_multiclass_with_accuracy(
            ct_true, ct_pred, ct_class_names
        )
    )

    plot_ct_confusion_matrix(
        ct_true, ct_pred, ct_class_names
    )

    # =======================
    # TRAINING CURVES
    # =======================
    plot_accuracy(history)
    plot_loss(history)


######################################################################
########### Function Execution Started ###############################
######################################################################

train_ds, val_ds, xray_classes, ct_classes = load_and_prepare_data()

model = build_and_compile_model(xray_classes, ct_classes)

history = train_model(model, train_ds, val_ds)

xray_true, xray_pred, ct_true, ct_pred = run_validation(model, val_ds)

run_reports_and_plots(
        xray_true, xray_pred,
        ct_true, ct_pred,
        xray_classes, ct_classes,
        history
    )

