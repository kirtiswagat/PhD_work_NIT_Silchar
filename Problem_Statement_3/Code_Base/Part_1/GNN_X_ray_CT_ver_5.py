#####################################################
#         Implementing                              #
#         Sparse Categorical Cross-Entropy          #
#         Loss                                      #
#####################################################
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# 1. GNN LAYER & AUGMENTATION
# =====================================================
class GATFeatureOptimizer(layers.Layer):
    def __init__(self, num_nodes, embedding_dim, **kwargs):
        super(GATFeatureOptimizer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.disease_nodes = self.add_weight(shape=(num_nodes, embedding_dim), initializer="glorot_uniform", trainable=True)
        self.attn_kernel = self.add_weight(shape=(2 * embedding_dim, 1), initializer="glorot_uniform", trainable=True)

    def call(self, patient_features):
        batch_size = tf.shape(patient_features)[0]
        nodes_batched = tf.repeat(tf.expand_dims(self.disease_nodes, 0), batch_size, axis=0)
        patient_expanded = tf.repeat(tf.expand_dims(patient_features, 1), self.num_nodes, axis=1)
        combined = tf.concat([patient_expanded, nodes_batched], axis=-1)
        attn_weights = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(combined, self.attn_kernel)), axis=1)
        return patient_features + tf.reduce_sum(attn_weights * nodes_batched, axis=1)

augmenter = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

def load_img(path, augment=False):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (224, 224)) / 255.0
    if augment:
        img = augmenter(tf.expand_dims(img, 0), training=True)[0]
    return img

# =====================================================
# 3. HYBRID DATA LOADING
# =====================================================
def get_final_aligned_data(xray_dir, ct_dir, tr_n, va_n):
    # Fixed X-ray classes
    # x_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', ]
    x_classes = ['Pleural_Effusion', 'Edema', 'Lung_Opacity']

    # Dynamic Cancer classes from folder names
    c_classes = sorted([d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d)) and not d.startswith('.')])

    x_tr_p, c_tr_p, x_tr_l, c_tr_l = [], [], [], []
    x_va_p, c_va_p, x_va_l, c_va_l = [], [], [], []

    print(f"X-ray targets: {x_classes}")
    print(f"Detected Cancer folders: {c_classes}")

    for i, x_cls in enumerate(x_classes):
        x_path = os.path.join(xray_dir, x_cls)
        if not os.path.exists(x_path): continue

        x_imgs = [os.path.join(x_path, f) for f in os.listdir(x_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        np.random.shuffle(x_imgs)

        # Ensure we have a matching folder in CT, else rotate through available ones
        c_target_idx = i % len(c_classes)
        c_cls_folder = c_classes[c_target_idx]
        c_path = os.path.join(ct_dir, c_cls_folder)
        c_imgs = [os.path.join(c_path, f) for f in os.listdir(c_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        np.random.shuffle(c_imgs)

        n_tr = min(len(x_imgs), tr_n)

        # Pairs for training
        x_tr_p.extend(x_imgs[:n_tr])
        c_tr_p.extend([np.random.choice(c_imgs) for _ in range(n_tr)])
        x_tr_l.extend([i] * n_tr)
        c_tr_l.extend([c_target_idx] * n_tr)

        # Pairs for validation
        n_va = min(len(x_imgs) - n_tr, va_n)
        x_va_p.extend(x_imgs[n_tr:n_tr+n_va])
        c_va_p.extend([np.random.choice(c_imgs) for _ in range(n_va)])
        x_va_l.extend([i] * n_va)
        c_va_l.extend([c_target_idx] * n_va)

    return (x_tr_p, c_tr_p, x_tr_l, c_tr_l), (x_va_p, c_va_p, x_va_l, c_va_l), x_classes, c_classes

# =====================================================
# 2. FIXED WEIGHT CALCULATION
# =====================================================
def calculate_multi_output_weights(x_labels, c_labels):
    # Ensure keys are strictly integers to avoid TypeError
    x_w = class_weight.compute_class_weight('balanced', classes=np.unique(x_labels), y=x_labels)
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(c_labels), y=c_labels)
    return {
        'disease_out': {int(i): float(w) for i, w in enumerate(x_w)},
        'cancer_out': {int(i): float(w) for i, w in enumerate(c_w)}
    }


# =====================================================
# 3. EVALUATION & ROC-AUC
# =====================================================
def plot_high_res_results(history, d_true, d_pred, c_true, c_pred, x_cls, c_cls):
    # d_true and c_true are already sparse integer labels, so use them directly
    d_pred_idx, c_pred_idx = np.argmax(d_pred, axis=1), np.argmax(c_pred, axis=1)

    plt.figure(figsize=(14, 5), dpi=300)
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(d_true, d_pred_idx), annot=True, fmt='g', cmap='Blues', xticklabels=x_cls, yticklabels=x_cls)
    plt.title('X-ray Confusion Matrix'); plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(c_true, c_pred_idx), annot=True, fmt='g', cmap='Oranges', xticklabels=c_cls, yticklabels=c_cls)
    plt.title('Cancer Confusion Matrix'); plt.show()

#============================================
# 4. FINAL ANALYTICS & VISUALIZATION
# =====================================================
def plot_final_results(history, d_true, d_pred, c_true, c_pred, x_cls, c_cls):
    """Handles One-Hot encoded labels for high-resolution diagnostic reporting."""

    # d_true and c_true are already sparse integer labels, so use them directly
    d_true_idx = d_true
    c_true_idx = c_true

    # Convert probability predictions to integer class indices
    d_pred_idx = np.argmax(d_pred, axis=1)
    c_pred_idx = np.argmax(c_pred, axis=1)

    # 1. Loss and Accuracy Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss History (Label Smoothed)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history['disease_out_accuracy'], label='Xray Acc')
    axes[1].plot(history.history['cancer_out_accuracy'], label='Cancer Acc')
    axes[1].set_title('Accuracy History')
    axes[1].legend(); axes[1].grid(True)
    plt.show()

    # 2. High-Resolution Confusion Matrices
    plt.figure(figsize=(14, 5), dpi=300)
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(d_true_idx, d_pred_idx), annot=True, fmt='g',
                cmap='Blues', xticklabels=x_cls, yticklabels=x_cls)
    plt.title('X-ray Confusion Matrix (Fixed Weighting)')

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(c_true_idx, c_pred_idx), annot=True, fmt='g',
                cmap='Oranges', xticklabels=c_cls, yticklabels=c_cls)
    plt.title('Cancer Confusion Matrix (Fixed Weighting)')
    plt.show()

    # 3. Final Classification Reports
    print("\n--- X-ray Diagnostic Report (Sensitivity/Recall Focus) ---")
    print(classification_report(d_true_idx, d_pred_idx, target_names=x_cls))

    print("\n--- Cancer Diagnostic Report (Sensitivity/Recall Focus) ---")
    print(classification_report(c_true_idx, c_pred_idx, target_names=c_cls))
# =====================================================
# DATA PIPELINE (ONE-HOT FOR LABEL SMOOTHING)
# =====================================================
def prepare_datasets(xtp, ctp, xtl, ctl, xvp, cvp, xvl, cvl, x_cls, c_cls):
    def map_fn(xp, cp, xl, cl, augment=False):
        # Convert to One-Hot to allow CategoricalCrossentropy with label smoothing
        return (load_img(xp, augment), load_img(cp, augment)), \
               (tf.one_hot(xl, len(x_cls)), tf.one_hot(cl, len(c_cls)))

    train_ds = tf.data.Dataset.from_tensor_slices((xtp, ctp, xtl, ctl)).map(
        lambda xp, cp, xl, cl: map_fn(xp, cp, xl, cl, True)).batch(32).shuffle(100)
    val_ds = tf.data.Dataset.from_tensor_slices((xvp, cvp, xvl, cvl)).map(
        lambda xp, cp, xl, cl: map_fn(xp, cp, xl, cl, False)).batch(32)
    return train_ds, val_ds

# =====================================================
# 5. MAIN EXECUTION
# =====================================================
def main():
    x_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    c_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset/Lung_Cancer_LIDC_Y_Net/LIDC_Y-Net'

    (xtp, ctp, xtl, ctl), (xvp, cvp, xvl, cvl), x_cls, c_cls = get_final_aligned_data(x_dir, c_dir, 500, 150)
    # multi_weights = calculate_multi_output_weights(xtl, ctl) # No longer needed here if using standard losses
    train_ds, val_ds = prepare_datasets(xtp, ctp, xtl, ctl, xvp, cvp, xvl, cvl, x_cls, c_cls)

    x_in, c_in = layers.Input(shape=(224, 224, 3)), layers.Input(shape=(224, 224, 3))
    base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

    # Regularization to address rising Val Loss
    fused = layers.Concatenate()([base(x_in), base(c_in)])
    fused = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(fused)
    fused = layers.Dropout(0.5)(fused)

    opt = GATFeatureOptimizer(num_nodes=len(x_cls)+len(c_cls), embedding_dim=512)(fused)

    # NAME layers to match keys in multi_weights
    d_out = layers.Dense(len(x_cls), activation='softmax', name='disease_out')(opt)
    c_out = layers.Dense(len(c_cls), activation='softmax', name='cancer_out')(opt)

    model = Model([x_in, c_in], [d_out, c_out])

    # Use CategoricalCrossentropy for label_smoothing support with one-hot labels
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss={'disease_out': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                        'cancer_out': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)},
                  metrics={'disease_out': 'accuracy', 'cancer_out': 'accuracy'})


    # Training with Scheduler and Early Stopping
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,

        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]
    )

    # Collect predictions
    d_true_one_hot, c_true_one_hot, d_pred, c_pred = [], [], [], [] # Renamed for clarity
    for (xi, ci), (yd, yc) in val_ds:
        res = model.predict((xi, ci), verbose=0)
        d_true_one_hot.extend(yd.numpy()); d_pred.extend(res[0])
        c_true_one_hot.extend(yc.numpy()); c_pred.extend(res[1])

    # Convert one-hot labels back to sparse integers for plotting functions
    d_true_sparse = np.argmax(np.array(d_true_one_hot), axis=1)
    c_true_sparse = np.argmax(np.array(c_true_one_hot), axis=1)

    plot_high_res_results(history, d_true_sparse,
                          np.array(d_pred), c_true_sparse,
                          np.array(c_pred), x_cls, c_cls)

    plot_final_results(history, d_true_sparse,
                          np.array(d_pred), c_true_sparse,
                          np.array(c_pred), x_cls, c_cls)

if __name__ == "__main__":
    main()
