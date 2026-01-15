import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from google.colab import drive

# 1. SETUP & DRIVE MOUNTING
drive.mount('/content/drive')

# Resource Check
gpu_info = tf.test.gpu_device_name()
if gpu_info != '/device:GPU:0':
    print('Warning: GPU not found. High-performance training requires a GPU.')
else:
    print('Connected to GPU:', gpu_info)



# Resource Check
gpu_info = tf.test.gpu_device_name()
if gpu_info != '/device:GPU:0':
    print('Warning: GPU not found. High-performance training requires a GPU.')
else:
    print('Connected to GPU:', gpu_info)

# =====================================================
# 2. GNN LAYER (Graph Attention Optimization)
# =====================================================
class GATFeatureOptimizer(layers.Layer):
    """
    Optimizes Patient Features by attending to learned Disease-Class Nodes.
    Implements a Single-layer Graph Attention Network (GAT).
    """
    def __init__(self, num_nodes, embedding_dim, **kwargs):
        super(GATFeatureOptimizer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Disease Nodes: Represent global semantic knowledge
        self.disease_nodes = self.add_weight(
            shape=(num_nodes, embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="disease_nodes"
        )

        # Attention kernel: a^T [h_i || h_j]
        self.attn_kernel = self.add_weight(
            shape=(2 * embedding_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_kernel"
        )

    def call(self, patient_features):
        batch_size = tf.shape(patient_features)[0]

        # Prepare Nodes for attention (Batch, Num_Nodes, Dim)
        disease_nodes_batched = tf.repeat(tf.expand_dims(self.disease_nodes, 0), batch_size, axis=0)
        patient_expanded = tf.repeat(tf.expand_dims(patient_features, 1), self.num_nodes, axis=1)

        # Compute Attention Coefficients
        combined = tf.concat([patient_expanded, disease_nodes_batched], axis=-1)
        attn_scores = tf.matmul(combined, self.attn_kernel)
        attn_weights = tf.nn.softmax(tf.nn.leaky_relu(attn_scores), axis=1)

        # Refine Patient Features via weighted sum of disease relationships
        node_influence = tf.reduce_sum(attn_weights * disease_nodes_batched, axis=1)
        return patient_features + node_influence

    def get_config(self):
        config = super().get_config()
        config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})
        return config

# =====================================================
# 3. MODEL ARCHITECTURE
# =====================================================
class DualModalEncoder(Model):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.xray_backbone = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
        self.ct_backbone = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

        self.xray_proj = layers.Dense(feature_dim, activation='relu')
        self.ct_proj = layers.Dense(feature_dim, activation='relu')
        self.fusion = layers.Dense(feature_dim, activation='relu')

    def call(self, inputs, training=None):
        xray, ct = inputs
        xf = self.xray_proj(self.xray_backbone(xray, training=training))
        cf = self.ct_proj(self.ct_backbone(ct, training=training))
        return self.fusion(tf.concat([xf, cf], axis=-1))

def build_gnn_model(num_disease, num_cancer):
    x_in = layers.Input(shape=(224, 224, 3), name='xray_input')
    c_in = layers.Input(shape=(224, 224, 3), name='ct_input')

    # Feature Extraction & Fusion
    fused_features = DualModalEncoder()( [x_in, c_in] )

    # GNN Optimization
    total_nodes = num_disease + num_cancer
    optimized = GATFeatureOptimizer(num_nodes=total_nodes, embedding_dim=512)(fused_features)

    # Classification Heads
    d_out = layers.Dense(num_disease, activation='softmax', name='disease_out')(optimized)
    c_out = layers.Dense(num_cancer, activation='softmax', name='cancer_out')(optimized)

    return Model([x_in, c_in], [d_out, c_out])

# =====================================================
# 4. DATA PIPELINE & HELPERS
# =====================================================
def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img / 255.0

def make_ds(paths, labels, batch_size=32):
    def gen():
        for p, l in zip(paths, labels):
            yield load_img(p), l
    ds = tf.data.Dataset.from_generator(
        gen, output_signature=(tf.TensorSpec((224, 224, 3), tf.float32), tf.TensorSpec((), tf.int32))
    )
    return ds.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

def get_data(base_dir, tr_n, va_n):
    cls = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for i, c in enumerate(cls):
        p = os.path.join(base_dir, c)
        imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        np.random.shuffle(imgs)
        tr_p.extend(imgs[:tr_n]); tr_l.extend([i]*len(imgs[:tr_n]))
        va_p.extend(imgs[tr_n:tr_n+va_n]); va_l.extend([i]*len(imgs[tr_n:tr_n+va_n]))
    return (tr_p, tr_l), (va_p, va_l), cls


# =====================================================
# 5. MAIN EXECUTION
# =====================================================
def main():
    # Paths (Update if needed)
    gnn_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    ct_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset'

    # Load all available data and then align
    (gx_tr_all, gl_tr_all), (gx_va_all, gl_va_all), d_classes = get_data(gnn_dir, 100, 50)
    (cx_tr_all, cl_tr_all), (cx_va_all, cl_va_all), c_classes = get_data(ct_dir, 100, 50)

    # Align the number of samples for paired datasets BEFORE creating TF Datasets
    min_train_len = min(len(gx_tr_all), len(cx_tr_all))
    min_val_len = min(len(gx_va_all), len(cx_va_all))

    # Truncate lists to the minimum length
    gx_tr = gx_tr_all[:min_train_len]
    gl_tr = gl_tr_all[:min_train_len]
    cx_tr = cx_tr_all[:min_train_len]
    cl_tr = cl_tr_all[:min_train_len]

    gx_va = gx_va_all[:min_val_len]
    gl_va = gl_va_all[:min_val_len]
    cx_va = cx_va_all[:min_val_len]
    cl_va = cl_va_all[:min_val_len]

    # Calculate Steps
    batch_size = 32
    train_steps = min_train_len // batch_size
    val_steps = min_val_len // batch_size

    # Add checks for insufficient samples
    if train_steps == 0:
        print(
            f"Error: Not enough aligned training samples ({min_train_len}) to create even one batch (size {batch_size}). Aborting.")
        return

    print(f"Aligned training samples: {min_train_len}, resulting in {train_steps} steps per epoch.")
    print(f"Aligned validation samples: {min_val_len}, resulting in {val_steps} validation steps.")

    # Create Datasets
    # make_ds already has drop_remainder=True, ensuring consistent batch sizes
    train_ds_gnn = make_ds(gx_tr, gl_tr, batch_size=batch_size)
    train_ds_ct = make_ds(cx_tr, cl_tr, batch_size=batch_size)

    # Shuffle after zipping batches to ensure paired samples stay together, but use a buffer of batches
    train_ds = tf.data.Dataset.zip((train_ds_gnn, train_ds_ct)).map(lambda x, y: ((x[0], y[0]), (x[1], y[1]))).shuffle(
        buffer_size=train_steps).prefetch(tf.data.AUTOTUNE)

    val_ds_gnn = make_ds(gx_va, gl_va, batch_size=batch_size)
    val_ds_ct = make_ds(cx_va, cl_va, batch_size=batch_size)
    val_ds_zipped = tf.data.Dataset.zip((val_ds_gnn, val_ds_ct)).map(
        lambda x, y: ((x[0], y[0]), (x[1], y[1]))).prefetch(tf.data.AUTOTUNE)

    # Only repeat validation dataset if there are actual validation steps
    if val_steps > 0:
        final_val_ds = val_ds_zipped.repeat()
        print("Validation data will be used.")
    else:
        final_val_ds = None
        print("Warning: Not enough aligned validation samples to create even one batch. Validation will be skipped.")

    # Build & Compile
    model = build_gnn_model(len(d_classes), len(c_classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics={'disease_out': 'accuracy', 'cancer_out': 'accuracy'})

    # Callbacks
    lr_sch = callbacks.ReduceLROnPlateau(monitor='val_loss' if val_steps > 0 else 'loss', factor=0.5, patience=2,
                                         verbose=1)
    stop = callbacks.EarlyStopping(monitor='val_loss' if val_steps > 0 else 'loss', patience=5,
                                   restore_best_weights=True)

    # Train
    print(f"\nTraining for {train_steps} steps per epoch...")
    fit_kwargs = {
        'x': train_ds,
        'epochs': 15,  # Increased epochs from 10 to 15 for potentially better convergence
        'steps_per_epoch': train_steps,
        'callbacks': [lr_sch, stop],
        'verbose': 1
    }
    if final_val_ds is not None:
        fit_kwargs['validation_data'] = final_val_ds
        fit_kwargs['validation_steps'] = val_steps

    history = model.fit(**fit_kwargs)

    # --- EVALUATION & PLOTTING ---
    if final_val_ds is not None and val_steps > 0:
        print("\nGenerating final evaluation plots...")
        d_true, c_true, d_pred, c_pred = [], [], [], []
        # Use val_ds_zipped (non-repeating) for evaluation to avoid infinite loop
        for (x, c), (yd, yc) in val_ds_zipped.take(val_steps):
            res = model.predict((x, c), verbose=0)
            d_true.extend(yd.numpy());
            c_true.extend(yc.numpy())
            d_pred.extend(res[0]);
            c_pred.extend(res[1])

        d_pred, c_pred = np.array(d_pred), np.array(c_pred)

        # Plot training history (loss and accuracy for both outputs)
        plot_training_history(history)

        if len(d_true) == 0:
            print("Warning: No true disease labels collected for evaluation. Skipping disease plots.")
        else:
            # Disease Metrics
            y_true_g_oh = tf.keras.utils.to_categorical(d_true, len(d_classes))
            plot_multiclass_roc(y_true_g_oh, d_pred, d_classes, "Disease")
            plot_confusion_matrix(d_true, np.argmax(d_pred, axis=1), d_classes, "Disease")

        if len(c_true) == 0:
            print("Warning: No true cancer labels collected for evaluation. Skipping cancer plots.")
        else:
            # Cancer Metrics
            y_true_c_oh = tf.keras.utils.to_categorical(c_true, len(c_classes))
            plot_multiclass_roc(y_true_c_oh, c_pred, c_classes, "Cancer")
            plot_confusion_matrix(c_true, np.argmax(c_pred, axis=1), c_classes, "Cancer")

    else:
        print("Skipping final evaluation plots due to insufficient validation data.")

    model.save('/content/drive/MyDrive/GNN_Final_Model.keras')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
import numpy as np

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Loss Plots
    axes[0,0].plot(history.history['disease_out_loss'], label='Train'); axes[0,0].plot(history.history['val_disease_out_loss'], label='Val')
    axes[0,0].set_title('Disease Loss'); axes[0,0].grid(True); axes[0,0].legend()
    axes[0,1].plot(history.history['cancer_out_loss'], label='Train'); axes[0,1].plot(history.history['val_cancer_out_loss'], label='Val')
    axes[0,1].set_title('Cancer Loss'); axes[0,1].grid(True); axes[0,1].legend()
    # Accuracy Plots
    axes[1,0].plot(history.history['disease_out_accuracy'], label='Train'); axes[1,0].plot(history.history['val_disease_out_accuracy'], label='Val')
    axes[1,0].set_title('Disease Accuracy'); axes[1,0].grid(True); axes[1,0].legend()
    axes[1,1].plot(history.history['cancer_out_accuracy'], label='Train'); axes[1,1].plot(history.history['val_cancer_out_accuracy'], label='Val')
    axes[1,1].set_title('Cancer Accuracy'); axes[1,1].grid(True); axes[1,1].legend()
    plt.tight_layout(); plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

def plot_multiclass_roc(y_true_one_hot, y_pred_probs, class_names, title_suffix=""):
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve: {title_suffix}'); plt.legend(); plt.grid(True); plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, title=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {title}'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

if __name__ == "__main__":
    main()