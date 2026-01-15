import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
# from google.colab import drive

# # 1. SETUP & DRIVE MOUNTING
# drive.mount('/content/drive')

# =====================================================
# 2. GNN LAYER (Graph Attention Optimization)
# =====================================================
class GATFeatureOptimizer(layers.Layer):
    def __init__(self, num_nodes, embedding_dim, **kwargs):
        super(GATFeatureOptimizer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.disease_nodes = self.add_weight(
            shape=(num_nodes, embedding_dim),
            initializer="glorot_uniform", trainable=True, name="disease_nodes"
        )
        self.attn_kernel = self.add_weight(
            shape=(2 * embedding_dim, 1),
            initializer="glorot_uniform", trainable=True, name="attn_kernel"
        )

    def call(self, patient_features):
        batch_size = tf.shape(patient_features)[0]
        nodes_batched = tf.repeat(tf.expand_dims(self.disease_nodes, 0), batch_size, axis=0)
        patient_expanded = tf.repeat(tf.expand_dims(patient_features, 1), self.num_nodes, axis=1)
        combined = tf.concat([patient_expanded, nodes_batched], axis=-1)
        attn_weights = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(combined, self.attn_kernel)), axis=1)
        node_influence = tf.reduce_sum(attn_weights * nodes_batched, axis=1)
        return patient_features + node_influence

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

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img / 255.0

# =====================================================
# 4. FINAL ANALYTICS & VISUALIZATION
# =====================================================
def plot_final_results(history, d_true, d_pred, c_true, c_pred, x_cls, c_cls):
    # 1. Training/Accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['loss'], label='Train'); axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss History'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(history.history['disease_out_accuracy'], label='Xray Acc')
    axes[1].plot(history.history['cancer_out_accuracy'], label='Cancer Acc')
    axes[1].set_title('Accuracy History'); axes[1].legend(); axes[1].grid(True)
    plt.show()

    # 2. Confusion Matrices
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(d_true, np.argmax(d_pred, 1)), annot=True, fmt='g', cmap='Blues', xticklabels=x_cls, yticklabels=x_cls)
    plt.title('X-ray Confusion Matrix')
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(c_true, np.argmax(c_pred, 1)), annot=True, fmt='g', cmap='Oranges', xticklabels=c_cls, yticklabels=c_cls)
    plt.title('Cancer Confusion Matrix')
    plt.show()

    # 3. Disagreement Heatmap (X-ray vs Cancer Preds)
    plt.figure(figsize=(8, 6))
    disagreement = confusion_matrix(np.argmax(d_pred, 1), np.argmax(c_pred, 1))
    sns.heatmap(disagreement, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Disagreement Heatmap (X-ray vs Cancer Predictions)')
    plt.xlabel('Cancer Predicted'); plt.ylabel('X-ray Predicted')
    plt.show()

    # 4. Reports
    print("\n--- X-ray Diagnostic Report ---")
    print(classification_report(d_true, np.argmax(d_pred, 1), target_names=x_cls))
    print("\n--- Cancer Diagnostic Report ---")
    print(classification_report(c_true, np.argmax(c_pred, 1), target_names=c_cls))


# =====================================================
# 5. MAIN EXECUTION
# =====================================================
def main():
    x_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    c_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset/Lung_Cancer_LIDC_Y_Net/LIDC_Y-Net'

    (xtp, ctp, xtl, ctl), (xvp, cvp, xvl, cvl), x_cls, c_cls = get_final_aligned_data(x_dir, c_dir, 500, 150)

    def process(x_p, c_p, x_l, c_l):
        return (load_img(x_p), load_img(c_p)), (x_l, c_l)

    train_ds = tf.data.Dataset.from_tensor_slices((xtp, ctp, xtl, ctl)).map(process).batch(32).shuffle(100)
    val_ds = tf.data.Dataset.from_tensor_slices((xvp, cvp, xvl, cvl)).map(process).batch(32)

    x_in, c_in = layers.Input(shape=(224, 224, 3)), layers.Input(shape=(224, 224, 3))
    base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    fused = layers.Dense(512, activation='relu')(layers.Concatenate()([base(x_in), base(c_in)]))
    opt = GATFeatureOptimizer(num_nodes=len(x_cls)+len(c_cls), embedding_dim=512)(fused)

    model = Model([x_in, c_in], [layers.Dense(len(x_cls), activation='softmax', name='disease_out')(opt),
                                 layers.Dense(len(c_cls), activation='softmax', name='cancer_out')(opt)])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics={'disease_out': 'accuracy', 'cancer_out': 'accuracy'})

    # Learning Rate Scheduler
    lr_sch = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

    history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[lr_sch])

    # Final Detailed Evaluation
    d_true, c_true, d_pred, c_pred = [], [], [], []
    for (xi, ci), (yd, yc) in val_ds:
        res = model.predict((xi, ci), verbose=0)
        d_true.extend(yd.numpy()); d_pred.extend(res[0])
        c_true.extend(yc.numpy()); c_pred.extend(res[1])

    plot_final_results(history, d_true, np.array(d_pred), c_true, np.array(c_pred), x_cls, c_cls)

if __name__ == "__main__":
    main()