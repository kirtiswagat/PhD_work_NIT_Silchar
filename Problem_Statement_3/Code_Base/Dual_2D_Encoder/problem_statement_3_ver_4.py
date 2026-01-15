import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# DualModalEncoder: separate encoders + feature fusion
class DualModalEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, dropout_rate=0.3):
        super().__init__()
        # X-ray encoder
        self.xray_encoder = EfficientNetB0(include_top=False, weights='imagenet',
                                          input_shape=input_shape, pooling='avg')
        self.xray_encoder.trainable = True
        self.xray_feat_extractor = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate/2)
        ])
        # CT encoder
        self.ct_encoder = EfficientNetB0(include_top=False, weights='imagenet',
                                        input_shape=input_shape, pooling='avg')
        self.ct_encoder.trainable = True
        self.ct_feat_extractor = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate/2)
        ])
        # Fusion layer
        self.fusion = tf.keras.Sequential([
            layers.Concatenate(),
            layers.Dense(feature_dim*2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
        ])

    def call(self, inputs, training=None):
        xray_input, ct_input = inputs
        xray_features = self.xray_encoder(xray_input, training=training)
        xray_features = self.xray_feat_extractor(xray_features, training=training)
        ct_features = self.ct_encoder(ct_input, training=training)
        ct_features = self.ct_feat_extractor(ct_features, training=training)
        concatenated = tf.concat([xray_features, ct_features], axis=-1)
        fused = self.fusion(concatenated, training=training)
        return fused

# Build multi-task classifier model
def build_classifier(dual_encoder, num_diseases, num_cancer_classes):
    xray_in = layers.Input(shape=(224,224,3))
    ct_in = layers.Input(shape=(224,224,3))
    features = dual_encoder([xray_in, ct_in])
    # Multi-label disease prediction
    disease_out = layers.Dense(num_diseases, activation='sigmoid', name='disease_out')(features)
    # Multi-class cancer prediction
    cancer_out = layers.Dense(num_cancer_classes, activation='softmax', name='cancer_out')(features)
    model = Model(inputs=[xray_in, ct_in], outputs=[disease_out, cancer_out])
    return model

# Utils to load and preprocess images
def load_and_preprocess_image(path, image_size=(224,224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img

# Create dataset from folder of images organized by class/subfolders
def create_dataset_from_folder(base_dir, class_names, image_size=(224,224), batch_size=32, shuffle=True, multi_label=True):
    filepaths = []
    labels = []
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, base_dir)
                parts = rel_path.split(os.sep)
                label_vec = np.zeros(len(class_names), dtype=np.float32) if multi_label else None
                if multi_label:
                    for p in parts:
                        if p in label_to_idx:
                            label_vec[label_to_idx[p]] = 1.0
                else:
                    # Single-label mode - assign label of immediate folder
                    label_name = parts[0]
                    label_vec = label_to_idx[label_name]
                filepaths.append(path)
                labels.append(label_vec)

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    def gen():
        for fp, lbl in zip(filepaths, labels):
            img = load_and_preprocess_image(fp, image_size)
            yield img, lbl

    output_types = (tf.float32, tf.float32) if multi_label else (tf.float32, tf.int32)
    output_shapes = ((*image_size, 3), (len(class_names) if multi_label else ()))
    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)

    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Paired dataset generator (xray_ds & ct_ds) without alignment (random pairing)
def create_paired_dataset(xray_ds, ct_ds):
    zipped = tf.data.Dataset.zip((xray_ds, ct_ds))
    # Format: ((xray_img_batch, xray_labels), (ct_img_batch, ct_labels)) -> ([xray_img_batch, ct_img_batch], [xray_labels, ct_labels])
    paired_ds = zipped.map(lambda x, y: ([x[0], y[0]], [x[1], y[1]]))
    return paired_ds

# Training procedure for multi-task
def train_multitask_model(model, train_ds, val_ds, test_ds, epochs=10, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
                'disease_out': 'binary_crossentropy',
                'cancer_out': 'sparse_categorical_crossentropy'
            },
        metrics={
                'disease_out': [
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall'),
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
                                ],
                'cancer_out': [
                                tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                                tf.keras.metrics.AUC(name='auc'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
                                ],
                }
    )
    callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
                    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
                ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print("\nEvaluating on test data:")
    # Evaluate multilabel confusion for disease_out
    y_true, y_pred = [], []
    for x, y in test_ds:
        preds = model.predict(x)
        pred_labels = (preds[0] >= 0.5).astype(int)
        y_pred.append(pred_labels)
        y_true.append(y[0].numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    for i, cm in enumerate(cms):
        tn, fp, fn, tp = cm.ravel()
        print(f"Disease class {i}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Cancer classification report
    y_true_cancer = np.concatenate([y[1].numpy() for _, y in test_ds], axis=0)
    y_pred_cancer = np.argmax(np.concatenate([model.predict(x)[1] for x, _ in test_ds], axis=0), axis=1)
    print("\nCancer classification report:")
    print(classification_report(y_true_cancer, y_pred_cancer))
    return history


def plot_history(history):
    plt.figure(figsize=(14, 6))
    
    # Plot accuracy for disease_out (multi-label)
    plt.subplot(2, 2, 1)
    plt.plot(history.history.get('disease_out_accuracy', []), label='Train Disease Acc')
    plt.plot(history.history.get('val_disease_out_accuracy', []), label='Val Disease Acc')
    plt.title('Disease Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss for disease_out
    plt.subplot(2, 2, 2)
    plt.plot(history.history.get('disease_out_loss', []), label='Train Disease Loss')
    plt.plot(history.history.get('val_disease_out_loss', []), label='Val Disease Loss')
    plt.title('Disease Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy for cancer_out (multi-class)
    plt.subplot(2, 2, 3)
    plt.plot(history.history.get('cancer_out_accuracy', []), label='Train Cancer Acc')
    plt.plot(history.history.get('val_cancer_out_accuracy', []), label='Val Cancer Acc')
    plt.title('Cancer Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss for cancer_out
    plt.subplot(2, 2, 4)
    plt.plot(history.history.get('cancer_out_loss', []), label='Train Cancer Loss')
    plt.plot(history.history.get('val_cancer_out_loss', []), label='Val Cancer Loss')
    plt.title('Cancer Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def compute_f1_score(model, dataset, threshold=0.5):
    y_true = []
    y_pred = []
    for x, y in dataset:
        preds = model.predict(x)
        # assuming model has multiple outputs; adjust indexing accordingly
        pred_labels = (preds[0] >= threshold).astype(int)  # disease output; index 0 if first output
        y_true.append(y[0].numpy())                        # true disease labels; adjust indexing
        y_pred.append(pred_labels)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1 Score (Disease prediction): {f1:.4f}")
    return f1

def plot_roc_curves(model, dataset, class_names):
    y_true = []
    y_scores = []
    for x, y in dataset:
        preds = model.predict(x)
        y_true.append(y[0].numpy())     # true disease labels
        y_scores.append(preds[0])       # predicted disease probabilities
    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)
    
    plt.figure(figsize=(10,8))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Disease Prediction')
    plt.legend(loc='lower right')
    plt.show()

def plot_multilabel_roc(y_true, y_scores, class_names, title='ROC Curve for Multilabel Classification'):
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_multiclass_roc(y_true, y_scores, class_names, title='ROC Curve for Multiclass Classification'):
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def compute_and_plot_roc(model, test_ds, xray_classes, ct_classes):
    y_true_xray = []
    y_scores_xray = []
    y_true_ct = []
    y_scores_ct = []
    
    for (x_in_batch, y_batch) in test_ds:
        preds = model.predict(x_in_batch)
        # preds: list with [disease_probs, cancer_probs]
        
        y_true_xray.append(y_batch[0].numpy())
        y_scores_xray.append(preds[0])
        
        y_true_ct.append(y_batch[1].numpy())
        y_scores_ct.append(preds[1])
    
    y_true_xray = np.vstack(y_true_xray)
    y_scores_xray = np.vstack(y_scores_xray)
    y_true_ct = np.concatenate(y_true_ct)
    y_scores_ct = np.vstack(y_scores_ct)
    
    plot_multilabel_roc(y_true_xray, y_scores_xray, xray_classes, title='Chest X-ray ROC Curves')
    plot_multiclass_roc(y_true_ct, y_scores_ct, ct_classes, title='CT Scan ROC Curves')


# Creation of CT Dataset
def get_ct_filepaths_labels(ct_dir, class_names):
    filepaths = []
    labels = []
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(ct_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return filepaths, labels

def create_ct_dataset(filepaths, labels, batch_size=32, shuffle=True, image_size=(224,224)):
    def _gen():
        for fp, label in zip(filepaths, labels):
            img = tf.io.read_file(fp)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            img = img / 255.0
            yield img, label

    output_signature = (
        tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    
    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds



# Example of usage
def main():
    xray_classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                   'Enlarged_Cardiomediastinum', 'Fracture', 'Lung_Lesion',
                   'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                   'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
    ct_classes = ['benign', 'malignant', 'normal']

    # Paths
    xray_train_dir = '/content/drive/MyDrive/CheXpert/train'
    xray_test_dir = '/content/drive/MyDrive/CheXpert/test'
    # ct_train_dir = '/content/drive/MyDrive/CT/train'
    # ct_test_dir = '/content/drive/MyDrive/CT/test'

    ct_dir = '/path/to/CT'
    ct_classes = ['benign', 'malignant', 'normal']

    filepaths, labels = get_ct_filepaths_labels(ct_dir, ct_classes)
    filepaths = np.array(filepaths)
    labels = np.array(labels)

    # Split to train/val/test
    train_fp, temp_fp, train_lbl, temp_lbl = train_test_split(filepaths, labels, test_size=0.3, stratify=labels, random_state=42)
    val_fp, test_fp, val_lbl, test_lbl = train_test_split(temp_fp, temp_lbl, test_size=0.5, stratify=temp_lbl, random_state=42)


    # Load datasets
    xray_train_ds = create_dataset_from_folder(xray_train_dir, xray_classes, batch_size=32, shuffle=True, multi_label=True)
    xray_val_ds = create_dataset_from_folder(xray_test_dir, xray_classes, batch_size=32, shuffle=False, multi_label=True)
    # ct_train_ds = create_dataset_from_folder(ct_train_dir, ct_classes, batch_size=32, shuffle=True, multi_label=False)
    # ct_val_ds = create_dataset_from_folder(ct_test_dir, ct_classes, batch_size=32, shuffle=False, multi_label=False)

    ct_train_ds = create_ct_dataset(train_fp, train_lbl, batch_size=32)
    ct_val_ds = create_ct_dataset(val_fp, val_lbl, batch_size=32)

    # Create paired datasets (random pairs, no alignment)
    train_ds = create_paired_dataset(xray_train_ds, ct_train_ds)
    val_ds = create_paired_dataset(xray_val_ds, ct_val_ds)
    test_ds = val_ds  # For demo, using val as test

    # Build model
    dual_encoder = DualModalEncoder(input_shape=(224,224,3), feature_dim=512)
    model = build_classifier(dual_encoder, num_diseases=len(xray_classes), num_cancer_classes=len(ct_classes))

    # Train model
    history = train_multitask_model(model, train_ds, val_ds, test_ds, epochs=10, lr=1e-4)

    # Plot training/validation accuracy and loss here if desired
    plot_history(history)

    # Compute and print F1 score for disease task
    compute_f1_score(model, test_ds, threshold=0.5)

    # Plot ROC curves for disease task
    # Compute and plot ROC curves on test set
    compute_and_plot_roc(model, test_ds, xray_classes, ct_classes)

if __name__ == "__main__":
    main()
