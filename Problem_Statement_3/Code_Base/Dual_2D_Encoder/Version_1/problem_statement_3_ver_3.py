import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix

# Squeeze-and-Excitation block
def squeeze_and_excitation_block(inputs, ratio=16):
    channel_dim = inputs.shape[-1]
    squeeze = tf.reduce_mean(inputs, axis=1, keepdims=True)
    excitation = layers.Dense(channel_dim // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channel_dim, activation='sigmoid')(excitation)
    return layers.Multiply()([inputs, excitation])

# Dual-encoder with feature fusion
class DualEfficientNetEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, use_shared_weights=False, dropout_rate=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_shared_weights = use_shared_weights
        self.dropout_rate = dropout_rate
        
        self.encoder1 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self.encoder1.trainable = True
        
        if use_shared_weights:
            self.encoder2 = self.encoder1
        else:
            self.encoder2 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
            self.encoder2.trainable = True
        
        self.feature_extractor1 = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate / 2),
        ])
        self.feature_extractor2 = self.feature_extractor1 if use_shared_weights else tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate / 2),
        ])
        
        self.fusion_layer = tf.keras.Sequential([
            layers.Lambda(lambda x: squeeze_and_excitation_block(x)),
            layers.Dense(feature_dim * 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(feature_dim, activation='relu'),
            layers.BatchNormalization(),
        ])
        
    def call(self, inputs, training=None):
        if isinstance(inputs, list) and len(inputs) == 2:
            input1, input2 = inputs
        else:
            input1 = input2 = inputs
        
        feat1 = self.encoder1(input1, training=training)
        feat2 = self.encoder2(input2, training=training)
        
        f1 = self.feature_extractor1(feat1, training=training)
        f2 = self.feature_extractor2(feat2, training=training)
        
        concat = layers.Concatenate()([f1, f2])
        fused = self.fusion_layer(concat, training=training)
        return fused

# Classification model definition
def build_classification_model(dual_encoder, num_classes):
    dual_encoder.trainable = False
    input1 = layers.Input(shape=(224,224,3))
    input2 = layers.Input(shape=(224,224,3))
    
    features = dual_encoder([input1, input2])
    x = layers.Dense(256, activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='sigmoid')(x)  # multi-label
    
    model = Model(inputs=[input1, input2], outputs=output)
    return model

# Utility to load image and preprocess
def load_and_preprocess_image(path, image_size=(224,224)):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img

# Dataset creation from folder structure
def create_multilabel_dataset(data_dir, labels, image_size=(224,224), batch_size=32, shuffle=True):
    filepaths = []
    multilabels = []
    label_to_index = {label: idx for idx,label in enumerate(labels)}

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png','.jpg','.jpeg')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, data_dir)
                parts = rel_path.split(os.sep)

                # Assumes one or more disease folders in path; map to multi-hot vector
                label_vector = np.zeros(len(labels), dtype=np.float32)
                for p in parts:
                    if p in label_to_index:
                        label_vector[label_to_index[p]] = 1.0

                filepaths.append(file_path)
                multilabels.append(label_vector)

    filepaths = np.array(filepaths)
    multilabels = np.array(multilabels)

    # Prepare TF datasets
    def gen():
        for fp, lbl in zip(filepaths, multilabels):
            img1 = load_and_preprocess_image(fp, image_size)
            # For demo, duplicate img1 as second input (modify if you have a second modality)
            img2 = img1
            yield (img1, img2), lbl

    output_types = ((tf.float32, tf.float32), tf.float32)
    output_shapes = (((*image_size,3), (*image_size,3)), (len(labels),))

    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Training function
def train_model(model, train_ds, val_ds, test_ds, epochs=10, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        ]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    
    print("\nEvaluating on test set...")
    compute_multilabel_confusion_matrix(model, test_ds)
    return history

# Confusion matrix utility
def compute_multilabel_confusion_matrix(model, dataset, threshold=0.5):
    y_true = []
    y_pred = []
    for x, y in dataset:
        preds = model.predict(x)
        preds_bin = (preds >= threshold).astype(int)
        y_true.append(y.numpy())
        y_pred.append(preds_bin)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    for idx, cm in enumerate(cms):
        tn, fp, fn, tp = cm.ravel()
        print(f"Class {idx}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return cms

# Grad-CAM utilities
def compute_gradcam(model, img_tensor, class_idx, last_conv_layer='top_conv'):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1,2))
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    cam = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img_tensor.shape[1], img_tensor.shape[2]))
    return heatmap.numpy().squeeze()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap='viridis'):    
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap)
    colored_heatmap = np.delete(colored_heatmap, 3, 2)  # remove alpha
    img = img / np.max(img)
    overlayed_img = colored_heatmap[..., :3] * alpha + img[..., :3]
    overlayed_img = overlayed_img / np.max(overlayed_img)
    return overlayed_img


def visualize_gradcam(model, input_tensor, class_idx):
    heatmap = compute_gradcam(model, input_tensor, class_idx)
    img = input_tensor[0].numpy().astype(np.uint8)
    overlay = overlay_heatmap(img, heatmap)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap='viridis')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()

# Plot training curves
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.show()

# Example usage
def main():
    chex_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                   'Enlarged_Cardiomediastinum', 'Fracture', 'Lung_Lesion',
                   'Lung_Opacity', 'No_Finding', 'Pleural_Effusion',
                   'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
    
    train_dir = '/content/drive/MyDrive/CheXpert/train'
    test_dir = '/content/drive/MyDrive/CheXpert/test'
    
    full_df = None  # If you have a CSV, load and split accordingly
    
    # Prepare datasets from folders (demo: treating folder names as labels)
    train_ds = create_multilabel_dataset(train_dir, chex_labels, batch_size=32, shuffle=True)
    val_ds = create_multilabel_dataset(test_dir, chex_labels, batch_size=32, shuffle=False)  # For demo, using test as val
    test_ds = create_multilabel_dataset(test_dir, chex_labels, batch_size=32, shuffle=False)

    encoder = DualEfficientNetEncoder(input_shape=(224,224,3), feature_dim=512)
    model = build_classification_model(encoder, num_classes=len(chex_labels))

    history = train_model(model, train_ds, val_ds, test_ds, epochs=10, lr=1e-4)
    plot_history(history)

    # Grad-CAM visualization example (take one batch from test_ds)
    for batch, labels in test_ds.take(1):
        sample_input = [tf.expand_dims(batch[0][0], 0), tf.expand_dims(batch[1][0], 0)]
        visualize_gradcam(model, sample_input, class_idx=0)  # visualize class 0
        break

if __name__ == '__main__':
    main()
