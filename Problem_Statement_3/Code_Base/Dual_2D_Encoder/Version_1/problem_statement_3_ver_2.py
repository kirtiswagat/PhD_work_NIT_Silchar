import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt

# Squeeze-and-Excitation block
def squeeze_and_excitation_block(inputs, ratio=16):
    channel_dim = inputs.shape[-1]
    squeeze = tf.reduce_mean(inputs, axis=1, keepdims=True)
    excitation = layers.Dense(channel_dim // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channel_dim, activation='sigmoid')(excitation)
    return layers.Multiply()([inputs, excitation])

# DualEfficientNetEncoder definition
class DualEfficientNetEncoder(Model):
    def __init__(self, input_shape=(224,224,3), feature_dim=512, use_shared_weights=False, dropout_rate=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_shared_weights = use_shared_weights
        self.dropout_rate = dropout_rate

        self.encoder1 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self.encoder1.trainable = True
        if self.use_shared_weights:
            self.encoder2 = self.encoder1
        else:
            self.encoder2 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
            self.encoder2.trainable = True

        self.feature_extractor1 = tf.keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate/2),
        ])
        if self.use_shared_weights:
            self.feature_extractor2 = self.feature_extractor1
        else:
            self.feature_extractor2 = tf.keras.Sequential([
                layers.Dense(1024, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.feature_dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate/2),
            ])

        self.fusion_layer = tf.keras.Sequential([
            layers.Lambda(lambda x: squeeze_and_excitation_block(x)),
            layers.Dense(self.feature_dim * 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.feature_dim, activation='relu'),
            layers.BatchNormalization(),
        ])

    def call(self, inputs, training=None):
        if isinstance(inputs, list) and len(inputs)==2:
            input1, input2 = inputs
        else:
            input1 = input2 = inputs
        feat1 = self.encoder1(input1, training=training)
        feat2 = self.encoder2(input2, training=training)
        extracted1 = self.feature_extractor1(feat1, training=training)
        extracted2 = self.feature_extractor2(feat2, training=training)
        concat = layers.Concatenate()([extracted1, extracted2])
        fused = self.fusion_layer(concat, training=training)
        return fused

# Build classification model
def build_classification_model(dual_encoder, num_classes=14):
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
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=[input1, input2], outputs=outputs)

# Compute multilabel confusion matrices
def compute_multilabel_confusion_matrix(model, dataset, threshold=0.5):
    y_true, y_pred = [], []
    for x, y in dataset:
        preds = model.predict(x)
        preds_bin = (preds >= threshold).astype(int)
        y_pred.append(preds_bin)
        y_true.append(y.numpy())
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    for i, cm in enumerate(cms):
        tn, fp, fn, tp = cm.ravel()
        print(f"Label {i}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return cms

# Training function with metrics and callbacks
def train_model(model, train_ds, val_ds, test_ds, epochs=10, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
        ],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print("\nConfusion matrices on test set:")
    compute_multilabel_confusion_matrix(model, test_ds)
    return history

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

# Plot training results
def plot_history(history, metric='accuracy'):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history[metric], label='train_'+metric)
    plt.plot(history.history['val_'+metric], label='val_'+metric)
    plt.title(f'{metric} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example main execution stub
def main():
    # Prepare datasets: train_ds, val_ds, test_ds yielding ((img1, img2), labels)
    # Instantiate encoder and classification model
    dual_encoder = DualEfficientNetEncoder(input_shape=(224,224,3), feature_dim=512)
    classifier = build_classification_model(dual_encoder, num_classes=14)

    # Train and evaluate
    history = train_model(classifier, train_ds, val_ds, test_ds, epochs=10, lr=1e-4)

    # Plot training results
    plot_history(history)

    # Grad-CAM visualization on first test sample and first class
    test_batch, _ = next(iter(test_ds))
    sample_img = [tf.expand_dims(test_batch[0][0], 0), tf.expand_dims(test_batch[1][0], 0)]
    visualize_gradcam(classifier, sample_img, class_idx=0)

if __name__ == '__main__':
    main()
