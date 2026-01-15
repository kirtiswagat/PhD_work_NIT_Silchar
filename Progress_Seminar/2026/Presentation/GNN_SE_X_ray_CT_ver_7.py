##########################################################
#               Implementing GNN+SE                      #
#               With Categorical Focal Loss              #
##########################################################
from google.colab import drive

# 1. SETUP & DRIVE MOUNTING
drive.mount('/content/drive')

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. CATEGORICAL FOCAL LOSS WITH DYNAMIC ALPHA
# =====================================================
class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = focal_weight * cross_entropy

        if self.alpha is not None:
            alpha_tensor = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
            loss = loss * alpha_tensor

        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# =====================================================
# 2. GNN LAYER & AUGMENTATION
# =====================================================
class GATFeatureOptimizer(layers.Layer):
    def __init__(self, num_nodes, embedding_dim, **kwargs):
        super(GATFeatureOptimizer, self).__init__(**kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.disease_nodes = self.add_weight(shape=(num_nodes, embedding_dim), initializer="glorot_uniform",
                                             trainable=True)
        self.attn_kernel = self.add_weight(shape=(2 * embedding_dim, 1), initializer="glorot_uniform", trainable=True)

    def call(self, patient_features):
        batch_size = tf.shape(patient_features)[0]
        nodes_batched = tf.repeat(tf.expand_dims(self.disease_nodes, 0), batch_size, axis=0)
        patient_expanded = tf.repeat(tf.expand_dims(patient_features, 1), self.num_nodes, axis=1)
        combined = tf.concat([patient_expanded, nodes_batched], axis=-1)
        attn_weights = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(combined, self.attn_kernel)), axis=1)
        return patient_features + tf.reduce_sum(attn_weights * nodes_batched, axis=1)


def load_img(path, augment=False):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (224, 224)) / 255.0
    return img


# =====================================================
# 3. ANALYTICS: PR CURVES & GRAD-CAM
# =====================================================
def plot_precision_recall_curves(d_true_oh, d_pred, c_true_oh, c_pred, x_cls, c_cls):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, class_name in enumerate(x_cls):
        precision, recall, _ = precision_recall_curve(d_true_oh[:, i], d_pred[:, i])
        axes[0].plot(recall, precision,
                     label=f'{class_name} (AP={average_precision_score(d_true_oh[:, i], d_pred[:, i]):.2f})')
    axes[0].set_title('X-ray Precision-Recall');
    axes[0].legend()

    for i, class_name in enumerate(c_cls):
        precision, recall, _ = precision_recall_curve(c_true_oh[:, i], c_pred[:, i])
        axes[1].plot(recall, precision,
                     label=f'{class_name} (AP={average_precision_score(c_true_oh[:, i], c_pred[:, i]):.2f})')
    axes[1].set_title('Cancer Precision-Recall');
    axes[1].legend()
    plt.show()


def run_gradcam(model, img_pair, x_cls):
    xray_img_tensor = img_pair[0]
    ct_img_tensor = img_pair[1]

    # 1. Identify the EfficientNetB0 backbone layer within the main model
    efficientnet_backbone = None
    for layer in model.layers:
        if isinstance(layer, EfficientNetB0):  # EfficientNetB0 is a Model, so it has .layers
            efficientnet_backbone = layer
            break
    if efficientnet_backbone is None:
        raise ValueError("EfficientNetB0 layer not found in the model.")

    # The last convolutional layer before pooling in EfficientNetB0 (typically 'block7a_se_activate')
    target_conv_layer_name = 'block7a_se_activate'
    if target_conv_layer_name not in [layer.name for layer in efficientnet_backbone.layers]:
        # Fallback if the layer name differs (e.g., Keras internal change or different EffNet version)
        # Try to find the last layer that is a Conv2D or similar that isn't the final pooling.
        print(f"Warning: '{target_conv_layer_name}' not found. Attempting to find last Conv layer.")
        for layer in reversed(efficientnet_backbone.layers):
            if 'conv' in layer.name and not 'pool' in layer.name and hasattr(layer, 'output'):
                target_conv_layer_name = layer.name
                print(f"Using '{target_conv_layer_name}' for Grad-CAM.")
                break
        if target_conv_layer_name == 'block7a_se_activate':  # If no other was found, means no suitable conv layer
            raise ValueError("Could not find a suitable convolutional layer for Grad-CAM in EfficientNetB0.")

    # 2. Create a Grad-CAM specific model
    # This model takes the original inputs of your 'model'
    # and outputs: a) the features from the target_conv_layer (for X-ray path)
    #              b) the final 'disease_out' prediction.

    # Get the input tensors from the original model
    x_input = model.inputs[0]  # X-ray input
    c_input = model.inputs[1]  # CT input

    # Get the output of the target convolutional layer when x_input flows through the EfficientNet backbone
    # This is tricky because `efficientnet_backbone` is itself a Model and is called in the functional API.
    # We need to trace *through* the `efficientnet_backbone` as it is applied to `x_input`.

    # Create a sub-model of EfficientNetB0 that outputs the convolutional features
    base_conv_features_model = Model(
        inputs=efficientnet_backbone.input,
        outputs=efficientnet_backbone.get_layer(target_conv_layer_name).output
    )

    # Get the features (pre-pooling) for the X-ray input
    x_conv_features = base_conv_features_model(x_input)  # This is the tensor we want to get gradients from

    # Re-build the rest of the model's forward pass *from these x_conv_features*
    # This assumes the model uses GlobalAveragePooling2D after the 'block7a_se_activate' layer.
    # Since the main model was built with pooling='avg', the `base` output is already pooled.
    # We need to explicitly pool these features before feeding them into the rest of the fusion block.
    x_pooled_features = layers.GlobalAveragePooling2D(name='gradcam_x_pooling')(x_conv_features)
    c_pooled_features = efficientnet_backbone(c_input)  # CT input still goes through the full pooled backbone

    # Continue from the `fused` layer onwards, recreating the path with the new `x_pooled_features`
    # Find the layers after the backbone outputs in the original model
    # We need to identify the exact layers and their order/connections.

    # Simplified: Get the actual layers from the main model and re-apply them.
    # Assuming the structure is: [base(x_in), base(c_in)] -> Concatenate -> Dense -> GAT -> ...

    # Find the Concatenate layer and subsequent layers
    concat_layer = model.get_layer(name='concatenate')  # Assuming default name 'concatenate'
    dense_fused_layer = model.get_layer(name='dense')  # First Dense layer after concat
    gat_optimizer_layer = model.get_layer(name='gat_feature_optimizer')  # Assuming default name
    disease_output_layer = model.get_layer(name='disease_out')
    # cancer_output_layer = model.get_layer(name='cancer_out') # Not needed for X-ray Grad-CAM

    # Reconstruct the path for Grad-CAM
    fused_features_reconstructed = concat_layer([x_pooled_features, c_pooled_features])
    fused_features_reconstructed = dense_fused_layer(fused_features_reconstructed)
    # Check if fused layer has dropout if applicable (from main model)
    # Iterate through the original model's layers to find if there's dropout after this dense layer
    dropout_layer_after_dense = None
    found_dense = False
    for layer in model.layers:
        if layer.name == dense_fused_layer.name:  # Match the dense layer itself
            found_dense = True
        elif found_dense and isinstance(layer, layers.Dropout):  # Look for dropout immediately after
            fused_features_reconstructed = layer(fused_features_reconstructed)
            dropout_layer_after_dense = layer  # Keep track if we applied dropout
            break

    optimized_features_reconstructed = gat_optimizer_layer(fused_features_reconstructed)
    disease_prediction_reconstructed = disease_output_layer(optimized_features_reconstructed)

    with tf.GradientTape() as tape:
        tape.watch(x_conv_features)  # Watch the actual convolutional features

        # Now, pass the *original inputs* through the reconstructed path inside the tape
        # to get the disease prediction, ensuring `x_conv_features` is part of this computation.

        # The Grad-CAM model needs to output the `x_conv_features` AND `disease_prediction_reconstructed`
        # for the tape to link them correctly.

        # The most straightforward way is to explicitly build a model for Grad-CAM that outputs both.
        # This requires `x_input` and `c_input` to be `Input` tensors.
        # Let's create `grad_cam_model` which takes the `model.inputs` and outputs the intermediate conv features for X-ray
        # and the final disease prediction.

        # Create a model for Grad-CAM that outputs both the target conv features and the final disease prediction
        grad_cam_model = Model(
            inputs=[x_input, c_input],  # Original model inputs
            outputs=[
                base_conv_features_model(x_input),  # Conv features from X-ray path
                disease_prediction_reconstructed  # Final disease prediction
            ]
        )

        # Run this combined model inside the tape
        conv_outputs, predictions = grad_cam_model([xray_img_tensor, ct_img_tensor])

        # Target the disease output and the predicted class
        predicted_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Pool the gradients across all the filters
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by the corresponding gradient weight and sum them up
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]

    # Normalize the heatmap
    heatmap = tf.maximum(tf.squeeze(heatmap), 0) / tf.math.reduce_max(heatmap)

    # Display results
    plt.figure(figsize=(6, 6))
    plt.imshow(xray_img_tensor[0])
    plt.imshow(tf.image.resize(heatmap, (224, 224)), cmap='jet', alpha=0.4)
    plt.title(f"Grad-CAM for X-ray (Predicted: {x_cls[predicted_class_idx]}) ")
    plt.axis('off')
    plt.show()


# =====================================================
# 4. DATA & TRAINING
# =====================================================
def get_final_aligned_data(xray_dir, ct_dir, tr_n, va_n):
    x_classes = ['Pleural_Effusion', 'Edema', 'Lung_Opacity']
    c_classes = sorted(
        [d for d in os.listdir(ct_dir) if os.path.isdir(os.path.join(ct_dir, d)) and not d.startswith('.')])

    x_tr_p, c_tr_p, x_tr_l, c_tr_l = [], [], [], []
    x_va_p, c_va_p, x_va_l, c_va_l = [], [], [], []

    for i, x_cls_name in enumerate(x_classes):
        x_path = os.path.join(xray_dir, x_cls_name)
        if not os.path.exists(x_path): continue
        x_imgs = [os.path.join(x_path, f) for f in os.listdir(x_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        np.random.shuffle(x_imgs)
        c_target_idx = i % len(c_classes)
        c_path = os.path.join(ct_dir, c_classes[c_target_idx])
        c_imgs = [os.path.join(c_path, f) for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        n_tr = min(len(x_imgs), tr_n)
        x_tr_p.extend(x_imgs[:n_tr]);
        x_tr_l.extend([i] * n_tr)
        c_tr_p.extend([np.random.choice(c_imgs) for _ in range(n_tr)]);
        c_tr_l.extend([c_target_idx] * n_tr)

        n_va = min(len(x_imgs) - n_tr, va_n)
        x_va_p.extend(x_imgs[n_tr:n_tr + n_va]);
        x_va_l.extend([i] * n_va)
        c_va_p.extend([np.random.choice(c_imgs) for _ in range(n_va)]);
        c_va_l.extend([c_target_idx] * n_va)

    return (x_tr_p, c_tr_p, x_tr_l, c_tr_l), (x_va_p, c_va_p, x_va_l, c_va_l), x_classes, c_classes


def prepare_datasets(xtp, ctp, xtl, ctl, xvp, cvp, xvl, cvl, x_cls, c_cls):
    def map_fn(xp, cp, xl, cl):
        return (load_img(xp), load_img(cp)), (tf.one_hot(xl, len(x_cls)), tf.one_hot(cl, len(c_cls)))

    train_ds = tf.data.Dataset.from_tensor_slices((xtp, ctp, xtl, ctl)).map(map_fn).batch(16).shuffle(100)
    val_ds = tf.data.Dataset.from_tensor_slices((xvp, cvp, xvl, cvl)).map(map_fn).batch(16)
    return train_ds, val_ds


# =====================================================
# 4. FINAL ANALYTICS & VISUALIZATION
# =====================================================
def plot_final_results(history, d_true, d_pred, c_true, c_pred, x_cls, c_cls):
    """Handles reporting and visualization for the Focal Loss trained model."""

    # d_true and c_true are already sparse integer labels
    d_true_idx = d_true
    c_true_idx = c_true

    # Convert probability predictions to integer class indices
    d_pred_idx = np.argmax(d_pred, axis=1)
    c_pred_idx = np.argmax(c_pred, axis=1)

    # 1. Loss and Accuracy Curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

    # Plotting Total Loss (Sum of Focal Losses)
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Total Focal Loss History')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend();
    axes[0].grid(True)

    # Plotting Accuracy
    axes[1].plot(history.history['disease_out_accuracy'], label='X-ray Acc', color='blue')
    axes[1].plot(history.history['cancer_out_accuracy'], label='Cancer Acc', color='orange')
    axes[1].set_title('Accuracy History')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend();
    axes[1].grid(True)
    plt.show()

    # 2. High-Resolution Confusion Matrices
    plt.figure(figsize=(14, 5), dpi=300)

    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(d_true_idx, d_pred_idx), annot=True, fmt='g',
                cmap='Blues', xticklabels=x_cls, yticklabels=x_cls)
    plt.title('X-ray Confusion Matrix (Focal Loss + Weighted)')

    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(c_true_idx, c_pred_idx), annot=True, fmt='g',
                cmap='Oranges', xticklabels=c_cls, yticklabels=c_cls)
    plt.title('Cancer Confusion Matrix (Focal Loss + Weighted)')
    plt.show()

    # 3. Final Classification Reports
    print("\n" + "=" * 60)
    print("--- X-RAY DIAGNOSTIC REPORT (FOCAL LOSS OPTIMIZED) ---")
    print("=" * 60)
    print(classification_report(d_true_idx, d_pred_idx, target_names=x_cls))

    print("\n" + "=" * 60)
    print("--- CANCER DIAGNOSTIC REPORT (FOCAL LOSS OPTIMIZED) ---")
    print("=" * 60)
    print(classification_report(c_true_idx, c_pred_idx, target_names=c_cls))


class SEScale1D(layers.Layer):
    """
    Squeeze-and-Excitation block for 1D feature vectors.
    Recalibrates modality features based on global importance.
    """
    def __init__(self, reduction=16, **kwargs):
        super(SEScale1D, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channel = input_shape[-1]
        self.se_path = tf.keras.Sequential([
            layers.Dense(channel // self.reduction, activation='relu', use_bias=False),
            layers.Dense(channel, activation='sigmoid', use_bias=False)
        ])
        super(SEScale1D, self).build(input_shape)

    def call(self, inputs):
        scale = self.se_path(inputs)
        return inputs * scale


def main():
    # 1. Path Setup
    x_dir = '/content/drive/MyDrive/Kaggel_direct_download/AP_Frontal_CheXpert/1000_nos'
    c_dir = '/content/drive/MyDrive/Kaggel_direct_download/Cancer_Dataset/Lung_Cancer_LIDC_Y_Net/LIDC_Y-Net'

    # 2. Data Loading & Alignment
    (xtp, ctp, xtl, ctl), (xvp, cvp, xvl, cvl), x_cls, c_cls = get_final_aligned_data(x_dir, c_dir, 500, 150)

    # 3. Calculate Weights for Focal Loss
    x_w = class_weight.compute_class_weight('balanced', classes=np.unique(xtl), y=xtl)
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(ctl), y=ctl)

    # 4. Prepare Datasets
    train_ds, val_ds = prepare_datasets(xtp, ctp, xtl, ctl, xvp, cvp, xvl, cvl, x_cls, c_cls)

    # --- 5. MODEL ARCHITECTURE ---

    # A. Dual 2D Encoders (EfficientNetB0)
    x_in = layers.Input(shape=(224, 224, 3), name='xray_input')
    c_in = layers.Input(shape=(224, 224, 3), name='ct_input')

    base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    base._name = 'efficientnetb0'

    x_feat = base(x_in)  # X-ray features
    c_feat = base(c_in)  # CT features

    # B. Lightweight Multi-Modal Fusion
    # Concatenate CT and X-ray embeddings
    fused = layers.Concatenate(name='modality_concat')([x_feat, c_feat])

    # SE Attention Block (Lightweight recalibration)
    fused = SEScale1D(reduction=16, name='se_attention_block')(fused)

    # C. Graph-based Feature Optimization (Simplified GNN)
    # Project to the GNN embedding dimension
    fused_projected = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(fused)
    fused_projected = layers.Dropout(0.3)(fused_projected)

    # GAT Layer: Optimizes patient features using disease relationships
    opt = GATFeatureOptimizer(
        num_nodes=len(x_cls) + len(c_cls),
        embedding_dim=512,
        name='gat_feature_optimizer'
    )(fused_projected)

    # D. Classifier
    d_out = layers.Dense(len(x_cls), activation='softmax', name='disease_out')(opt)
    c_out = layers.Dense(len(c_cls), activation='softmax', name='cancer_out')(opt)

    model = Model(inputs=[x_in, c_in], outputs=[d_out, c_out])

    # 6. Compile with Focal Loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={
            'disease_out': CategoricalFocalLoss(gamma=2.0, alpha=x_w),
            'cancer_out': CategoricalFocalLoss(gamma=2.0, alpha=c_w)
        },
        metrics={'disease_out': 'accuracy', 'cancer_out': 'accuracy'}
    )

    # 7. Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
        ]
    )

    # 8. Evaluation & Analytics
    d_true_one_hot, c_true_one_hot, d_pred, c_pred = [], [], [], []
    for (xi, ci), (yd, yc) in val_ds:
        res = model.predict((xi, ci), verbose=0)
        d_true_one_hot.extend(yd.numpy())
        d_pred.extend(res[0])
        c_true_one_hot.extend(yc.numpy())
        c_pred.extend(res[1])

    d_true_sparse = np.argmax(np.array(d_true_one_hot), axis=1)
    c_true_sparse = np.argmax(np.array(c_true_one_hot), axis=1)
    d_pred_arr = np.array(d_pred)
    c_pred_arr = np.array(c_pred)

    # Final Analytics & Visuals
    plot_final_results(history, d_true_sparse, d_pred_arr, c_true_sparse, c_pred_arr, x_cls, c_cls)
    plot_precision_recall_curves(np.array(d_true_one_hot), d_pred_arr, np.array(c_true_one_hot), c_pred_arr, x_cls,
                                 c_cls)

    # # Grad-CAM for interpretability
    # sample_batch = next(iter(val_ds))
    # (sample_x, sample_c), _ = sample_batch
    # run_gradcam(model, [sample_x[0:1], sample_c[0:1]], x_cls)

    if __name__ == "__main__":
        main()