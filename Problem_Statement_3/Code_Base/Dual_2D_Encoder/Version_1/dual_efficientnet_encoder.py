
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DualEfficientNetEncoder(Model):
    """
    Dual 2D Encoder using EfficientNet-B0 for CheXpert dataset
    Produces low-dimensional feature vectors (256-512D)
    """

    def __init__(self, input_shape=(224, 224, 3),feature_dim=512, use_shared_weights=False, dropout_rate=0.3, name='dual_efficientnet_encoder'):
        """
        Initialize Dual EfficientNet Encoder

        Args:
            input_shape: Input image shape (height, width, channels)
            feature_dim: Output feature dimension (256, 384, or 512)
            use_shared_weights: Whether to share weights between encoders
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super(DualEfficientNetEncoder, self).__init__(name=name)

        self.input_shape_model = input_shape
        self.feature_dim = feature_dim
        self.use_shared_weights = use_shared_weights
        self.dropout_rate = dropout_rate

        # Build the dual encoder architecture
        self._build_encoders()
        self._build_feature_extractors()
        self._build_fusion_layer()

    def _build_encoders(self):
        """Build the two EfficientNet-B0 encoders"""

        # First encoder (Frontal view)
        self.encoder1 = EfficientNetB0(include_top=False, weights='imagenet',input_shape=self.input_shape_model, pooling='avg'  # Global average pooling
                                        )

        # Make encoder1 trainable for fine-tuning
        self.encoder1.trainable = True

        # Second encoder (Lateral view or augmented frontal)
        if self.use_shared_weights:
            # Share weights between encoders
            self.encoder2 = self.encoder1
        else:
            # Separate encoder with different weights
            self.encoder2 = EfficientNetB0(include_top=False, weights='imagenet', input_shape=self.input_shape_model, pooling='avg'
                                    )
            self.encoder2.trainable = True

    def _build_feature_extractors(self):
        """Build feature extraction layers"""

        # Feature extraction for encoder 1
        self.feature_extractor1 = keras.Sequential([
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate/2)
        ], name='feature_extractor1')

        # Feature extraction for encoder 2
        if self.use_shared_weights:
            self.feature_extractor2 = self.feature_extractor1
        else:
            self.feature_extractor2 = keras.Sequential([
                layers.Dense(1024, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.feature_dim, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(self.dropout_rate/2)
            ], name='feature_extractor2')

    def _build_fusion_layer(self):
        """Build feature fusion layer"""

        self.fusion_layer = keras.Sequential([
            layers.Dense(self.feature_dim * 2, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.feature_dim, activation='relu'),
            layers.BatchNormalization(),
            layers.L2Normalize(axis=1)  # L2 normalization for better feature representation
        ], name='fusion_layer')

    def call(self, inputs, training=None):
        """
        Forward pass of the dual encoder

        Args:
            inputs: List of two input tensors [input1, input2] or single tensor
            training: Whether in training mode

        Returns:
            Fused feature vector of specified dimension
        """

        if isinstance(inputs, list) and len(inputs) == 2:
            input1, input2 = inputs
        else:
            # If single input provided, use it for both encoders
            input1 = input2 = inputs

        # Extract features from both encoders (1280-dim from EfficientNet-B0)
        features1 = self.encoder1(input1, training=training)
        features2 = self.encoder2(input2, training=training)

        # Apply feature extraction layers
        extracted_features1 = self.feature_extractor1(features1, training=training)
        extracted_features2 = self.feature_extractor2(features2, training=training)

        # Concatenate features
        concatenated_features = layers.Concatenate()([extracted_features1, extracted_features2])

        # Apply fusion layer
        fused_features = self.fusion_layer(concatenated_features, training=training)

        return fused_features

    def get_individual_features(self, inputs, training=None):
        """
        Get features from individual encoders (useful for analysis)

        Args:
            inputs: List of two input tensors [input1, input2]
            training: Whether in training mode

        Returns:
            Tuple of (features1, features2, fused_features)
        """

        if isinstance(inputs, list) and len(inputs) == 2:
            input1, input2 = inputs
        else:
            input1 = input2 = inputs

        # Extract features from both encoders
        features1 = self.encoder1(input1, training=training)
        features2 = self.encoder2(input2, training=training)

        # Apply feature extraction layers
        extracted_features1 = self.feature_extractor1(features1, training=training)
        extracted_features2 = self.feature_extractor2(features2, training=training)

        # Get fused features
        fused_features = self.call(inputs, training=training)

        return extracted_features1, extracted_features2, fused_features


class CheXpertDataProcessor:
    """
    Data processor for CheXpert dataset
    """

    def __init__(self, data_dir, csv_file, image_size=(224, 224)):
        """
        Initialize data processor

        Args:
            data_dir: Directory containing CheXpert images
            csv_file: Path to CSV file with labels
            image_size: Target image size for resizing
        """
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.image_size = image_size

        # CheXpert pathology labels
        self.pathology_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

    def preprocess_image(self, image_path):
        """
        Preprocess a single image

        Args:
            image_path: Path to the image

        Returns:
            Preprocessed image tensor
        """
        # Read and decode image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Resize image
        image = tf.image.resize(image, self.image_size)

        # Normalize pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # EfficientNet expects values in [0, 255], so we'll multiply back
        image = image * 255.0

        return image

    def create_data_generator(self, df, batch_size=32, shuffle=True, dual_input=True):
        """
        Create data generator for training

        Args:
            df: DataFrame with image paths and labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            dual_input: Whether to create dual inputs (for dual encoder)

        Returns:
            tf.data.Dataset
        """

        def generator():
            indices = np.arange(len(df))
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                row = df.iloc[idx]
                image_path = os.path.join(self.data_dir, row['Path'])

                # Load and preprocess image
                image = self.preprocess_image(image_path)

                # Extract labels
                labels = row[self.pathology_labels].values.astype(np.float32)

                if dual_input:
                    # For dual input, we can use the same image for both inputs
                    # or apply different augmentations
                    image_augmented = self.augment_image(image)
                    yield ([image, image_augmented], labels)
                else:
                    yield (image, labels)

        # Create dataset
        if dual_input:
            output_signature = (
                (tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
                 tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32)),
                tf.TensorSpec(shape=(len(self.pathology_labels),), dtype=tf.float32)
            )
        else:
            output_signature = (
                tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(len(self.pathology_labels),), dtype=tf.float32)
            )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def augment_image(self, image):
        """
        Apply data augmentation to image

        Args:
            image: Input image tensor

        Returns:
            Augmented image tensor
        """
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)

        # Random rotation
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Ensure values stay in valid range
        image = tf.clip_by_value(image, 0.0, 255.0)

        return image


def build_classification_model(dual_encoder, num_classes=14):
    """
    Build classification model on top of dual encoder

    Args:
        dual_encoder: Trained dual encoder model
        num_classes: Number of output classes

    Returns:
        Classification model
    """

    # Freeze the dual encoder
    dual_encoder.trainable = False

    # Input layers
    input1 = layers.Input(shape=(224, 224, 3), name='input1')
    input2 = layers.Input(shape=(224, 224, 3), name='input2')

    # Extract features using dual encoder
    features = dual_encoder([input1, input2])

    # Classification head
    x = layers.Dense(256, activation='relu')(features)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output layer (sigmoid for multi-label classification)
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=[input1, input2], outputs=outputs, name='chexpert_classifier')

    return model


# Training function
def train_dual_encoder(model, train_dataset, val_dataset, epochs=10, learning_rate=1e-4):
    """
    Train the dual encoder model

    Args:
        model: Dual encoder model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Training history
    """

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',  # Multi-label classification
        metrics=['accuracy', 'precision', 'recall']
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'dual_encoder_chexpert.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
    ]

    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    return history


# Example usage and testing
def example_usage():
    """
    Example of how to use the Dual EfficientNet Encoder
    """

    print("=== Dual EfficientNet Encoder for CheXpert ===\n")

    # 1. Create dual encoder with different feature dimensions
    print("1. Creating Dual Encoder Models...")

    # 256-dimensional features
    dual_encoder_256 = DualEfficientNetEncoder(input_shape=(224, 224, 3),feature_dim=256,use_shared_weights=False, dropout_rate=0.3)

    # 512-dimensional features
    dual_encoder_512 = DualEfficientNetEncoder(input_shape=(224, 224, 3),feature_dim=512,use_shared_weights=False, dropout_rate=0.3)

    print(f"✓ Created 256D dual encoder")
    print(f"✓ Created 512D dual encoder")

    # 2. Test with dummy data
    print("\n2. Testing with dummy data...")

    # Create dummy input data (batch of 2 images)
    dummy_input1 = tf.random.normal((2, 224, 224, 3)) * 255.0
    dummy_input2 = tf.random.normal((2, 224, 224, 3)) * 255.0

    # Test 256D encoder
    features_256 = dual_encoder_256([dummy_input1, dummy_input2])
    print(f"✓ 256D encoder output shape: {features_256.shape}")

    # Test 512D encoder
    features_512 = dual_encoder_512([dummy_input1, dummy_input2])
    print(f"✓ 512D encoder output shape: {features_512.shape}")

    # 3. Test individual feature extraction
    print("\n3. Testing individual feature extraction...")

    feat1, feat2, fused = dual_encoder_512.get_individual_features([dummy_input1, dummy_input2])
    print(f"✓ Individual features shape: {feat1.shape}, {feat2.shape}")
    print(f"✓ Fused features shape: {fused.shape}")

    # 4. Build classification model
    print("\n4. Building classification model...")

    classifier = build_classification_model(dual_encoder_512, num_classes=14)
    print(f"✓ Classification model created")
    print(f"✓ Total parameters: {classifier.count_params():,}")

    # 5. Model summary
    print("\n5. Model Architecture Summary:")
    print("\n--- Dual Encoder (512D) ---")
    print(f"Encoder 1: EfficientNet-B0 (1280 features) -> Dense(1024) -> Dense(512)")
    print(f"Encoder 2: EfficientNet-B0 (1280 features) -> Dense(1024) -> Dense(512)")
    print(f"Fusion: Concat(512+512) -> Dense(1024) -> Dense(512) -> L2Norm")
    print(f"Final output: 512-dimensional feature vector")

    return dual_encoder_256, dual_encoder_512, classifier

# Run example
if __name__ == "__main__":
    dual_encoder_256, dual_encoder_512, classifier = example_usage()
