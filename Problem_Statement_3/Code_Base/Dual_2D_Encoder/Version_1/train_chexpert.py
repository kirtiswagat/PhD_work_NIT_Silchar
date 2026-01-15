
# ===============================
# CheXpert Training Script
# ===============================

import tensorflow as tf
from dual_efficientnet_encoder import DualEfficientNetEncoder, CheXpertDataProcessor, train_dual_encoder, build_classification_model
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'data_dir': 'path/to/chexpert/images',  # Update with your CheXpert data directory
    'train_csv': 'path/to/chexpert/train.csv',  # Update with your train CSV file
    'valid_csv': 'path/to/chexpert/valid.csv',  # Update with your validation CSV file
    'image_size': (224, 224),
    'batch_size': 16,  # Adjust based on your GPU memory
    'learning_rate': 1e-4,
    'epochs': 50,
    'feature_dim': 512,  # 256, 384, or 512
    'use_shared_weights': False,
    'dropout_rate': 0.3
}

def prepare_chexpert_data():
    """
    Prepare CheXpert dataset for training
    """
    print("Preparing CheXpert dataset...")

    # Load CSV files
    train_df = pd.read_csv(CONFIG['train_csv'])
    valid_df = pd.read_csv(CONFIG['valid_csv'])

    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(valid_df)}")

    # Handle uncertain labels (CheXpert specific)
    # Convert uncertain (-1) to positive (1) or negative (0) based on your strategy
    pathology_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # Replace NaN with 0 (negative) and uncertain (-1) with 0
    for col in pathology_columns:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            train_df[col] = train_df[col].replace(-1, 0)  # Convert uncertain to negative

            valid_df[col] = valid_df[col].fillna(0)
            valid_df[col] = valid_df[col].replace(-1, 0)

    # Filter only frontal view images for this example
    train_df = train_df[train_df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
    valid_df = valid_df[valid_df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

    print(f"✓ Frontal view training samples: {len(train_df)}")
    print(f"✓ Frontal view validation samples: {len(valid_df)}")

    return train_df, valid_df

def create_model():
    """
    Create the dual encoder model
    """
    print(f"Creating Dual EfficientNet Encoder with {CONFIG['feature_dim']}D features...")

    # Create dual encoder
    dual_encoder = DualEfficientNetEncoder(
        input_shape=(*CONFIG['image_size'], 3),
        feature_dim=CONFIG['feature_dim'],
        use_shared_weights=CONFIG['use_shared_weights'],
        dropout_rate=CONFIG['dropout_rate']
    )

    # Build classification model
    classifier = build_classification_model(dual_encoder, num_classes=14)

    print(f"✓ Model created with {classifier.count_params():,} parameters")

    return dual_encoder, classifier

def train_model():
    """
    Complete training pipeline
    """
    print("="*60)
    print("CHEXPERT DUAL ENCODER TRAINING")
    print("="*60)

    # 1. Prepare data
    try:
        train_df, valid_df = prepare_chexpert_data()
    except FileNotFoundError:
        print("⚠️  CheXpert dataset files not found!")
        print("Please update the CONFIG paths with your CheXpert dataset location.")
        print("For demonstration, creating synthetic data...")
        train_df, valid_df = create_synthetic_data()

    # 2. Create data processor
    data_processor = CheXpertDataProcessor(
        data_dir=CONFIG['data_dir'],
        csv_file=CONFIG['train_csv'],
        image_size=CONFIG['image_size']
    )

    # 3. Create data generators
    print("\nCreating data generators...")

    # For demonstration with synthetic data
    if 'synthetic' in locals():
        train_dataset = create_synthetic_dataset(len(train_df), CONFIG['batch_size'])
        valid_dataset = create_synthetic_dataset(len(valid_df), CONFIG['batch_size'])
    else:
        train_dataset = data_processor.create_data_generator(
            train_df, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            dual_input=True
        )
        valid_dataset = data_processor.create_data_generator(
            valid_df, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            dual_input=True
        )

    print("✓ Data generators created")

    # 4. Create model
    dual_encoder, classifier = create_model()

    # 5. Train model
    print("\nStarting training...")

    history = train_dual_encoder(
        model=classifier,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate']
    )

    # 6. Save final model
    print("\nSaving models...")
    dual_encoder.save_weights('dual_encoder_weights.h5')
    classifier.save('chexpert_classifier_complete.h5')
    print("✓ Models saved!")

    # 7. Plot training history
    plot_training_history(history)

    return dual_encoder, classifier, history

def create_synthetic_data():
    """
    Create synthetic data for demonstration when real CheXpert data is not available
    """
    print("Creating synthetic data for demonstration...")

    # Create synthetic training data
    train_data = {
        'Path': [f'synthetic_train_image_{i}.jpg' for i in range(1000)],
        'Frontal/Lateral': ['Frontal'] * 1000
    }

    # Add pathology labels (random for demonstration)
    pathology_columns = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    np.random.seed(42)
    for col in pathology_columns:
        train_data[col] = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])

    train_df = pd.DataFrame(train_data)

    # Create synthetic validation data
    valid_data = {
        'Path': [f'synthetic_valid_image_{i}.jpg' for i in range(200)],
        'Frontal/Lateral': ['Frontal'] * 200
    }

    for col in pathology_columns:
        valid_data[col] = np.random.choice([0, 1], size=200, p=[0.8, 0.2])

    valid_df = pd.DataFrame(valid_data)

    return train_df, valid_df

def create_synthetic_dataset(num_samples, batch_size):
    """
    Create synthetic dataset for demonstration
    """
    def generator():
        for _ in range(num_samples):
            # Generate synthetic chest X-ray images
            image1 = tf.random.normal((224, 224, 3)) * 255.0
            image2 = tf.random.normal((224, 224, 3)) * 255.0

            # Generate synthetic labels
            labels = tf.random.uniform((14,)) > 0.8
            labels = tf.cast(labels, tf.float32)

            yield ([image1, image2], labels)

    output_signature = (
        (tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(14,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✓ Training history plot saved as 'training_history.png'")

def feature_extraction_example(dual_encoder):
    """
    Example of using the trained dual encoder for feature extraction
    """
    print("\n" + "="*60)
    print("FEATURE EXTRACTION EXAMPLE")
    print("="*60)

    # Create sample images
    sample_image1 = tf.random.normal((5, 224, 224, 3)) * 255.0  # Batch of 5 images
    sample_image2 = tf.random.normal((5, 224, 224, 3)) * 255.0

    # Extract features
    features = dual_encoder([sample_image1, sample_image2])

    print(f"✓ Input images shape: {sample_image1.shape}")
    print(f"✓ Extracted features shape: {features.shape}")
    print(f"✓ Feature dimension: {features.shape[-1]}D")

    # Demonstrate feature properties
    print(f"✓ Feature vector L2 norm (should be ~1.0): {tf.norm(features[0]).numpy():.3f}")
    print(f"✓ Feature statistics:")
    print(f"  - Mean: {tf.reduce_mean(features).numpy():.3f}")
    print(f"  - Std:  {tf.math.reduce_std(features).numpy():.3f}")
    print(f"  - Min:  {tf.reduce_min(features).numpy():.3f}")
    print(f"  - Max:  {tf.reduce_max(features).numpy():.3f}")

    return features

if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Train the model
    dual_encoder, classifier, history = train_model()

    # Demonstrate feature extraction
    features = feature_extraction_example(dual_encoder)

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("📁 Files created:")
    print("  - dual_encoder_weights.h5 (encoder weights)")
    print("  - chexpert_classifier_complete.h5 (complete model)")
    print("  - training_history.png (training plots)")
    print("\n🚀 Your dual encoder is ready for feature extraction!")
