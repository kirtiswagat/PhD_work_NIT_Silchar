
# ===============================
# Quick Start Example
# ===============================

import tensorflow as tf
from dual_efficientnet_encoder import DualEfficientNetEncoder, build_classification_model
import numpy as np

# 1. Create the Dual Encoder
print("Creating Dual EfficientNet Encoder...")

# For 512-dimensional features (recommended)
dual_encoder = DualEfficientNetEncoder(
    input_shape=(224, 224, 3),
    feature_dim=512,              # Can be 256, 384, or 512
    use_shared_weights=False,     # Set True to share weights between encoders
    dropout_rate=0.3
)

print(f"✓ Dual encoder created with {dual_encoder.feature_dim}D output")

# 2. Test with sample data
print("\nTesting with sample chest X-ray images...")

# Simulate two chest X-ray images (frontal and lateral views)
frontal_view = tf.random.normal((1, 224, 224, 3)) * 255.0    # Batch of 1 image
lateral_view = tf.random.normal((1, 224, 224, 3)) * 255.0    # Batch of 1 image

# Extract features
features = dual_encoder([frontal_view, lateral_view])
print(f"✓ Extracted features shape: {features.shape}")
print(f"✓ Feature vector dimension: {features.shape[-1]}D")

# 3. Get individual encoder features (for analysis)
feat1, feat2, fused = dual_encoder.get_individual_features([frontal_view, lateral_view])
print(f"\nIndividual encoder outputs:")
print(f"✓ Encoder 1 features: {feat1.shape}")
print(f"✓ Encoder 2 features: {feat2.shape}")
print(f"✓ Fused features: {fused.shape}")

# 4. Build complete classification model for CheXpert
print("\nBuilding complete classification model...")

classifier = build_classification_model(dual_encoder, num_classes=14)
print(f"✓ Complete model parameters: {classifier.count_params():,}")

# 5. Model architecture summary
print("\n" + "="*50)
print("DUAL ENCODER ARCHITECTURE")
print("="*50)
print("Input: Two 224x224x3 chest X-ray images")
print("│")
print("├── Encoder 1: EfficientNet-B0")
print("│   ├── Backbone: 1280 features")  
print("│   ├── Dense(1024) + ReLU + BatchNorm + Dropout")
print("│   └── Dense(512) + ReLU + BatchNorm + Dropout")
print("│")
print("├── Encoder 2: EfficientNet-B0") 
print("│   ├── Backbone: 1280 features")
print("│   ├── Dense(1024) + ReLU + BatchNorm + Dropout")
print("│   └── Dense(512) + ReLU + BatchNorm + Dropout")
print("│")
print("├── Feature Fusion:")
print("│   ├── Concatenate: [512 + 512] = 1024")
print("│   ├── Dense(1024) + ReLU + BatchNorm + Dropout")
print("│   ├── Dense(512) + ReLU + BatchNorm")
print("│   └── L2 Normalization")
print("│")
print(f"Output: {dual_encoder.feature_dim}-dimensional feature vector")
print("="*50)
