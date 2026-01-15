
# ===============================
# Feature Extraction Utilities
# ===============================

import tensorflow as tf
import numpy as np
from dual_efficientnet_encoder import DualEfficientNetEncoder
import cv2
import os
from pathlib import Path

class CheXpertFeatureExtractor:
    """
    Utility class for extracting features from chest X-ray images
    using the trained Dual EfficientNet Encoder
    """

    def __init__(self, model_path=None, feature_dim=512):
        """
        Initialize feature extractor

        Args:
            model_path: Path to trained model weights
            feature_dim: Feature dimension (256, 384, or 512)
        """
        self.feature_dim = feature_dim

        # Create dual encoder
        self.dual_encoder = DualEfficientNetEncoder(
            input_shape=(224, 224, 3),
            feature_dim=feature_dim,
            use_shared_weights=False,
            dropout_rate=0.0  # No dropout during inference
        )

        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.dual_encoder.load_weights(model_path)
            print(f"✓ Loaded model weights from {model_path}")
        else:
            print("⚠️  No pre-trained weights loaded. Using random initialization.")

    def preprocess_image(self, image_path):
        """
        Preprocess image for feature extraction

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # Assume it's already a numpy array

        # Resize to target size
        image = cv2.resize(image, (224, 224))

        # Convert to float32 and normalize
        image = image.astype(np.float32)

        # EfficientNet expects values in [0, 255]
        if image.max() <= 1.0:
            image = image * 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return tf.convert_to_tensor(image)

    def extract_features_single(self, image_path1, image_path2=None):
        """
        Extract features from a single image or pair of images

        Args:
            image_path1: Path to first image (e.g., frontal view)
            image_path2: Path to second image (e.g., lateral view). If None, uses image_path1

        Returns:
            Feature vector of shape (1, feature_dim)
        """
        # Preprocess images
        image1 = self.preprocess_image(image_path1)
        image2 = self.preprocess_image(image_path2 if image_path2 else image_path1)

        # Extract features
        features = self.dual_encoder([image1, image2], training=False)

        return features.numpy()

    def extract_features_batch(self, image_paths1, image_paths2=None, batch_size=32):
        """
        Extract features from batch of images

        Args:
            image_paths1: List of paths to first images
            image_paths2: List of paths to second images. If None, uses image_paths1
            batch_size: Batch size for processing

        Returns:
            Feature matrix of shape (num_images, feature_dim)
        """
        if image_paths2 is None:
            image_paths2 = image_paths1

        if len(image_paths1) != len(image_paths2):
            raise ValueError("Number of image paths must match")

        num_images = len(image_paths1)
        all_features = []

        print(f"Extracting features from {num_images} images...")

        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)
            batch_paths1 = image_paths1[i:batch_end]
            batch_paths2 = image_paths2[i:batch_end]

            # Preprocess batch
            batch_images1 = []
            batch_images2 = []

            for path1, path2 in zip(batch_paths1, batch_paths2):
                try:
                    img1 = self.preprocess_image(path1)
                    img2 = self.preprocess_image(path2)
                    batch_images1.append(img1[0])  # Remove batch dimension
                    batch_images2.append(img2[0])
                except Exception as e:
                    print(f"Error processing {path1}: {e}")
                    continue

            if not batch_images1:
                continue

            # Stack into batch tensors
            batch_tensor1 = tf.stack(batch_images1)
            batch_tensor2 = tf.stack(batch_images2)

            # Extract features
            batch_features = self.dual_encoder([batch_tensor1, batch_tensor2], training=False)
            all_features.append(batch_features.numpy())

            print(f"✓ Processed {batch_end}/{num_images} images")

        # Concatenate all features
        if all_features:
            return np.concatenate(all_features, axis=0)
        else:
            return np.empty((0, self.feature_dim))

    def extract_features_from_directory(self, image_dir, output_file=None):
        """
        Extract features from all images in a directory

        Args:
            image_dir: Directory containing images
            output_file: Path to save features (optional)

        Returns:
            Dictionary with image names as keys and features as values
        """
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_dir = Path(image_dir)
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))

        image_paths = sorted([str(p) for p in image_paths])

        if not image_paths:
            print(f"No images found in {image_dir}")
            return {}

        print(f"Found {len(image_paths)} images in {image_dir}")

        # Extract features
        features = self.extract_features_batch(image_paths)

        # Create result dictionary
        result = {}
        for i, path in enumerate(image_paths):
            if i < len(features):
                result[os.path.basename(path)] = features[i]

        # Save to file if requested
        if output_file:
            np.savez_compressed(output_file, **result)
            print(f"✓ Features saved to {output_file}")

        return result

    def similarity_search(self, query_features, database_features, top_k=5):
        """
        Find most similar images based on feature similarity

        Args:
            query_features: Query feature vector (1, feature_dim)
            database_features: Database feature matrix (N, feature_dim)
            top_k: Number of top similar images to return

        Returns:
            Indices and similarities of top-k most similar images
        """
        # Ensure query is 2D
        if len(query_features.shape) == 1:
            query_features = query_features.reshape(1, -1)

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_features, axis=1, keepdims=True)
        db_norm = np.linalg.norm(database_features, axis=1, keepdims=True)

        similarities = np.dot(query_features, database_features.T) / (query_norm * db_norm.T)
        similarities = similarities.flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]

        return top_indices, top_similarities

# Example usage functions
def demo_feature_extraction():
    """
    Demonstrate feature extraction capabilities
    """
    print("="*60)
    print("CHEXPERT FEATURE EXTRACTION DEMO")
    print("="*60)

    # Initialize feature extractor
    extractor = CheXpertFeatureExtractor(feature_dim=512)

    # Create sample images for demonstration
    print("\n1. Creating sample images...")
    sample_images = []
    for i in range(3):
        # Create synthetic chest X-ray-like images
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        # Add some structure (simulate lung regions)
        img[50:174, 50:174] = img[50:174, 50:174] * 0.7
        sample_images.append(img)

    print(f"✓ Created {len(sample_images)} sample images")

    # Extract features from single image
    print("\n2. Single image feature extraction...")
    features_single = extractor.extract_features_single(sample_images[0])
    print(f"✓ Features shape: {features_single.shape}")
    print(f"✓ Feature norm: {np.linalg.norm(features_single):.3f}")

    # Extract features from multiple images
    print("\n3. Batch feature extraction...")
    features_batch = extractor.extract_features_batch(sample_images)
    print(f"✓ Batch features shape: {features_batch.shape}")

    # Demonstrate similarity search
    print("\n4. Similarity search demo...")
    query_features = features_batch[0:1]  # Use first image as query
    indices, similarities = extractor.similarity_search(
        query_features, features_batch, top_k=3
    )

    print("Top similar images:")
    for i, (idx, sim) in enumerate(zip(indices, similarities)):
        print(f"  {i+1}. Image {idx}: similarity = {sim:.3f}")

    print("\n✅ Demo completed successfully!")

    return extractor, features_batch

if __name__ == "__main__":
    # Run demonstration
    extractor, features = demo_feature_extraction()

    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("1. Train your model using train_chexpert.py")
    print("2. Load the trained weights:")
    print("   extractor = CheXpertFeatureExtractor('dual_encoder_weights.h5')")
    print("3. Extract features from images:")
    print("   features = extractor.extract_features_single('image.jpg')")
    print("4. Use features for downstream tasks:")
    print("   - Image retrieval")
    print("   - Classification")
    print("   - Clustering")
    print("   - Similarity search")
