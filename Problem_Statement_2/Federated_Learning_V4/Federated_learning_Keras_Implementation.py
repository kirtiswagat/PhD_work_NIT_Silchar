import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
class Config:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_BINARY = 10
    EPOCHS_MULTI = 15
    VAL_SPLIT = 0.2
    TRAIN_LIMIT = 500
    TEST_LIMIT = 200

    # Directories
    BASE_DIR = "/content/chexpert_dataset"
    TRAIN_DIR = "/content/chexpert_dataset/train"
    TEST_DIR = "/content/chexpert_dataset/test"
    RESULTS_DIR = "/content/drive/MyDrive/Colab_Datasets/federated_models_keras_implementation/results"

    # Classes
    PATHOLOGY_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural_Effusion"]
    ALL_CLASSES = ["No_Finding"] + PATHOLOGY_CLASSES

    def __init__(self):
        self.setup_directories()
        self.setup_gpu()

    def setup_directories(self):
        self.BINARY_MODEL_DIR = os.path.join(self.RESULTS_DIR, "binary_models")
        self.MULTICLASS_MODEL_DIR = os.path.join(self.RESULTS_DIR, "multiclass_model")
        self.VISUALIZATION_DIR = os.path.join(self.RESULTS_DIR, "visualizations")

        for dir_path in [self.BINARY_MODEL_DIR, self.MULTICLASS_MODEL_DIR, self.VISUALIZATION_DIR]:
            os.makedirs(dir_path, exist_ok=True)

    def setup_gpu(self):
        physical_gpus = tf.config.list_physical_devices('GPU')
        if physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(physical_gpus[0], True)
                print("GPU available and configured for growth.")
            except Exception as e:
                print("Error setting up GPU:", e)
        else:
            print("No GPU found. Running on CPU.")

# ===== DATA HANDLING =====
class DataHandler:
    def __init__(self, config):
        self.config = config

    def get_image_label_arrays(self, root_dir, class_list, max_per_class):
        """Load image paths and labels for specified classes"""
        images, labels = [], []

        for idx, class_name in enumerate(class_list):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue

            # Get image files
            files = [f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files = sorted([os.path.join(class_dir, f) for f in files])

            # Limit number of files if specified
            if max_per_class and len(files) > max_per_class:
                files = files[:max_per_class]

            images.extend(files)
            labels.extend([idx] * len(files))

            print(f"Loaded {len(files)} images for class {class_name}")

        return np.array(images), np.array(labels)

    def create_dataset(self, image_paths, labels, shuffle=True, augment=False):
        """Create TensorFlow dataset from image paths and labels"""
        if len(image_paths) == 0:
            print("Warning: Empty image list")
            return None

        def parse_function(path, label):
            # Read and decode image
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.config.IMG_SIZE)
            image = tf.cast(image, tf.float32) / 255.0

            # Apply augmentation if specified
            if augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.1)
                image = tf.image.random_contrast(image, 0.9, 1.1)

            return image, label

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        if shuffle:
            dataset = dataset.shuffle(len(image_paths))

        dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def prepare_binary_data(self, disease):
        """Prepare data for binary classification (No_Finding vs Disease)"""
        binary_classes = ["No_Finding", disease]

        # Load training data
        train_imgs, train_lbls = self.get_image_label_arrays(
            self.config.TRAIN_DIR, binary_classes, self.config.TRAIN_LIMIT
        )

        # Load test data
        test_imgs, test_lbls = self.get_image_label_arrays(
            self.config.TEST_DIR, binary_classes, self.config.TEST_LIMIT
        )

        if len(train_imgs) == 0 or len(test_imgs) == 0:
            print(f"Insufficient data for {disease}")
            return None

        # Split training data into train/validation
        indices = np.random.permutation(len(train_imgs))
        val_split_idx = int(len(indices) * self.config.VAL_SPLIT)

        val_indices = indices[:val_split_idx]
        train_indices = indices[val_split_idx:]

        # Create datasets
        train_ds = self.create_dataset(
            train_imgs[train_indices], train_lbls[train_indices],
            shuffle=True, augment=True
        )
        val_ds = self.create_dataset(
            train_imgs[val_indices], train_lbls[val_indices],
            shuffle=False, augment=False
        )
        test_ds = self.create_dataset(
            test_imgs, test_lbls, shuffle=False, augment=False
        )

        return {
            'train': train_ds,
            'val': val_ds,
            'test': test_ds,
            'classes': binary_classes
        }

    def prepare_multiclass_data(self):
        """Prepare data for multi-class classification"""
        # Load training data
        train_imgs, train_lbls = self.get_image_label_arrays(
            self.config.TRAIN_DIR, self.config.ALL_CLASSES, self.config.TRAIN_LIMIT
        )

        # Load test data
        test_imgs, test_lbls = self.get_image_label_arrays(
            self.config.TEST_DIR, self.config.ALL_CLASSES, self.config.TEST_LIMIT
        )

        if len(train_imgs) == 0 or len(test_imgs) == 0:
            print("Insufficient data for multi-class")
            return None

        # Split training data
        indices = np.random.permutation(len(train_imgs))
        val_split_idx = int(len(indices) * self.config.VAL_SPLIT)

        val_indices = indices[:val_split_idx]
        train_indices = indices[val_split_idx:]

        # Create datasets
        train_ds = self.create_dataset(
            train_imgs[train_indices], train_lbls[train_indices],
            shuffle=True, augment=True
        )
        val_ds = self.create_dataset(
            train_imgs[val_indices], train_lbls[val_indices],
            shuffle=False, augment=False
        )
        test_ds = self.create_dataset(
            test_imgs, test_lbls, shuffle=False, augment=False
        )

        return {
            'train': train_ds,
            'val': val_ds,
            'test': test_ds,
            'classes': self.config.ALL_CLASSES
        }

# ===== MODEL UTILITIES =====
class ModelUtils:
    @staticmethod
    def create_mobilenet_model(num_classes, use_pretrained=True):
        """Create MobileNetV2-based model"""
        weights = 'imagenet' if use_pretrained else None

        # Create base model
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights=weights
        )
        base_model.trainable = False

        # Add custom head
        inputs = keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model

    @staticmethod
    def get_backbone_weights(model):
        """Extract backbone weights from model"""
        # Find MobileNetV2 layer
        mobilenet_layer = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                mobilenet_layer = layer
                break

        if mobilenet_layer is None:
            raise ValueError("MobileNetV2 layer not found")

        # Get weights from layers that have parameters
        backbone_weights = []
        for layer in mobilenet_layer.layers:
            if len(layer.get_weights()) > 0:
                backbone_weights.append(layer.get_weights())

        return backbone_weights

    @staticmethod
    def set_backbone_weights(model, backbone_weights):
        """Set backbone weights in model"""
        # Find MobileNetV2 layer
        mobilenet_layer = None
        for layer in model.layers:
            if 'mobilenetv2' in layer.name.lower():
                mobilenet_layer = layer
                break

        if mobilenet_layer is None:
            raise ValueError("MobileNetV2 layer not found")

        # Set weights for layers that have parameters
        weight_idx = 0
        for layer in mobilenet_layer.layers:
            if len(layer.get_weights()) > 0:
                if weight_idx < len(backbone_weights):
                    try:
                        layer.set_weights(backbone_weights[weight_idx])
                        weight_idx += 1
                    except Exception as e:
                        print(f"Warning: Could not set weights for layer {layer.name}: {e}")

# ===== METRICS AND EVALUATION =====
class SimpleMetrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate basic metrics safely"""
        metrics = {}

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        return metrics

    @staticmethod
    def evaluate_model(model, dataset, class_names):
        """Evaluate model on dataset"""
        all_true = []
        all_pred = []
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in dataset:
            # Get predictions
            predictions = model.predict(batch_x, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)

            # Calculate loss
            loss = model.compiled_loss(batch_y, predictions)
            total_loss += loss
            num_batches += 1

            # Store predictions
            all_true.extend(batch_y.numpy())
            all_pred.extend(pred_classes)

        # Calculate metrics
        metrics = SimpleMetrics.calculate_metrics(all_true, all_pred)
        metrics['loss'] = float(total_loss / num_batches) if num_batches > 0 else 0.0

        return metrics, all_true, all_pred

# ===== VISUALIZATION =====
class Visualizer:
    @staticmethod
    def plot_training_history(history, title, save_path):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot accuracy
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_results(metrics, save_dir, filename):
        """Save metrics to JSON file"""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, filename), 'w') as f:
            json.dump(metrics, f, indent=2)

# ===== TRAINING CLASSES =====
class BinaryTrainer:
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.trained_models = {}

    def train_binary_model(self, disease):
        """Train binary classifier for specific disease"""
        print(f"\n{'='*50}")
        print(f"Training Binary Model: No_Finding vs {disease}")
        print(f"{'='*50}")

        # Prepare data
        data = self.data_handler.prepare_binary_data(disease)
        if data is None:
            print(f"Skipping {disease} - insufficient data")
            return None

        # Create model
        model = ModelUtils.create_mobilenet_model(num_classes=2)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Training
        history = model.fit(
            data['train'],
            validation_data=data['val'],
            epochs=self.config.EPOCHS_BINARY,
            verbose=1
        )

        # Evaluate on test set
        test_metrics, y_true, y_pred = SimpleMetrics.evaluate_model(
            model, data['test'], data['classes']
        )

        # Save model
        model_path = os.path.join(self.config.BINARY_MODEL_DIR, f"{disease}_binary.h5")
        model.save(model_path)

        # Save visualizations
        vis_dir = os.path.join(self.config.VISUALIZATION_DIR, disease)
        os.makedirs(vis_dir, exist_ok=True)

        Visualizer.plot_training_history(
            history.history,
            f"{disease} Binary Classification",
            os.path.join(vis_dir, "training_history.png")
        )

        Visualizer.save_results(
            test_metrics, vis_dir, "test_metrics.json"
        )

        # Print results
        print(f"\n{disease} Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1 Score: {test_metrics['f1']:.3f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")

        return {
            'model': model,
            'history': history.history,
            'test_metrics': test_metrics,
            'model_path': model_path
        }

    def train_all_binary_models(self):
        """Train binary models for all pathology classes"""
        results = {}

        for disease in self.config.PATHOLOGY_CLASSES:
            result = self.train_binary_model(disease)
            if result is not None:
                results[disease] = result
                # Clear memory
                keras.backend.clear_session()

        self.trained_models = results
        return results

class FederatedAggregator:
    @staticmethod
    def aggregate_backbone_weights(model_paths):
        """Aggregate backbone weights from multiple binary models"""
        if not model_paths:
            print("No models to aggregate")
            return None

        print(f"\nAggregating weights from {len(model_paths)} models...")

        all_backbone_weights = []

        # Load all models and extract backbone weights
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = keras.models.load_model(path)
                    backbone_weights = ModelUtils.get_backbone_weights(model)
                    all_backbone_weights.append(backbone_weights)
                    keras.backend.clear_session()
                except Exception as e:
                    print(f"Error loading model {path}: {e}")
            else:
                print(f"Model not found: {path}")

        if not all_backbone_weights:
            print("No valid models found for aggregation")
            return None

        # Average the weights
        aggregated_weights = []
        num_layers = len(all_backbone_weights[0])

        for layer_idx in range(num_layers):
            layer_weights = []
            num_weight_tensors = len(all_backbone_weights[0][layer_idx])

            for weight_idx in range(num_weight_tensors):
                # Collect this weight tensor from all models
                weight_arrays = [
                    model_weights[layer_idx][weight_idx]
                    for model_weights in all_backbone_weights
                ]
                # Average them
                avg_weight = np.mean(weight_arrays, axis=0)
                layer_weights.append(avg_weight)

            aggregated_weights.append(layer_weights)

        print(f"Successfully aggregated weights from {len(all_backbone_weights)} models")
        return aggregated_weights

class MultiClassTrainer:
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler

    def train_multiclass_model(self, aggregated_weights=None):
        """Train multi-class classifier with optional aggregated weights"""
        print(f"\n{'='*50}")
        print("Training Multi-Class Model")
        print(f"{'='*50}")

        # Prepare data
        data = self.data_handler.prepare_multiclass_data()
        if data is None:
            print("Insufficient data for multi-class training")
            return None

        # Create model
        model = ModelUtils.create_mobilenet_model(
            num_classes=len(self.config.ALL_CLASSES),
            use_pretrained=(aggregated_weights is None)
        )

        # Set aggregated weights if available
        if aggregated_weights is not None:
            try:
                ModelUtils.set_backbone_weights(model, aggregated_weights)
                print("Successfully loaded aggregated backbone weights")
            except Exception as e:
                print(f"Error setting aggregated weights: {e}")
                print("Continuing with pretrained weights")

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Training
        history = model.fit(
            data['train'],
            validation_data=data['val'],
            epochs=self.config.EPOCHS_MULTI,
            verbose=1
        )

        # Evaluate on test set
        test_metrics, y_true, y_pred = SimpleMetrics.evaluate_model(
            model, data['test'], data['classes']
        )

        # Save model
        model_path = os.path.join(self.config.MULTICLASS_MODEL_DIR, "multiclass_federated.h5")
        model.save(model_path)

        # Save visualizations
        Visualizer.plot_training_history(
            history.history,
            "Multi-Class Federated Learning",
            os.path.join(self.config.MULTICLASS_MODEL_DIR, "training_history.png")
        )

        Visualizer.save_results(
            test_metrics, self.config.MULTICLASS_MODEL_DIR, "test_metrics.json"
        )

        # Print results
        print(f"\nMulti-Class Results:")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1 Score: {test_metrics['f1']:.3f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")

        return {
            'model': model,
            'history': history.history,
            'test_metrics': test_metrics,
            'model_path': model_path
        }

# ===== MAIN PIPELINE =====
class FederatedLearningPipeline:
    def __init__(self):
        self.config = Config()
        self.data_handler = DataHandler(self.config)
        self.binary_trainer = BinaryTrainer(self.config, self.data_handler)
        self.multiclass_trainer = MultiClassTrainer(self.config, self.data_handler)

    def run(self):
        """Run the complete federated learning pipeline"""
        print("Starting Federated Learning Pipeline")
        print("="*60)

        # Stage 1: Train binary models
        print("\nSTAGE 1: Training Binary Classifiers")
        binary_results = self.binary_trainer.train_all_binary_models()

        if not binary_results:
            print("No binary models were trained successfully")
            return

        # Stage 2: Aggregate weights
        print("\nSTAGE 2: Aggregating Backbone Weights")
        model_paths = [result['model_path'] for result in binary_results.values()]
        aggregated_weights = FederatedAggregator.aggregate_backbone_weights(model_paths)

        # Stage 3: Train multi-class model
        print("\nSTAGE 3: Training Multi-Class Model")
        multiclass_result = self.multiclass_trainer.train_multiclass_model(aggregated_weights)

        # Final summary
        print("\n" + "="*60)
        print("FEDERATED LEARNING PIPELINE COMPLETE")
        print("="*60)

        print(f"\nBinary Models Trained: {len(binary_results)}")
        for disease, result in binary_results.items():
            print(f"  {disease}: {result['test_metrics']['accuracy']:.2f}% accuracy")

        if multiclass_result:
            print(f"\nMulti-Class Model:")
            print(f"  Accuracy: {multiclass_result['test_metrics']['accuracy']:.2f}%")
            print(f"  F1 Score: {multiclass_result['test_metrics']['f1']:.3f}")

        print(f"\nResults saved to: {self.config.RESULTS_DIR}")

# ===== EXECUTION =====
def main():
    """Main execution function"""
    pipeline = FederatedLearningPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()