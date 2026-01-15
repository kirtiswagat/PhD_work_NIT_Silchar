import os, numpy as np, tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        loss = -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)


class DiseaseGraphLayer(layers.Layer):
    """
    TRUE Phase-1 disease graph:
    - Nodes = diseases
    - Edges = co-occurrence prior (fixed)
    - Input queries disease space
    """

    def __init__(self, num_diseases, emb_dim=256):
        super().__init__()
        self.num_diseases = num_diseases
        self.emb_dim = emb_dim

    def build(self, input_shape):
        # Disease embeddings (one per disease)
        self.disease_embeddings = self.add_weight(
            shape=(self.num_diseases, self.emb_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="disease_embeddings"
        )

        # Disease–disease prior (identity = Phase-1 minimal)
        self.co_occurrence = tf.eye(self.num_diseases)

    def call(self, x):
        """
        x: (batch, emb_dim)
        """
        # Attention scores: patient embedding ↔ disease nodes
        attn = tf.matmul(x, self.disease_embeddings, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)

        # Graph aggregation
        disease_context = tf.matmul(attn, self.disease_embeddings)

        return x + disease_context

def load_img(path):
    img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (224,224))
    return img / 255.0


def build_dataset(img_paths, labels, num_classes, batch=16):
    def map_fn(p, y):
        return load_img(p), tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(map_fn).shuffle(512).batch(batch).prefetch(2)
    return ds


def build_phase1_model(num_diseases):
    base = EfficientNetB0(
        include_top=False, pooling='avg', weights='imagenet'
    )

    inp = layers.Input((224,224,3))
    feat = base(inp)

    # Stable projection
    feat = layers.Dense(256, activation='relu')(feat)

    # Disease Graph
    feat = DiseaseGraphLayer(num_diseases, 256)(feat)

    out = layers.Dense(num_diseases, activation='sigmoid')(feat)

    return Model(inp, out)

