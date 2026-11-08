"""Shared, reproducible TensorFlow utilities for VisionGuard Task 2."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "230885")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT / "dataset" / "raw"
SPLIT_DIR = PROJECT / "splits"
EVIDENCE_DIR = PROJECT / "evidence" / "task_2"
ARTIFACT_DIR = PROJECT / "artifacts" / "task_2"
MODEL_DIR = ARTIFACT_DIR / "models"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
SEED = 230885
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
LABELS = {"real": 0, "AI-generated": 1}


def configure_reproducibility(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def augmentation_layer(seed: int = SEED):
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, fill_mode="reflect", seed=seed + 1),
            layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect", seed=seed + 2),
            layers.RandomContrast(0.12, seed=seed + 3),
        ],
        name="training_augmentation",
    )


def decode_resize(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    # Preserve aspect ratio; unused space is padded instead of stretching the image.
    image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1], antialias=True)
    return image, tf.cast(label, tf.float32)


def read_split(split_name: str):
    path = SPLIT_DIR / f"{split_name}.csv"
    frame = pd.read_csv(path)
    required = {"relative_path", "label", "class", "sha256"}
    if required - set(frame.columns):
        raise ValueError(f"Missing split columns in {path}: {sorted(required - set(frame.columns))}")
    full_paths = [str(DATA_ROOT / value) for value in frame["relative_path"]]
    missing = [value for value in full_paths if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images are missing; first: {missing[0]}")
    return frame, full_paths


def make_dataset(split_name: str, training: bool, batch_size: int = BATCH_SIZE, augment: bool = True):
    frame, paths = read_split(split_name)
    dataset = tf.data.Dataset.from_tensor_slices((paths, frame["label"].astype("float32").to_numpy()))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if training and augment:
        augmenter = augmentation_layer()
        dataset = dataset.map(lambda images, labels: (augmenter(images, training=True), labels),
                              num_parallel_calls=1, deterministic=True)
    return dataset.prefetch(tf.data.AUTOTUNE)


def metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]


def build_baseline(learning_rate: float = 1e-3):
    inputs = keras.Input((*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv_{filters}")(x)
        x = layers.BatchNormalization(name=f"bn_{filters}")(x)
        x = layers.ReLU(name=f"relu_{filters}")(x)
        x = layers.MaxPooling2D(name=f"pool_{filters}")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    x = layers.Dropout(0.40, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.BinaryCrossentropy(), metrics=metrics())
    return model


def build_transfer(dropout: float = 0.35, learning_rate: float = 1e-3,
                   unfrozen_layers: int = 0, weights: str | None = "imagenet"):
    base = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weights
    )
    base.trainable = False
    if unfrozen_layers:
        for layer in base.layers[-unfrozen_layers:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        # Setting individual layers trainable requires the model flag to be true.
        base.trainable = True
        for layer in base.layers[:-unfrozen_layers]:
            layer.trainable = False
        for layer in base.layers[-unfrozen_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    inputs = keras.Input((*IMAGE_SIZE, 3), name="image")
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    # Keep BatchNormalization behavior in inference mode during both stages.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ai_probability")(x)
    model = keras.Model(inputs, outputs, name="mobilenetv2_detector")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.BinaryCrossentropy(), metrics=metrics())
    return model, base


def callbacks_for(name: str, patience: int = 3):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.ModelCheckpoint(CHECKPOINT_DIR / f"{name}.keras", monitor="val_loss",
                                        save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2,
                                         min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(ARTIFACT_DIR / "logs" / f"{name}.csv"),
    ]


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


configure_reproducibility()
