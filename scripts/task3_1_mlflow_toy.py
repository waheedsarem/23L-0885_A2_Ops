"""Task 3.1: two reproducible local MLflow runs on Fashion-MNIST."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "230885")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT = Path(__file__).resolve().parents[1]
TRACKING_DB = PROJECT / "mlflow.db"
EVIDENCE_DIR = PROJECT / "evidence" / "task_3_1"
SEED = 230885
EXPERIMENT = "task_3_1_fashion_mnist_toy"


class MLflowMetrics(keras.callbacks.Callback):
    """Log every Keras epoch explicitly to the active MLflow run."""
    def on_epoch_end(self, epoch, logs=None):
        for name, value in (logs or {}).items():
            if np.isscalar(value):
                mlflow.log_metric(name, float(value), step=epoch + 1)


def configure_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_model(learning_rate):
    model = keras.Sequential([
        layers.Input((28, 28), name="image"),
        layers.Rescaling(1 / 255.0, name="rescale"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="hidden_64"),
        layers.Dropout(0.25, name="dropout"),
        layers.Dense(10, activation="softmax", name="class_probabilities"),
    ], name="fashion_mnist_tiny")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_curve(history, destination, learning_rate):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history.history[metric], marker="o", label=f"train {metric}")
        ax.plot(history.history[f"val_{metric}"], marker="o", label=f"validation {metric}")
        ax.set(xlabel="epoch", ylabel=metric, title=f"LR={learning_rate:g}: {metric}")
        ax.grid(alpha=.25); ax.legend()
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def run_one(x_train, y_train, x_val, y_val, learning_rate):
    configure_seed()
    model = build_model(learning_rate)
    run_name = f"adam_lr_{learning_rate:g}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "task": "3.1", "dataset": "Fashion-MNIST", "model_family": "tiny_dense",
            "purpose": "local MLflow tool lab", "seed": str(SEED),
        })
        mlflow.log_params({
            "learning_rate": learning_rate, "optimizer": "Adam", "epochs": 5,
            "batch_size": 128, "hidden_units": 64, "dropout": 0.25,
            "train_examples": len(x_train), "validation_examples": len(x_val),
        })
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=5, batch_size=128, verbose=2,
                            callbacks=[MLflowMetrics()])
        final = {"final_train_loss": history.history["loss"][-1],
                 "final_train_accuracy": history.history["accuracy"][-1],
                 "final_val_loss": history.history["val_loss"][-1],
                 "final_val_accuracy": history.history["val_accuracy"][-1]}
        mlflow.log_metrics({key: float(value) for key, value in final.items()})
        curve = EVIDENCE_DIR / f"{run_name}_curves.png"
        save_curve(history, curve, learning_rate)
        mlflow.log_artifact(curve, artifact_path="figures")
        summary = EVIDENCE_DIR / f"{run_name}_summary.json"
        summary.write_text(json.dumps({"run_id": run.info.run_id,
                                       "learning_rate": learning_rate, **final}, indent=2),
                           encoding="utf-8")
        mlflow.log_artifact(summary, artifact_path="summaries")
        try:
            mlflow.keras.log_model(model, name="model")
        except TypeError:  # MLflow 2.x compatibility
            mlflow.keras.log_model(model, artifact_path="model")
        return {"run_id": run.info.run_id, "run_name": run_name,
                "learning_rate": learning_rate, **{k: float(v) for k, v in final.items()}}


def main():
    configure_seed()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    # Current MLflow versions require a database metadata backend. SQLite is
    # local, portable, and needs no external service.
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT)
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    # Fixed small slices keep the lab fast; validation is disjoint from training.
    x_train, y_train = x_train[:6000], y_train[:6000]
    x_val, y_val = x_train[-1000:], y_train[-1000:]
    x_train, y_train = x_train[:-1000], y_train[:-1000]
    results = [run_one(x_train, y_train, x_val, y_val, lr) for lr in (1e-3, 1e-4)]
    comparison = EVIDENCE_DIR / "toy_run_comparison.json"
    comparison.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Tracking URI: sqlite:///{TRACKING_DB.as_posix()}")
    print("Start UI: .\\.venv\\Scripts\\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000")


if __name__ == "__main__":
    main()
