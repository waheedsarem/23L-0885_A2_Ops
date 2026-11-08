"""Tasks 3.3-3.6: DagsHub runs, artifacts, comparison, and model registry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow import MlflowClient
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from task2_common import (EVIDENCE_DIR as TASK2_EVIDENCE, IMAGE_SIZE, SEED,
                          build_transfer, configure_reproducibility, make_dataset)

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence" / "task_3_3_to_3_6"
MODEL_NAME = "VisionGuardDetector"
EXPERIMENT = "VisionGuard_controlled_experiments"

RUNS = [
    {"name": "manual_baseline_reference", "logging": "manual", "family": "baseline",
     "learning_rate": 1e-3, "dropout": .40, "augmentation": True, "unfrozen_layers": 0},
    {"name": "autolog_baseline_lr_3e-4", "logging": "autolog", "family": "baseline",
     "learning_rate": 3e-4, "dropout": .40, "augmentation": True, "unfrozen_layers": 0},
    {"name": "autolog_baseline_dropout_025", "logging": "autolog", "family": "baseline",
     "learning_rate": 1e-3, "dropout": .25, "augmentation": True, "unfrozen_layers": 0},
    {"name": "autolog_baseline_no_augmentation", "logging": "autolog", "family": "baseline",
     "learning_rate": 1e-3, "dropout": .40, "augmentation": False, "unfrozen_layers": 0},
    {"name": "manual_mobilenet_frozen", "logging": "manual", "family": "mobilenetv2",
     "learning_rate": 1e-3, "dropout": .35, "augmentation": True, "unfrozen_layers": 0},
    {"name": "autolog_mobilenet_unfreeze20", "logging": "autolog", "family": "mobilenetv2",
     "learning_rate": 1e-5, "dropout": .35, "augmentation": True, "unfrozen_layers": 20},
]


class ManualMLflowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def build_baseline(dropout, learning_rate):
    inputs = keras.Input((*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 255.0)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x); x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(keras.optimizers.Adam(learning_rate), "binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])
    return model


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                   text=True).strip()


def dvc_version():
    text = (PROJECT / "dataset" / "raw.dvc").read_text(encoding="utf-8")
    match = re.search(r"md5:\s*([^\s]+)", text)
    return match.group(1) if match else "unknown"


def curve(history, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history.history[metric], marker="o", label=f"train {metric}")
        ax.plot(history.history[f"val_{metric}"], marker="o", label=f"validation {metric}")
        ax.set(title=f"{title}: {metric}", xlabel="epoch", ylabel=metric)
        ax.grid(alpha=.25); ax.legend()
    fig.savefig(path, dpi=160, facecolor="white"); plt.close(fig)


def log_common_artifacts(config, history):
    run_dir = EVIDENCE / config["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "training_curves.png"
    curve(history, curve_path, config["name"])
    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index_label="epoch")
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    mlflow.log_artifacts(run_dir, artifact_path="training_evidence")
    mlflow.log_artifact(PROJECT / "requirements.txt", artifact_path="reproducibility")
    mlflow.log_artifacts(PROJECT / "splits", artifact_path="reproducibility/splits")


def run_experiment(config, commit, data_version):
    configure_reproducibility()
    keras.backend.clear_session()
    if config["family"] == "baseline":
        model = build_baseline(config["dropout"], config["learning_rate"])
    else:
        model, _ = build_transfer(config["dropout"], config["learning_rate"],
                                  config["unfrozen_layers"])
    train = make_dataset("train", training=True, augment=config["augmentation"])
    validation = make_dataset("validation", training=False, augment=False)

    if config["logging"] == "autolog":
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=False)
    else:
        mlflow.tensorflow.autolog(disable=True)

    with mlflow.start_run(run_name=config["name"]) as active:
        mlflow.log_params({**config, "epochs_max": 5, "batch_size": 16,
                           "image_size": "160x160", "optimizer": "Adam"})
        mlflow.set_tags({"task": "3.3-3.5", "git_commit": commit,
                         "dvc_data_version": data_version, "seed": str(SEED),
                         "logging_approach": config["logging"],
                         "model_family": config["family"]})
        callbacks = [keras.callbacks.EarlyStopping("val_loss", patience=2,
                                                   restore_best_weights=True)]
        if config["logging"] == "manual": callbacks.append(ManualMLflowCallback())
        history = model.fit(train, validation_data=validation, epochs=5,
                            callbacks=callbacks, verbose=2)
        best_epoch = int(pd.Series(history.history["val_loss"]).idxmin() + 1)
        final = {"best_val_loss": min(history.history["val_loss"]),
                 "best_val_accuracy": max(history.history["val_accuracy"]),
                 "best_epoch": best_epoch}
        mlflow.log_metrics({k: float(v) for k, v in final.items()})
        log_common_artifacts(config, history)
        if config["name"] == "manual_baseline_reference":
            cached = json.loads((TASK2_EVIDENCE / "test_metrics.json").read_text())["baseline"]
            mlflow.log_metrics({f"final_test_{k}": float(v) for k, v in cached.items()
                                if k in {"accuracy", "precision", "recall", "f1", "roc_auc"}})
        if config["family"] == "mobilenetv2":
            mlflow.log_artifact(TASK2_EVIDENCE / "test_evaluation.png", artifact_path="evaluation")
            mlflow.log_artifact(TASK2_EVIDENCE / "misclassified_10.png", artifact_path="evaluation")
            mlflow.log_artifact(TASK2_EVIDENCE / "test_metrics.csv", artifact_path="evaluation")
        if config["logging"] == "manual":
            mlflow.keras.log_model(model, name="model")
        result = {**config, "run_id": active.info.run_id,
                  **{k: float(v) for k, v in final.items()}}
    if config["logging"] == "autolog": mlflow.tensorflow.autolog(disable=True)
    return result


def register_best(results):
    ranked = sorted(results, key=lambda x: (x["best_val_accuracy"], -x["best_val_loss"]), reverse=True)
    # Register the two strongest validation runs as two model versions.
    versions = []
    for result in ranked[:2]:
        registered = mlflow.register_model(result["model_uri"], MODEL_NAME)
        versions.append({"version": int(registered.version), "run_id": result["run_id"],
                         "run_name": result["name"], "best_val_accuracy": result["best_val_accuracy"]})
    client = MlflowClient()
    champion = max(versions, key=lambda x: x["best_val_accuracy"])
    client.set_registered_model_alias(MODEL_NAME, "champion", str(champion["version"]))
    return versions, champion


def main():
    load_dotenv(PROJECT / ".env")
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
        if not os.environ.get(name): raise RuntimeError(f"Missing {name} in .env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    commit, data_version = git_commit(), dvc_version()
    results = [run_experiment(config, commit, data_version) for config in RUNS]
    versions, champion = register_best(results)
    summary = {"experiment": EXPERIMENT, "runs": results,
               "registry_versions": versions, "champion": champion}
    (EVIDENCE / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(EVIDENCE / "run_comparison.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
