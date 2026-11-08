"""Train baseline CNN and two-stage MobileNetV2; create Tasks 2.2-2.4 evidence."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from task2_common import (ARTIFACT_DIR, EVIDENCE_DIR, MODEL_DIR, build_baseline,
                          build_transfer, callbacks_for, configure_reproducibility,
                          make_dataset, save_json)


def history_frame(history, stage):
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", range(1, len(frame) + 1))
    frame.insert(1, "stage", stage)
    return frame


def model_evidence(model, name):
    lines = []
    model.summary(print_fn=lines.append)
    (EVIDENCE_DIR / f"{name}_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    rows = []
    for layer in model.layers:
        rows.append({"layer": layer.name, "type": layer.__class__.__name__,
                     "output_shape": str(getattr(layer, "output", "").shape),
                     "parameters": layer.count_params(), "trainable": layer.trainable})
    pd.DataFrame(rows).to_csv(EVIDENCE_DIR / f"{name}_layers.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.text(0.01, 0.99, "\n".join(lines), va="top", family="monospace", fontsize=7)
    fig.savefig(EVIDENCE_DIR / f"{name}_summary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_history(frame, name, stage_boundary=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(frame["epoch"], frame[metric], marker="o", ms=3, label=f"train {metric}")
        ax.plot(frame["epoch"], frame[f"val_{metric}"], marker="o", ms=3, label=f"validation {metric}")
        best = frame[f"val_{metric}"].idxmin() if metric == "loss" else frame[f"val_{metric}"].idxmax()
        ax.scatter(frame.loc[best, "epoch"], frame.loc[best, f"val_{metric}"], s=80,
                   marker="*", color="red", zorder=5, label="best validation epoch")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="black", ls="--", lw=1, label="fine-tuning starts")
        ax.set(xlabel="epoch", ylabel=metric, title=f"{name}: {metric}")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(EVIDENCE_DIR / f"{name}_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def train_baseline(train, validation):
    model = build_baseline()
    model_evidence(model, "baseline_cnn")
    history = model.fit(train, validation_data=validation, epochs=15,
                        callbacks=callbacks_for("baseline", patience=4), verbose=2)
    frame = history_frame(history, "baseline")
    frame.to_csv(EVIDENCE_DIR / "baseline_history.csv", index=False)
    plot_history(frame, "baseline")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "baseline_final.keras")
    return {"epochs_run": len(frame), "best_epoch": int(frame["val_loss"].idxmin() + 1),
            "best_val_loss": float(frame["val_loss"].min()),
            "best_val_accuracy": float(frame["val_accuracy"].max())}


def train_transfer(train, validation):
    model, base = build_transfer(dropout=.35, learning_rate=1e-3, unfrozen_layers=0)
    frozen_names = [layer.name for layer in base.layers if layer.trainable]
    h1 = model.fit(train, validation_data=validation, epochs=8,
                   callbacks=callbacks_for("transfer_frozen", patience=3), verbose=2)
    f1 = history_frame(h1, "frozen_feature_extractor")

    # Rebuild with the last 20 layers eligible for fine-tuning, then transfer stage-1 weights.
    tuned, tuned_base = build_transfer(dropout=.35, learning_rate=1e-5, unfrozen_layers=20)
    tuned.set_weights(model.get_weights())
    trainable = [layer.name for layer in tuned_base.layers if layer.trainable]
    frozen = [layer.name for layer in tuned_base.layers if not layer.trainable]
    h2 = tuned.fit(train, validation_data=validation, epochs=8,
                   callbacks=callbacks_for("transfer_finetuned", patience=3), verbose=2)
    f2 = history_frame(h2, "fine_tuned")
    f2["epoch"] += len(f1)
    combined = pd.concat([f1, f2], ignore_index=True)
    combined.to_csv(EVIDENCE_DIR / "transfer_history.csv", index=False)
    plot_history(combined, "transfer", stage_boundary=len(f1))
    model_evidence(tuned, "mobilenetv2_finetuned")
    save_json(EVIDENCE_DIR / "transfer_trainable_layers.json", {
        "stage_1_trainable_base_layers": frozen_names,
        "stage_2_trainable_base_layer_count": len(trainable),
        "stage_2_trainable_base_layers": trainable,
        "stage_2_frozen_base_layer_count": len(frozen),
        "batch_normalization_kept_in_inference_mode": True,
        "stage_1_learning_rate": 1e-3, "stage_2_learning_rate": 1e-5,
    })
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tuned.save(MODEL_DIR / "transfer_final.keras")
    return {"stage_1_epochs": len(f1), "stage_2_epochs": len(f2),
            "best_epoch_overall": int(combined["val_loss"].idxmin() + 1),
            "best_val_loss": float(combined["val_loss"].min()),
            "best_val_accuracy": float(combined["val_accuracy"].max()),
            "trainable_base_layers_stage_2": len(trainable)}


def main():
    configure_reproducibility()
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    train = make_dataset("train", training=True, augment=True)
    validation = make_dataset("validation", training=False, augment=False)
    results = {"baseline": train_baseline(train, validation)}
    keras.backend.clear_session()
    results["transfer"] = train_transfer(train, validation)
    save_json(EVIDENCE_DIR / "training_summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
