"""Task 2.7: toy KerasTuner search and eight VisionGuard validation-only trials."""
from __future__ import annotations

import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras_tuner as kt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from task2_common import (ARTIFACT_DIR, EVIDENCE_DIR, SEED, build_transfer,
                          configure_reproducibility, make_dataset, save_json)


def toy_model(hp):
    model = keras.Sequential([
        layers.Input((28, 28)), layers.Rescaling(1 / 255), layers.Flatten(),
        layers.Dense(hp.Choice("units", [32, 64, 128]), activation="relu"),
        layers.Dropout(hp.Choice("dropout", [0.1, 0.3])),
        layers.Dense(10, activation="softmax")])
    model.compile(keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
                  "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def vision_model(hp):
    model, _ = build_transfer(
        dropout=hp.Choice("dropout", [0.2, 0.35, 0.5]),
        learning_rate=hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4, 3e-5]),
        unfrozen_layers=hp.Choice("unfrozen_layers", [0, 10, 20, 30]))
    return model


def trials_frame(tuner):
    rows = []
    for trial in tuner.oracle.trials.values():
        row = {"trial_id": trial.trial_id, "status": trial.status,
               "best_val_accuracy": trial.score}
        row.update(trial.hyperparameters.values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("best_val_accuracy", ascending=False)


def main():
    configure_reproducibility()
    tuner_root = ARTIFACT_DIR / "tuner"
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    # A small fixed subset keeps the toy demonstration fast and reproducible.
    x_train, y_train = x_train[:6000], y_train[:6000]
    toy = kt.RandomSearch(toy_model, objective="val_accuracy", max_trials=4, seed=SEED,
                          directory=tuner_root, project_name="toy_fashion", overwrite=True)
    toy.search(x_train, y_train, validation_split=.2, epochs=3, batch_size=128,
               callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=1)], verbose=2)
    toy_table = trials_frame(toy)
    toy_table.to_csv(EVIDENCE_DIR / "toy_tuner_trials.csv", index=False)

    train = make_dataset("train", training=True, augment=True)
    validation = make_dataset("validation", training=False, augment=False)
    vision = kt.RandomSearch(vision_model, objective="val_accuracy", max_trials=8, seed=SEED,
                             directory=tuner_root, project_name="visionguard", overwrite=True)
    vision.search(train, validation_data=validation, epochs=4,
                  callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=2,
                                                            restore_best_weights=True)], verbose=2)
    table = trials_frame(vision)
    table.to_csv(EVIDENCE_DIR / "vision_tuner_trials.csv", index=False)
    best = table.iloc[0].to_dict()
    save_json(EVIDENCE_DIR / "tuner_summary.json", {
        "toy_trials": len(toy_table), "vision_trials": len(table),
        "vision_best": best,
        "warning": "The best validation trial is not an unbiased test estimate; repeated trial selection can overfit validation data."
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"T{r.trial_id}\nlr={r.learning_rate:g}\nd={r.dropout}\nu={int(r.unfrozen_layers)}"
              for r in table.itertuples()]
    ax.bar(range(len(table)), table["best_val_accuracy"], color="#4472C4")
    ax.set_xticks(range(len(table)), labels, fontsize=8)
    ax.set(ylabel="best validation accuracy", title="VisionGuard KerasTuner trials (validation only)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(EVIDENCE_DIR / "vision_tuner_results.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"toy": toy_table.to_dict("records"),
                      "vision": table.to_dict("records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
