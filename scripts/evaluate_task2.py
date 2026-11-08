"""Tasks 2.5-2.6: validation threshold selection and one-time test evaluation."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from tensorflow import keras

from task2_common import (DATA_ROOT, EVIDENCE_DIR, MODEL_DIR,
                          configure_reproducibility, make_dataset, read_split,
                          save_json)


def choose_threshold(y, probabilities):
    # False positive = a real photo called fake. Cost it twice as heavily because
    # falsely accusing authentic content is the more harmful product error.
    candidates = np.linspace(.05, .95, 181)
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append((threshold, 2 * fp + fn, f1_score(y, pred, zero_division=0)))
    return min(rows, key=lambda row: (row[1], -row[2]))[0]


def metrics_at(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_model(name, path, validation, test, y_val, y_test):
    model = keras.models.load_model(path)
    val_p = model.predict(validation, verbose=0).ravel()
    threshold = choose_threshold(y_val, val_p)
    # This is the only test prediction call for this model.
    test_p = model.predict(test, verbose=0).ravel()
    result = metrics_at(y_test, test_p, threshold)
    return result, test_p


def main():
    configure_reproducibility()
    tuner = pd.read_csv(EVIDENCE_DIR / "vision_tuner_trials.csv")
    if len(tuner) < 8:
        raise RuntimeError(f"Expected at least 8 VisionGuard tuner trials, found {len(tuner)}")
    paths = {"baseline": MODEL_DIR / "baseline_final.keras",
             "transfer": MODEL_DIR / "transfer_final.keras"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing trained models: {missing}")

    val_frame, _ = read_split("validation")
    test_frame, test_paths = read_split("test")
    y_val = val_frame["label"].to_numpy(dtype=int)
    y_test = test_frame["label"].to_numpy(dtype=int)
    validation = make_dataset("validation", training=False, augment=False)
    test = make_dataset("test", training=False, augment=False)

    results, probabilities = {}, {}
    for name, path in paths.items():
        results[name], probabilities[name] = evaluate_model(
            name, path, validation, test, y_val, y_test)
    pd.DataFrame(results).T.reset_index(names="model").to_csv(
        EVIDENCE_DIR / "test_metrics.csv", index=False)
    save_json(EVIDENCE_DIR / "test_metrics.json", results)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for col, name in enumerate(("baseline", "transfer")):
        r, p = results[name], probabilities[name]
        matrix = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax = axes[0, col]; image = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=14)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "AI"],
               yticklabels=["real", "AI"], xlabel="predicted", ylabel="actual",
               title=f"{name} confusion matrix (t={r['threshold']:.2f})")
        fpr, tpr, _ = roc_curve(y_test, p)
        ax = axes[1, col]; ax.plot(fpr, tpr, label=f"AUC={r['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlabel="false-positive rate", ylabel="true-positive rate",
               title=f"{name} ROC curve"); ax.legend(); ax.grid(alpha=.25)
    fig.savefig(EVIDENCE_DIR / "test_evaluation.png", dpi=180, facecolor="white")
    plt.close(fig)

    # Error analysis uses the transfer model and the already-computed test predictions.
    p = probabilities["transfer"]; threshold = results["transfer"]["threshold"]
    pred = (p >= threshold).astype(int)
    wrong = np.flatnonzero(pred != y_test)
    confidence = np.abs(p[wrong] - threshold)
    selected = wrong[np.argsort(confidence)[::-1][:10]]
    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    for ax, idx in zip(axes.flat, selected):
        image = keras.utils.load_img(test_paths[idx])
        ax.imshow(image); ax.axis("off")
        actual = "AI-generated" if y_test[idx] else "real"
        predicted = "AI-generated" if pred[idx] else "real"
        hypothesis = ("Stylized rendering, unusual texture, compression, or edited lighting may resemble generative artifacts."
                      if y_test[idx] == 0 else
                      "The generated image is photorealistic and lacks obvious anatomical, text, or texture artifacts.")
        ax.set_title(f"actual: {actual}\npred: {predicted}, p(AI)={p[idx]:.2f}", fontsize=9)
        rows.append({"relative_path": test_frame.iloc[idx]["relative_path"], "actual": actual,
                     "predicted": predicted, "ai_probability": float(p[idx]),
                     "hypothesis": hypothesis})
    for ax in axes.flat[len(selected):]: ax.axis("off")
    fig.suptitle("Ten most confident transfer-model errors", fontsize=15, fontweight="bold")
    fig.savefig(EVIDENCE_DIR / "misclassified_10.png", dpi=180, facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(EVIDENCE_DIR / "misclassified_10.csv", index=False)
    print(json.dumps(results, indent=2))
    print(f"Saved {len(rows)} error-analysis examples")


if __name__ == "__main__":
    main()
