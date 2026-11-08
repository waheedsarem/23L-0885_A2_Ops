"""Finalize Tasks 3.5-3.6 after all six DagsHub runs have completed."""
from __future__ import annotations
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
from mlflow import MlflowClient
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence" / "task_3_3_to_3_6"
EXPERIMENT = "VisionGuard_controlled_experiments"
MODEL_NAME = "VisionGuardDetector"
MANUAL_RUNS = {
    "manual_baseline_reference": "a2b66867048d4d62ac9611c885f4c31b",
    "manual_mobilenet_frozen": "b7f3d59c2984427290bb5d1c7c020c37",
}


def field(obj, name):
    return getattr(obj, name, None) or (obj.to_dictionary().get(name) if hasattr(obj, "to_dictionary") else None)


def main():
    load_dotenv(PROJECT / ".env")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Missing experiment: {EXPERIMENT}")

    runs = mlflow.search_runs([experiment.experiment_id], output_format="pandas")
    run_cols = [c for c in ["run_id", "tags.mlflow.runName", "params.family", "params.logging",
                             "params.learning_rate", "params.dropout", "params.augmentation",
                             "params.unfrozen_layers", "metrics.best_val_accuracy",
                             "metrics.best_val_loss", "metrics.best_epoch"] if c in runs.columns]
    comparison = runs[run_cols].copy().sort_values("metrics.best_val_accuracy", ascending=False)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(EVIDENCE / "run_comparison.csv", index=False)

    logged = list(client.search_logged_models(experiment_ids=[experiment.experiment_id]))
    by_run = {}
    for model in logged:
        source_run = field(model, "source_run_id")
        model_id = field(model, "model_id")
        if source_run and model_id:
            by_run[source_run] = model_id

    versions = []
    for run_name, run_id in MANUAL_RUNS.items():
        model_id = by_run.get(run_id)
        if not model_id:
            raise RuntimeError(f"No MLflow 3 logged model found for {run_name} ({run_id})")
        registered = mlflow.register_model(f"models:/{model_id}", MODEL_NAME)
        versions.append({"version": int(registered.version), "run_name": run_name,
                         "run_id": run_id, "logged_model_id": model_id})

    champion = next(v for v in versions if v["run_name"] == "manual_mobilenet_frozen")
    client.set_registered_model_alias(MODEL_NAME, "champion", str(champion["version"]))
    summary = {
        "experiment_id": experiment.experiment_id,
        "completed_run_count": int(len(runs)),
        "registered_versions": versions,
        "champion": champion,
        "selection_reason": "MobileNetV2 achieved the strongest validation accuracy among models with a successfully logged reusable model artifact.",
        "autolog_note": "TensorFlow autologging captured parameters and epoch metrics but its automatic model upload returned INVALID_PARAMETER_VALUE on the DagsHub backend; manual logging captured the model reliably.",
    }
    (EVIDENCE / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
