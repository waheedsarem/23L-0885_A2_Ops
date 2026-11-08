"""Task 3.2: repeat the best Task 3.1 toy run on DagsHub MLflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import mlflow
from tensorflow import keras

from task3_1_mlflow_toy import configure_seed, run_one

PROJECT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT / ".env"
EXPECTED_URI = "https://dagshub.com/waheedsarem/23L-0885_A2_Ops.mlflow"


def main():
    if not ENV_FILE.is_file():
        raise FileNotFoundError("Missing .env. Copy .env.example to .env and add your DagsHub token.")
    load_dotenv(ENV_FILE, override=False)
    required = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")
    if os.environ["MLFLOW_TRACKING_PASSWORD"] == "PASTE_YOUR_DAGSHUB_ACCESS_TOKEN_HERE":
        raise RuntimeError("Replace the token placeholder in .env before running.")
    if os.environ["MLFLOW_TRACKING_URI"] != EXPECTED_URI:
        raise RuntimeError(f"Unexpected tracking URI; expected {EXPECTED_URI}")

    configure_seed()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("task_3_2_dagshub_toy")
    (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
    x_train, y_train = x_train[:6000], y_train[:6000]
    x_val, y_val = x_train[-1000:], y_train[-1000:]
    x_train, y_train = x_train[:-1000], y_train[:-1000]
    result = run_one(x_train, y_train, x_val, y_val, learning_rate=1e-3)
    print(json.dumps(result, indent=2))
    print(f"DagsHub MLflow: {EXPECTED_URI}")


if __name__ == "__main__":
    main()
