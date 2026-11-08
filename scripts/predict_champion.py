"""Load the DagsHub MLflow champion alias and classify one image."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT / ".env")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model = mlflow.keras.load_model("models:/VisionGuardDetector@champion")
sample = next((PROJECT / "dataset" / "raw" / "RealArt" / "RealArt").glob("*"))
with Image.open(sample) as image:
    image = ImageOps.pad(image.convert("RGB"), (160, 160), color="black")
    batch = np.expand_dims(np.asarray(image, dtype="float32"), 0)
probability = float(model.predict(batch, verbose=0).ravel()[0])
print({"image": str(sample.relative_to(PROJECT)), "ai_probability": probability,
       "prediction": "AI-generated" if probability >= .5 else "real"})
