"""Flask application factory and validated in-memory upload routes."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import uuid

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .inference import InferenceService, InvalidImageError
from .database import PredictionStore

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def create_app(config=None, inference_service=None):
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env", override=False)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MODEL_PATH=os.getenv("MODEL_PATH", str(project / "artifacts/task_2/models/transfer_final.keras")),
        MODEL_THRESHOLD=float(os.getenv("MODEL_THRESHOLD", "0.825")),
        MODEL_NAME=os.getenv("MODEL_NAME", "VisionGuardDetector"),
        MODEL_VERSION=os.getenv("MODEL_VERSION", "champion-v2"),
        DATABASE_URL=os.getenv("DATABASE_URL", ""),
        DATABASE_PATH=Path(os.getenv("DATABASE_PATH", str(project / "instance/predictions.sqlite3"))),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        MAX_HISTORY=int(os.getenv("MAX_HISTORY", "50")),
    )
    if config:
        app.config.update(config)
    store = PredictionStore(app.config["DATABASE_URL"], Path(app.config["DATABASE_PATH"]))
    app.extensions["prediction_store"] = store
    service = inference_service or InferenceService(app.config["MODEL_PATH"], app.config["MODEL_THRESHOLD"])
    app.extensions["inference_service"] = service

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))

    @app.post("/predict")
    def predict():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return render_template("error.html", title="No image selected",
                                   message="Choose a JPG, PNG or WebP image and try again."), 400
        safe_name = secure_filename(uploaded.filename)
        extension = Path(safe_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS or uploaded.mimetype not in ALLOWED_MIME:
            return render_template("error.html", title="Unsupported file type",
                                   message="Only JPG, PNG and WebP images are accepted."), 400
        image_bytes = uploaded.read()  # stays in memory; never persisted to disk
        try:
            result = service.predict_bytes(image_bytes)
        except InvalidImageError as exc:
            return render_template("error.html", title="Invalid image", message=str(exc)), 400
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        request_id = str(uuid.uuid4())
        store.insert((request_id, created_at, safe_name, result["label"], result["confidence"],
                      result["ai_probability"], result["threshold"], app.config["MODEL_NAME"],
                      app.config["MODEL_VERSION"]))
        return render_template("result.html", filename=safe_name, result=result)

    @app.get("/history")
    def history():
        selected_label = request.args.get("label", "").strip()
        selected_model = request.args.get("model", "").strip()
        if selected_label not in {"Real", "AI-generated"}:
            selected_label = ""
        rows = store.history(selected_label, selected_model, app.config["MAX_HISTORY"])
        models = store.models()
        return render_template("history.html", rows=rows, models=models,
                               selected_label=selected_label, selected_model=selected_model)

    @app.get("/api/history")
    def history_api():
        try:
            limit = max(1, min(int(request.args.get("limit", "50")), 200))
        except ValueError:
            return jsonify(error="limit must be an integer between 1 and 200"), 400
        label = request.args.get("label", "").strip()
        if label:
            if label not in {"Real", "AI-generated"}:
                return jsonify(error="label must be Real or AI-generated"), 400
        rows = store.history(label=label, limit=limit)
        return jsonify(records=rows, count=len(rows))

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=service.model is not None,
                       threshold=service.threshold, model_path=str(service.model_path),
                       model_name=app.config["MODEL_NAME"], model_version=app.config["MODEL_VERSION"],
                       database="postgresql" if store.postgres else "sqlite",
                       history_records="persistent")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return render_template("error.html", title="File too large",
                               message=f"The upload limit is {limit} MB."), 413

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found",
                               message="That page does not exist."), 404

    return app
