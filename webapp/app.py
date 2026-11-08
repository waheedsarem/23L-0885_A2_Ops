"""Flask application factory and validated in-memory upload routes."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from .inference import InferenceService, InvalidImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


def init_database(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL,
            confidence REAL NOT NULL, ai_probability REAL NOT NULL, threshold REAL NOT NULL)""")


def create_app(config=None, inference_service=None):
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env", override=False)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MODEL_PATH=os.getenv("MODEL_PATH", str(project / "artifacts/task_2/models/transfer_final.keras")),
        MODEL_THRESHOLD=float(os.getenv("MODEL_THRESHOLD", "0.825")),
        DATABASE_PATH=Path(os.getenv("DATABASE_PATH", str(project / "instance/predictions.sqlite3"))),
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))),
        MAX_HISTORY=int(os.getenv("MAX_HISTORY", "50")),
    )
    if config:
        app.config.update(config)
    init_database(Path(app.config["DATABASE_PATH"]))
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
        with sqlite3.connect(app.config["DATABASE_PATH"]) as connection:
            connection.execute("""INSERT INTO predictions
                (created_at, filename, label, confidence, ai_probability, threshold)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (created_at, safe_name, result["label"], result["confidence"],
                 result["ai_probability"], result["threshold"]))
        return render_template("result.html", filename=safe_name, result=result)

    @app.get("/history")
    def history():
        with sqlite3.connect(app.config["DATABASE_PATH"]) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                                      (app.config["MAX_HISTORY"],)).fetchall()
        return render_template("history.html", rows=rows)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=service.model is not None,
                       threshold=service.threshold, model_path=str(service.model_path))

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
