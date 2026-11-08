"""Flask application factory and validated in-memory upload routes."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3
import uuid

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
            request_id TEXT NOT NULL,
            created_at TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL,
            confidence REAL NOT NULL, ai_probability REAL NOT NULL, threshold REAL NOT NULL,
            model_name TEXT NOT NULL, model_version TEXT NOT NULL)""")
        existing = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
        migrations = {
            "request_id": "TEXT NOT NULL DEFAULT 'legacy'",
            "model_name": "TEXT NOT NULL DEFAULT 'VisionGuardDetector'",
            "model_version": "TEXT NOT NULL DEFAULT 'pre-audit-trail'",
        }
        for column, definition in migrations.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(label)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, model_version)")


def create_app(config=None, inference_service=None):
    project = Path(__file__).resolve().parents[1]
    load_dotenv(project / ".env", override=False)
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MODEL_PATH=os.getenv("MODEL_PATH", str(project / "artifacts/task_2/models/transfer_final.keras")),
        MODEL_THRESHOLD=float(os.getenv("MODEL_THRESHOLD", "0.825")),
        MODEL_NAME=os.getenv("MODEL_NAME", "VisionGuardDetector"),
        MODEL_VERSION=os.getenv("MODEL_VERSION", "champion-v2"),
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
        request_id = str(uuid.uuid4())
        with sqlite3.connect(app.config["DATABASE_PATH"]) as connection:
            connection.execute("""INSERT INTO predictions
                (request_id, created_at, filename, label, confidence, ai_probability, threshold,
                 model_name, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (request_id, created_at, safe_name, result["label"], result["confidence"],
                 result["ai_probability"], result["threshold"], app.config["MODEL_NAME"],
                 app.config["MODEL_VERSION"]))
        return render_template("result.html", filename=safe_name, result=result)

    @app.get("/history")
    def history():
        selected_label = request.args.get("label", "").strip()
        selected_model = request.args.get("model", "").strip()
        clauses, values = [], []
        if selected_label in {"Real", "AI-generated"}:
            clauses.append("label = ?"); values.append(selected_label)
        if selected_model:
            clauses.append("model_name = ?"); values.append(selected_model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with sqlite3.connect(app.config["DATABASE_PATH"]) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ?",
                                      (*values, app.config["MAX_HISTORY"])).fetchall()
            models = [row[0] for row in connection.execute(
                "SELECT DISTINCT model_name FROM predictions ORDER BY model_name")]
        return render_template("history.html", rows=rows, models=models,
                               selected_label=selected_label, selected_model=selected_model)

    @app.get("/api/history")
    def history_api():
        try:
            limit = max(1, min(int(request.args.get("limit", "50")), 200))
        except ValueError:
            return jsonify(error="limit must be an integer between 1 and 200"), 400
        label = request.args.get("label", "").strip()
        clauses, values = [], []
        if label:
            if label not in {"Real", "AI-generated"}:
                return jsonify(error="label must be Real or AI-generated"), 400
            clauses.append("label = ?"); values.append(label)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with sqlite3.connect(app.config["DATABASE_PATH"]) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT ?",
                                      (*values, limit)).fetchall()
        return jsonify(records=[dict(row) for row in rows], count=len(rows))

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=service.model is not None,
                       threshold=service.threshold, model_path=str(service.model_path),
                       model_name=app.config["MODEL_NAME"], model_version=app.config["MODEL_VERSION"],
                       database="sqlite", history_records="persistent")

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
