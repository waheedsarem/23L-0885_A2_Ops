"""SQLite/PostgreSQL prediction audit storage."""
from __future__ import annotations

from pathlib import Path
import sqlite3


class PredictionStore:
    def __init__(self, database_url: str, sqlite_path: Path):
        self.database_url = database_url.strip()
        self.sqlite_path = Path(sqlite_path)
        self.postgres = self.database_url.startswith(("postgresql://", "postgres://"))
        self.initialize()

    def connect(self):
        if self.postgres:
            import psycopg
            return psycopg.connect(self.database_url)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        if self.postgres:
            statement = """CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL, ai_probability DOUBLE PRECISION NOT NULL,
                threshold DOUBLE PRECISION NOT NULL, model_name TEXT NOT NULL,
                model_version TEXT NOT NULL)"""
        else:
            statement = """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                created_at TEXT NOT NULL, filename TEXT NOT NULL, label TEXT NOT NULL,
                confidence REAL NOT NULL, ai_probability REAL NOT NULL, threshold REAL NOT NULL,
                model_name TEXT NOT NULL, model_version TEXT NOT NULL)"""
        with self.connect() as connection:
            connection.execute(statement)
            if not self.postgres:
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

    def insert(self, values):
        marker = "%s" if self.postgres else "?"
        placeholders = ", ".join([marker] * 9)
        with self.connect() as connection:
            connection.execute(f"""INSERT INTO predictions
                (request_id, created_at, filename, label, confidence, ai_probability, threshold,
                 model_name, model_version) VALUES ({placeholders})""", values)

    def history(self, label="", model="", limit=50):
        marker = "%s" if self.postgres else "?"
        clauses, values = [], []
        if label:
            clauses.append(f"label = {marker}")
            values.append(label)
        if model:
            clauses.append(f"model_name = {marker}")
            values.append(model)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(limit)
        with self.connect() as connection:
            cursor = connection.execute(
                f"SELECT * FROM predictions {where} ORDER BY id DESC LIMIT {marker}", values)
            names = [item.name if hasattr(item, "name") else item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

    def models(self):
        with self.connect() as connection:
            rows = connection.execute("SELECT DISTINCT model_name FROM predictions ORDER BY model_name").fetchall()
            return [row[0] for row in rows]
