"""Generate reproducible Task 5 database-trail evidence without loading the large model."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sqlite3
import sys

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from webapp import create_app


EVIDENCE = PROJECT / "evidence" / "task_5"
DATABASE = EVIDENCE / "audit_evidence.sqlite3"


class EvidenceService:
    model = object()
    model_path = Path("artifacts/task_2/models/transfer_final.keras")
    threshold = 0.825

    def __init__(self):
        self.index = 0

    def predict_bytes(self, _data):
        results = [
            {"label": "Real", "confidence": 0.917, "ai_probability": 0.083, "threshold": 0.825},
            {"label": "AI-generated", "confidence": 0.906, "ai_probability": 0.906, "threshold": 0.825},
            {"label": "Real", "confidence": 0.786, "ai_probability": 0.214, "threshold": 0.825},
        ]
        result = results[self.index]
        self.index += 1
        return result


def image_bytes(colour):
    stream = BytesIO()
    Image.new("RGB", (32, 32), colour).save(stream, "PNG")
    return stream.getvalue()


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if DATABASE.exists():
        DATABASE.unlink()
    config = {
        "TESTING": True,
        "DATABASE_PATH": DATABASE,
        "MODEL_NAME": "VisionGuardDetector",
        "MODEL_VERSION": "champion-v2",
    }
    service = EvidenceService()
    client = create_app(config, service).test_client()
    for filename, colour in [("known_real.png", "royalblue"), ("known_ai.png", "orange"),
                             ("second_real.png", "seagreen")]:
        response = client.post(
            "/predict",
            data={"image": (BytesIO(image_bytes(colour)), filename, "image/png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

    # Recreate the application to demonstrate records survive a process/app restart.
    restarted = create_app(config, EvidenceService()).test_client()
    api = restarted.get("/api/history?limit=10").get_json()
    filtered = restarted.get("/api/history?label=AI-generated&limit=10").get_json()
    health = restarted.get("/health").get_json()
    history_html = restarted.get("/history").get_data(as_text=True)
    history_html = history_html.replace('href="/static/style.css"',
                                        'href="../../webapp/static/style.css"')
    (EVIDENCE / "history.html").write_text(history_html, encoding="utf-8")

    with sqlite3.connect(DATABASE) as connection:
        columns = [dict(zip(("position", "name", "type", "not_null", "default", "primary_key"), row))
                   for row in connection.execute("PRAGMA table_info(predictions)")]
        indexes = [row[1] for row in connection.execute("PRAGMA index_list(predictions)")]
        count = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    proof = {
        "database": "SQLite",
        "database_file": str(DATABASE.relative_to(PROJECT)),
        "records_after_app_restart": count,
        "schema": columns,
        "indexes": indexes,
        "all_records_query": api,
        "filtered_ai_generated_query": filtered,
        "health": health,
        "privacy": "Only decision metadata is stored; uploaded image bytes are not retained.",
    }
    (EVIDENCE / "database_audit_proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps({"records_after_restart": count, "columns": [item["name"] for item in columns],
                      "filtered_records": filtered["count"]}, indent=2))


if __name__ == "__main__":
    main()
