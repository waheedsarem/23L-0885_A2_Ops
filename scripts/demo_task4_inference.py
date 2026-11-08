"""Task 4.1 console evidence on one known image from each class."""
from pathlib import Path
import sys
import json

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from webapp.inference import InferenceService
service = InferenceService(PROJECT / "artifacts/task_2/models/transfer_final.keras", threshold=.825)
results = []
for folder, expected in ((PROJECT / "dataset/raw/RealArt/RealArt", "Real"),
                         (PROJECT / "dataset/raw/AiArtData/AiArtData", "AI-generated")):
    image = next(path for path in folder.iterdir() if path.is_file())
    result = service.predict_bytes(image.read_bytes())
    record = {"file": str(image.relative_to(PROJECT)), "known_class": expected, **result}
    results.append(record)
    print(record)
(PROJECT / "evidence/task_4").mkdir(parents=True, exist_ok=True)
(PROJECT / "evidence/task_4/inference_demo.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
