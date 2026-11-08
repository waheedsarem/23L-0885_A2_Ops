"""Exercise the live Task 4 server and save exact rendered responses for screenshots."""
from pathlib import Path
import json
import requests

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "evidence" / "task_4" / "pages"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8000"


def save_response(name, response):
    html = response.text.replace('href="/static/', f'href="{BASE}/static/')
    (OUT / f"{name}.html").write_text(html, encoding="utf-8")
    return {"case": name, "status": response.status_code, "bytes": len(response.content)}


records = []
records.append(save_response("frontend", requests.get(f"{BASE}/", timeout=20)))
real_image = next((PROJECT / "dataset/raw/RealArt/RealArt").glob("*"))
with real_image.open("rb") as stream:
    records.append(save_response("valid_result", requests.post(
        f"{BASE}/predict", files={"image": (real_image.name, stream, "image/jpeg")}, timeout=120)))
records.append(save_response("history", requests.get(f"{BASE}/history", timeout=20)))
records.append(save_response("wrong_type", requests.post(
    f"{BASE}/predict", files={"image": ("notes.txt", b"not an image", "text/plain")}, timeout=20)))
records.append(save_response("corrupt_image", requests.post(
    f"{BASE}/predict", files={"image": ("looks-real.png", b"not a decodable image", "image/png")}, timeout=20)))
records.append(save_response("too_large", requests.post(
    f"{BASE}/predict", files={"image": ("huge.png", b"x" * (6 * 1024 * 1024), "image/png")}, timeout=30)))
health = requests.get(f"{BASE}/health", timeout=20)
(OUT / "health.json").write_text(json.dumps(health.json(), indent=2), encoding="utf-8")
records.append({"case": "health", "status": health.status_code, **health.json()})
(PROJECT / "evidence/task_4/live_route_checks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
