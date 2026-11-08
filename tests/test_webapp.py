from io import BytesIO
from pathlib import Path
from PIL import Image
import pytest
from webapp import create_app
from webapp.inference import InvalidImageError


class DummyService:
    model = object(); model_path = Path("dummy.keras"); threshold = .825
    def predict_bytes(self, data):
        try:
            with Image.open(BytesIO(data)) as image: image.verify()
        except Exception as exc: raise InvalidImageError("The uploaded file cannot be decoded as a valid image.") from exc
        return {"label":"Real", "confidence":.8, "ai_probability":.2, "threshold":.825}


@pytest.fixture
def client(tmp_path):
    app = create_app({"TESTING":True, "DATABASE_PATH":tmp_path/"test.sqlite3",
                      "MAX_CONTENT_LENGTH":2048, "MODEL_NAME":"TestDetector",
                      "MODEL_VERSION":"test-v1"}, DummyService())
    return app.test_client()


def png_bytes():
    stream=BytesIO(); Image.new("RGB",(8,8),"red").save(stream,"PNG"); return stream.getvalue()


def test_health(client):
    response=client.get("/health"); assert response.status_code==200; assert response.json["model_loaded"]


def test_valid_upload_and_history(client):
    response=client.post("/predict",data={"image":(BytesIO(png_bytes()),"../../safe.png","image/png")},content_type="multipart/form-data")
    assert response.status_code==200; assert b"Real" in response.data
    history=client.get("/history"); assert b"safe.png" in history.data; assert b"../" not in history.data
    assert b"TestDetector" in history.data; assert b"test-v1" in history.data
    api=client.get("/api/history?label=Real&limit=1"); assert api.status_code==200
    record=api.json["records"][0]; assert record["model_name"]=="TestDetector"
    assert record["model_version"]=="test-v1"; assert record["request_id"]


def test_wrong_type(client):
    response=client.post("/predict",data={"image":(BytesIO(b"text"),"bad.txt","text/plain")},content_type="multipart/form-data")
    assert response.status_code==400; assert b"Unsupported file type" in response.data


def test_fake_image_extension(client):
    response=client.post("/predict",data={"image":(BytesIO(b"not an image"),"fake.png","image/png")},content_type="multipart/form-data")
    assert response.status_code==400; assert b"Invalid image" in response.data


def test_too_large(client):
    response=client.post("/predict",data={"image":(BytesIO(b"x"*5000),"large.png","image/png")},content_type="multipart/form-data")
    assert response.status_code==413; assert b"File too large" in response.data


def test_history_persists_across_app_restart(tmp_path):
    database = tmp_path / "persistent.sqlite3"
    config = {"TESTING":True, "DATABASE_PATH":database, "MAX_CONTENT_LENGTH":2048,
              "MODEL_NAME":"PersistentDetector", "MODEL_VERSION":"v2"}
    first = create_app(config, DummyService()).test_client()
    response = first.post("/predict", data={"image":(BytesIO(png_bytes()),"audit.png","image/png")},
                          content_type="multipart/form-data")
    assert response.status_code == 200
    second = create_app(config, DummyService()).test_client()
    audit = second.get("/api/history").json
    assert audit["count"] == 1
    assert audit["records"][0]["filename"] == "audit.png"
