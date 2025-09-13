import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_image_upload():
    test_image_path = os.path.join(os.path.dirname(__file__), "sample.jpg")
    if not os.path.exists(test_image_path):
        # create a dummy image so tests don’t fail
        with open(test_image_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

    with open(test_image_path, "rb") as img:
        response = client.post(
            "/image-recognition/",
            files={"file": ("sample.jpg", img, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "description" in data

