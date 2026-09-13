from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_toxicity_rejects_empty_text():
    response = client.post("/toxicity", json={"text": ""})
    assert response.status_code == 400


def test_toxicity_returns_valid_shape():
    # NOTE: hits the real classifier model - slower, needs internet on
    # first run to download weights. Good enough for a hackathon suite.
    response = client.post(
        "/toxicity", json={"text": "You are so stupid, get lost."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in ("SAFE", "WARN")
    assert 0.0 <= data["toxicity_score"] <= 1.0
