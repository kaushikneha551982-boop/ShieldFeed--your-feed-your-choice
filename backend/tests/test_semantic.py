from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_semantic_filter_rejects_empty_posts():
    response = client.post(
        "/semantic-filter", json={"preference": "career anxiety", "posts": []}
    )
    assert response.status_code == 400


def test_semantic_filter_empty_preference_shows_everything():
    response = client.post(
        "/semantic-filter",
        json={
            "preference": "",
            "posts": [{"id": 1, "text": "Random unrelated post."}],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["action"] == "SHOW"


def test_semantic_filter_flags_semantically_related_post():
    # NOTE: this hits the real embedding model, so it is slower than a
    # typical unit test and needs internet access on first run
    # (to download the model). That's fine for a hackathon test suite.
    response = client.post(
        "/semantic-filter",
        json={
            "preference": "I want to avoid placement rejection and career anxiety.",
            "posts": [
                {
                    "id": 1,
                    "text": "Another student failed to clear their final interview.",
                },
                {"id": 2, "text": "Our college football team won today's match."},
            ],
        },
    )
    assert response.status_code == 200
    results = {r["post_id"]: r for r in response.json()["results"]}
    # The interview-failure post is semantically closer to "placement
    # rejection" even though it shares no keywords with the preference.
    assert results[1]["similarity"] > results[2]["similarity"]
