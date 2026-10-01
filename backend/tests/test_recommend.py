from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "text": "quiero algo de ciencia ficción pero con tensión ❤️",
    "media_types": ["movie", "tv", "anime"],
    "liked_ids": [],
}


def test_recommend_returns_contract():
    response = client.post("/recommend", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"results"}
    assert body["results"]

    for item in body["results"]:
        assert set(item) == {
            "id",
            "title",
            "media_type",
            "year",
            "genres",
            "poster_url",
            "score",
            "explanation",
        }
        assert isinstance(item["id"], str)
        assert isinstance(item["title"], str)
        assert item["media_type"] in {"movie", "tv", "anime"}
        assert item["year"] is None or isinstance(item["year"], int)
        assert isinstance(item["genres"], list)
        assert item["poster_url"] is None or isinstance(item["poster_url"], str)
        assert 0 <= item["score"] <= 1
        assert isinstance(item["explanation"], str)


def test_recommend_filters_by_media_type():
    response = client.post(
        "/recommend",
        json={**VALID_PAYLOAD, "media_types": ["anime"]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {item["media_type"] for item in results} == {"anime"}


def test_recommend_rejects_empty_text():
    response = client.post("/recommend", json={**VALID_PAYLOAD, "text": ""})

    assert response.status_code == 422


def test_recommend_rejects_whitespace_only_text():
    response = client.post("/recommend", json={**VALID_PAYLOAD, "text": "   "})

    assert response.status_code == 422


def test_recommend_rejects_invalid_media_type():
    response = client.post(
        "/recommend",
        json={**VALID_PAYLOAD, "media_types": ["documental"]},
    )

    assert response.status_code == 422


def test_recommend_rejects_empty_media_types():
    response = client.post("/recommend", json={**VALID_PAYLOAD, "media_types": []})

    assert response.status_code == 422


def test_recommend_limits_results_to_six():
    response = client.post("/recommend", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert len(response.json()["results"]) <= 6
