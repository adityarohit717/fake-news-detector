import pytest
import torch
from fastapi.testclient import TestClient

from src.app import MAX_CHARS, THRESHOLD, app


client = TestClient(app)

GOOD = "The government announced a new policy affecting millions of citizens."


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    def fake_run_model(text):
        # Deterministic fake probabilities for CI testing.
        return torch.tensor([0.30, 0.70])

    monkeypatch.setattr("src.app.run_model", fake_run_model)


def test_health():

    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid():

    r = client.post("/predict", json={"text": GOOD})

    assert r.status_code == 200

    d = r.json()

    assert d["prediction"] in {"FAKE", "REAL", "UNCERTAIN"}

    assert abs(
        d["fake_probability"] + d["real_probability"] - 100
    ) < 0.1

    assert 0 <= d["confidence"] <= 100


def test_threshold_logic():

    d = client.post("/predict", json={"text": GOOD}).json()

    top = max(
        d["fake_probability"],
        d["real_probability"]
    )

    if top >= THRESHOLD * 100:
        assert d["prediction"] != "UNCERTAIN"
    else:
        assert d["prediction"] == "UNCERTAIN"


def test_deterministic():

    a = client.post("/predict", json={"text": GOOD}).json()
    b = client.post("/predict", json={"text": GOOD}).json()

    assert a["prediction"] == b["prediction"]
    assert a["fake_probability"] == b["fake_probability"]


def test_empty_text():

    assert client.post(
        "/predict",
        json={"text": ""}
    ).status_code == 400


def test_whitespace_text():

    assert client.post(
        "/predict",
        json={"text": "   \n "}
    ).status_code == 400


def test_missing_field():

    assert client.post(
        "/predict",
        json={}
    ).status_code == 422


def test_text_too_long():

    assert client.post(
        "/predict",
        json={"text": "a" * (MAX_CHARS + 1)}
    ).status_code == 413