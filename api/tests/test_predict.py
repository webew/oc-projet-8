from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)


def test_predict_schema_and_values():
    payload = {"features": {"DAYS_BIRTH": -12000, "EXT_SOURCE_2": 0.2}}
    r = client.post("/predict", json=payload)
    print(r.status_code, r.text)

    assert r.status_code == 200
    data = r.json()

    assert "approved" in data
    assert "probability" in data
    assert "threshold" in data

    assert 0.0 <= data["probability"] <= 1.0
    print("Data:", data)
    if data["approved"]:
        assert data["probability"] < data["threshold"]
    else:
        assert data["probability"] >= data["threshold"]
