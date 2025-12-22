from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()

def test_detect_damage_no_file():
    r = client.post("/detect-damage")
    assert r.status_code in (400, 422)