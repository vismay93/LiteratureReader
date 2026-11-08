import hashlib
import json

import pytest
from fastapi.testclient import TestClient

import config

PDF = b"%PDF-1.4 fake paper"
PID = hashlib.sha256(PDF).hexdigest()[:16]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "lib")
    d = config.DATA_DIR / PID
    d.mkdir(parents=True)
    (d / "source.pdf").write_bytes(PDF)
    (d / "doc.json").write_text(json.dumps({"id": PID, "title": "T", "pages": [], "blocks": [
        {"id": "p0b0", "page": 0, "type": "text", "bbox": [0, 0, 1, 1], "text": "hi"}]}))
    import app
    with TestClient(app.app, base_url="http://127.0.0.1") as c:
        yield c


def test_full_mode_info(client, monkeypatch):
    monkeypatch.setattr(config, "READ_ONLY", False)
    assert client.get("/api/info").json() == {"read_only": False}


def test_read_only_serves_but_does_not_process(client, monkeypatch):
    monkeypatch.setattr(config, "READ_ONLY", True)
    assert client.get("/api/info").json() == {"read_only": True}
    assert [p["paper_id"] for p in client.get("/api/papers").json()] == [PID]
    assert client.get(f"/api/papers/{PID}").json()["title"] == "T"
    assert client.get(f"/api/papers/{PID}/pdf").content == PDF
    r = client.post("/api/papers", data={"arxiv": "1706.03762"})
    assert r.status_code == 403 and "Read-only" in r.json()["detail"]
    assert client.post(f"/api/papers/{PID}/blocks/p0b0/regenerate").status_code == 403
