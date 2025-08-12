import base64
import pytest
import spam_bert.app as app
from fastapi.testclient import TestClient

def test_rest_400_when_missing_body():
    api = app.create_app(default_model="org/model", default_threshold=0.6, local_dir=None, model_cache_dir=None)
    client = TestClient(api)
    r = client.post("/classify", json={})
    assert r.status_code == 400
    assert "Provide 'text' or 'eml_base64'" in r.text

def test_rest_400_invalid_base64():
    api = app.create_app(default_model="org/model", default_threshold=0.6, local_dir=None, model_cache_dir=None)
    client = TestClient(api)
    r = client.post("/classify", json={"eml_base64": "not-a-real-b64!!!"})
    assert r.status_code == 400

def test_rest_no_chunk_passthrough(monkeypatch):
    captured = {}
    def fake_classify(text, **kwargs):
        captured.update(kwargs)
        return {
            "decision": "ham", "spam_probability": 0.1, "threshold": kwargs.get("threshold", 0.6),
            "chunks": 1, "model_source": kwargs.get("model_name","x"), "chars_analyzed": len(text),
            "no_chunk": kwargs.get("no_chunk", False),
        }
    monkeypatch.setattr(app, "classify_text", fake_classify)

    api = app.create_app(default_model="org/model", default_threshold=0.6, local_dir=None, model_cache_dir=None)
    client = TestClient(api)
    r = client.post("/classify", json={"text":"hey", "no_chunk": True})
    assert r.status_code == 200
    assert r.json()["no_chunk"] is True
    assert captured.get("no_chunk") is True

def test_model_endpoint_smoke():
    api = app.create_app(default_model="org/model", default_threshold=0.6, local_dir=None, model_cache_dir=None)
    client = TestClient(api)
    r = client.get("/model")
    assert r.status_code == 200
    assert "resolved_source" in r.json()
