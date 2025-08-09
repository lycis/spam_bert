import base64
from pathlib import Path
import pytest
import spam_bert as app
import httpx

def test_extract_text_from_eml_plain(tmp_path: Path):
    eml = (
        "From: a@b\nTo: c@d\nSubject: Hi\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Hello there."
    ).encode()
    f = tmp_path / "x.eml"
    f.write_bytes(eml)
    txt = app.extract_text_from_eml(f)
    assert "Hello there." in txt

def test_extract_text_from_eml_html(tmp_path: Path):
    eml = (
        "From: a@b\nTo: c@d\nSubject: Hi\n"
        "Content-Type: text/html; charset=utf-8\n\n"
        "<p>Hello <b>you</b></p>"
    ).encode()
    f = tmp_path / "x.eml"
    f.write_bytes(eml)
    txt = app.extract_text_from_eml(f)
    assert "Hello you" in txt

def test_rest_classify_smoke(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # stub classify_text so REST stays fast
    def fake_classify(text, model_name=None, threshold=0.6, local_model_dir=None, model_cache_dir=None):
        return {
            "decision": "ham",
            "spam_probability": 0.1,
            "threshold": threshold,
            "chunks": 1,
            "model_source": model_name or "x",
            "chars_analyzed": len(text),
        }

    monkeypatch.setattr(app, "classify_text", fake_classify)
    api = app.create_app(default_model="org/model", default_threshold=0.6, local_dir=None, model_cache_dir=None)
    client = TestClient(api)

    r = client.post("/classify", json={"text": "hello world"})
    assert r.status_code == 200
    assert r.json()["decision"] == "ham"
