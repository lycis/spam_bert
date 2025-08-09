from pathlib import Path
import spam_bert as app

def test_resolve_prefers_explicit_local(tmp_path: Path, monkeypatch):
    local = tmp_path / "local_model"
    local.mkdir()
    (local / "dummy").write_text("ok")
    resolved = app.resolve_model_source("org/model", local, None)
    assert resolved == str(local)

def test_resolve_uses_default_local_when_present(tmp_path: Path, monkeypatch):
    # fake DEFAULT_LOCAL_MODEL_DIR to a temp dir
    default_local = tmp_path / "default_local"
    default_local.mkdir()
    (default_local / "ok").write_text("x")
    monkeypatch.setattr(app, "DEFAULT_LOCAL_MODEL_DIR", default_local)
    resolved = app.resolve_model_source("org/model", None, None)
    assert resolved == str(default_local)

def test_resolve_downloads_to_cache_when_requested(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache_root"
    cache.mkdir()

    # monkeypatch ensure_local_copy to avoid real download
    called = {}
    def fake_ensure(repo_id, cache_root):
        called["repo_id"] = repo_id
        path = cache_root / app.sanitized_repo_dir(repo_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        return path

    monkeypatch.setattr(app, "ensure_local_copy", fake_ensure)
    resolved = app.resolve_model_source("org/model", None, cache)
    assert Path(resolved).exists()
    assert called["repo_id"] == "org/model"
