from pathlib import Path
import spam_bert.app as app


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

def test_ensure_local_copy_downloads(monkeypatch, tmp_path):
    calls = {"snap": 0}
    def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks):
        calls["snap"] += 1
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir)/"config.json").write_text("{}")
    monkeypatch.setattr(app, "snapshot_download", fake_snapshot_download)

    target = app.ensure_local_copy("org/model", tmp_path)
    assert target.exists()
    assert calls["snap"] == 1

def test_ensure_local_copy_reuses(monkeypatch, tmp_path):
    # pre-create directory with a file so it reuses and doesn't call snapshot_download
    pre = tmp_path / app.sanitized_repo_dir("org/model")
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "ok").write_text("x")

    called = {"snap": 0}
    def fake_snapshot_download(*a, **k): called["snap"] += 1
    monkeypatch.setattr(app, "snapshot_download", fake_snapshot_download)

    target = app.ensure_local_copy("org/model", tmp_path)
    assert target == pre
    assert called["snap"] == 0

def test_resolve_model_uses_cache_when_hub_id(tmp_path, monkeypatch):
    # point to empty cache; simulate download
    def fake_ensure(repo_id, cache_root):
        path = cache_root / app.sanitized_repo_dir(repo_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}")
        return path
    monkeypatch.setattr(app, "ensure_local_copy", fake_ensure)
    resolved = app.resolve_model_source("org/model", None, tmp_path)
    assert Path(resolved).exists()
