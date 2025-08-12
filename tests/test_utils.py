from pathlib import Path
import spam_bert.app as app


def test_html_to_text_basic():
    html = "<p>Hello <b>you</b></p>"
    assert app.html_to_text(html) == "Hello you"

def test_clean_email_text_strips_b64_and_quotes():
    txt = "> quoted\nA" + ("A"*50) + "==\nBody"
    cleaned = app.clean_email_text(txt)
    assert "quoted" not in cleaned
    assert "A"*50 not in cleaned
    assert "Body" in cleaned

def test_is_hub_id_vs_path(tmp_path: Path):
    # Existing path -> not a hub id
    p = tmp_path / "models"
    p.mkdir()
    assert app.is_hub_id(str(p)) is False
    # org/name -> hub id
    assert app.is_hub_id("org/model") is True
    # single name -> hub id
    assert app.is_hub_id("bert-base-cased") is True

def test_sanitized_repo_dir():
    assert app.sanitized_repo_dir("org/model") == "org__model"

