import json
import sys
import builtins
import spam_bert.app as app
import pytest

def test_cli_no_input_exits(monkeypatch, capsys):
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit) as ex:
        app.main()
    assert ex.value.code == 2
    out = capsys.readouterr().err
    assert "Provide input text or path" in out

def test_cli_pretty_json(monkeypatch, capsys):
    # avoid file IO by faking load_text & classify_text
    monkeypatch.setattr(app, "load_text", lambda x: "hello")
    monkeypatch.setattr(sys, "argv", ["prog", "hello", "--pretty", "--threshold", "0.5"])
    def fake_classify(text, **kwargs):
        return {"decision":"ham","spam_probability":0.42,"threshold":kwargs.get("threshold",0.6),
                "chunks":1,"model_source":kwargs.get("model_name","x"),"chars_analyzed":len(text)}
    monkeypatch.setattr(app, "classify_text", fake_classify)
    app.main()
    out = capsys.readouterr().out
    # pretty JSON should include newlines/indentation
    assert out.strip().startswith("{")
    assert "\n  " in out
    data = json.loads(out)
    assert data["decision"] == "ham"
