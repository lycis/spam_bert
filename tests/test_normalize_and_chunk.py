import pytest
import spam_bert.app as app

def test_normalize_label_prob_variants():
    assert app._normalize_label_prob({"label":"spam","score":0.7}) == 0.7
    assert app._normalize_label_prob({"label":"label_1","score":0.3}) == 0.3
    assert app._normalize_label_prob({"label":"ham","score":0.8}) == pytest.approx(0.2, rel=1e-9)
    assert app._normalize_label_prob({"label":"not_spam","score":0.1}) == 0.9
    # weird label still handled (fallback by substring)
    assert app._normalize_label_prob({"label":"maybe_spam","score":0.4}) == 0.4
    assert app._normalize_label_prob({"label":"unknown","score":0.6}) == 0.4

def test_classify_with_no_chunk(monkeypatch):
    class Tok:
        def encode(self, t, add_special_tokens=False):  # would be long, but no_chunk forces single pass
            return list(range(1500))
    def fake_pipe(_src):
        def run(txt, truncation=True, max_length=512):
            return [{"label":"spam","score":0.75}]
        return run

    monkeypatch.setattr(app, "get_tokenizer", lambda s: Tok())
    monkeypatch.setattr(app, "get_pipeline_cached", lambda s: fake_pipe(s))
    out = app.classify_text("hello world " * 1000, model_name="x", no_chunk=True)
    assert out["decision"] == "spam"
    assert out["no_chunk"] is True
    assert out["chunks"] == 0

def test_classify_long_text_chunking_average(monkeypatch):
    # Make ~1200 "tokens" so we get multiple chunks
    class Tok:
        def encode(self, t, add_special_tokens=False):
            return list(range(1200))
        def decode(self, toks, skip_special_tokens=True):
            return "chunk"
    # Always return score 0.6 -> average stays 0.6
    def fake_pipe(_src):
        def run(txt, truncation=True, max_length=512):
            return [{"label":"spam","score":0.6}]
        return run

    monkeypatch.setattr(app, "get_tokenizer", lambda s: Tok())
    monkeypatch.setattr(app, "get_pipeline_cached", lambda s: fake_pipe(s))
    out = app.classify_text("x " * 2000, model_name="x", threshold=0.5, no_chunk=False)
    assert out["decision"] == "spam"
    assert out["no_chunk"] is False
    assert out["chunks"] >= 2
    assert 0.59 <= out["spam_probability"] <= 0.61
