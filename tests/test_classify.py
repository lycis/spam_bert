from pathlib import Path
import spam_bert as app

def test_classify_short_text(monkeypatch, fake_tokenizer, fake_pipeline_factory):
    # patch tokenizer & pipeline cache
    monkeypatch.setattr(app, "get_tokenizer", lambda src: fake_tokenizer)
    monkeypatch.setattr(app, "get_pipeline_cached", lambda src: fake_pipeline_factory(score=0.85, label="spam"))

    out = app.classify_text("free voucher now", model_name="org/model", threshold=0.6)
    assert out["decision"] == "spam"
    assert out["chunks"] == 1
    assert out["spam_probability"] == 0.85

def test_classify_long_text_chunking_average(monkeypatch, fake_tokenizer, fake_pipeline_factory):
    # Create 1200 "tokens" by making 1200 words
    long_text = "word " * 1200

    # Use fixed 0.75 spam score for every chunk
    monkeypatch.setattr(app, "get_tokenizer", lambda src: fake_tokenizer)
    monkeypatch.setattr(app, "get_pipeline_cached", lambda src: fake_pipeline_factory(score=0.75, label="spam"))

    out = app.classify_text(long_text, model_name="org/model", threshold=0.6)
    # CHUNK_SIZE=512, OVERLAP=32 => step 480 → starts at 0, 480, 960 => 3 chunks
    assert out["chunks"] == 3
    assert out["decision"] == "spam"
    assert out["spam_probability"] == 0.75
