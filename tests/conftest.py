import types
import pytest

@pytest.fixture
def fake_tokenizer():
    """
    A tiny stand-in for HF tokenizer:
    - encode: splits on spaces to make 'tokens'
    - decode: joins tokens back with spaces
    """
    class Tok:
        def encode(self, text, add_special_tokens=False):
            # very naive tokenization: one "token" per word
            return text.split()

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(tokens)
    return Tok()

@pytest.fixture
def fake_pipeline_factory():
    """
    Returns a function that builds a fake pipeline with a fixed spam-prob score.
    """
    def make(score=0.9, label="spam"):
        class Pipe:
            def __call__(self, text, truncation=True, max_length=512):
                return [{"label": label, "score": float(score)}]
        return Pipe()
    return make
