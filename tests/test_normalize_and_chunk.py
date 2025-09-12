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