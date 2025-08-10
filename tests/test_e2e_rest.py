import pytest
from pathlib import Path

# Only required at runtime for this test
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from huggingface_hub import snapshot_download

from spam_bert import main as app

MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

@pytest.mark.e2e
@pytest.mark.slow
def test_rest_e2e_with_real_model(tmp_path: Path, monkeypatch):
    """
    Boots the FastAPI app with a real (tiny) model vendored locally,
    posts a spammy text, and asserts we get a 'spam' decision.
    Skips automatically if offline download fails.
    """
    model_dir = tmp_path / "model"
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
    except Exception as e:
        pytest.skip(f"Could not download model (offline?): {e}")

    # Start the app with our local model directory to avoid network at inference time
    api = app.create_app(
        default_model=MODEL_ID,
        default_threshold=0.5,
        local_dir=str(model_dir),
        model_cache_dir=None
    )
    client = TestClient(api)

    # Health probe
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Classify a very spammy text
    payload = {"text":
               """
               From: "Exclusive Offers" <promo@amazingdealsnow.com>
To: unsuspecting.user@example.com
Subject: Congratulations! You’ve Won a $500 Gift Card 🎁
Date: Sat, 09 Aug 2025 14:23:11 +0000
Message-ID: <20250809142311.987654321@amazingdealsnow.com>
MIME-Version: 1.0
Content-Type: text/plain; charset="UTF-8"
Content-Transfer-Encoding: 7bit
X-Priority: 1 (Highest)
X-Mailer: SuperMailer Pro 4.2

Dear Valued Customer,

We are thrilled to inform you that your email address has been randomly selected 
to receive a **$500 Amazon Gift Card** — absolutely FREE!

👉 To claim your reward, simply click the secure link below:
http://amazingdealsnow.com/claim?winner-id=839274

Hurry! This exclusive offer is valid only for the next 24 hours.  
Don't miss out on this once-in-a-lifetime opportunity to shop for your favorite items at no cost.

Best regards,  
Customer Rewards Department  
Amazing Deals Now

P.S. You must confirm your details to receive the gift card. Unconfirmed claims will be forfeited.

               """}
    r = client.post("/classify", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Structure checks
    for key in ("decision", "spam_probability", "threshold", "chunks", "model_source", "chars_analyzed"):
        assert key in data

    # Behavior: tiny SMS spam model should mark this as spam with high prob
    assert data["decision"] in ("spam", "ham")
    assert 0.0 <= data["spam_probability"] <= 1.0
    assert data["chunks"] >= 1
#    assert "mrm8488" in data["model_source"] or "bert-tiny" in data["model_source"]

    # Strong expectation (relax if it ever flakes):
    assert data["decision"] == "spam"
    assert data["spam_probability"] >= 0.7
