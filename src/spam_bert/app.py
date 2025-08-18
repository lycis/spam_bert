#!/usr/bin/env python3
"""
Spam detector: CLI + REST (FastAPI) with controllable model caching.

What’s new:
- --model-cache-dir lets you pick where hub models are stored
- Auto snapshot_download to that folder on first use (no hidden ~/.cache surprises)
- Honors local model directories for offline packaging
- --no-chunk flag to disable chunking and classify with a single truncated pass
- Aggregation strategies for chunked inference: mean, max, median, topk_mean, quantile,
  length_weighted, position_decay, logit_mean, noisy_or, k_of_n, majority_vote

Concurrency fix:
- Avoid sharing a Fast tokenizer across threads (Rust “Already borrowed”).
- We now cache only the model; tokenizer is created per request.
"""

import argparse
import base64
import json
import os
import re
import sys
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List

# Transformers / HF Hub
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import snapshot_download  # hf_hub_url not needed here

# Email parsing
from email import policy
from email.parser import BytesParser

# REST
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None  # CLI still works without REST deps


# ---------- Config ----------
DEFAULT_MODEL = "prancyFox/tiny-bert-enron-spam"
DEFAULT_THRESHOLD = 0.6
# Fallback local dir (used if present and --local-model-dir not provided)
DEFAULT_LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "email-spam"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 32  # tokens
DEFAULT_AGGREGATION = "mean"

# Allow forcing slow (Python) tokenizer if desired
USE_SLOW_TOKENIZER = os.getenv("SPAMBERT_USE_SLOW_TOKENIZER", "0").strip() in {"1", "true", "yes"}


# ---------- Utilities ----------
def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", text).strip()


def clean_email_text(text: str) -> str:
    # strip long base64-ish blobs & quoted lines, collapse whitespace
    text = re.sub(r"[A-Za-z0-9+/=]{40,}", " ", text)
    text = "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith((">", "|"))
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_eml(path: Path) -> str:
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                parts.append(part.get_content())
            elif ctype == "text/html" and not parts:
                parts.append(html_to_text(part.get_content()))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            parts.append(msg.get_content())
        elif ctype == "text/html":
            parts.append(html_to_text(msg.get_content()))

    text = "\n\n".join(p for p in parts if p)
    if not text:
        try:
            body = msg.get_body(preferencelist=("plain", "html"))
            if body:
                text = body.get_content()
                if body.get_content_type() == "text/html":
                    text = html_to_text(text)
        except Exception:
            text = msg.as_string()

    return clean_email_text(text)


def load_text(input_arg: str) -> str:
    p = Path(input_arg)
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".eml":
            return extract_text_from_eml(p)
        else:
            return clean_email_text(p.read_text(encoding="utf-8", errors="ignore"))
    return clean_email_text(input_arg)


def is_hub_id(s: str) -> bool:
    """
    Heuristic: treat strings that look like 'org/name' or 'name' as hub ids
    (and not local paths). If it’s an existing path, it’s not a hub id.
    """
    if Path(s).exists():
        return False
    # crude heuristic: allow letters, digits, dash, underscore, slash
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)?", s))


def sanitized_repo_dir(repo_id: str) -> str:
    # Place each model in a distinct subfolder inside model-cache-dir
    return repo_id.replace("/", "__")


# ---------- Aggregation helpers ----------
def _safe_probs(probs: List[float], eps: float = 1e-6) -> List[float]:
    return [min(max(p, eps), 1.0 - eps) for p in probs]


def aggregate_probs(
    probs: List[float],
    method: str = DEFAULT_AGGREGATION,
    *,
    chunk_lengths: Optional[List[int]] = None,
    k: int = 3,
    p: float = 0.9,
    per_chunk_thr: float = 0.7,
    decay: float = 0.9
) -> float:
    """
    Fuse chunk spam probabilities into a single spam probability.
    Returns a float in [0,1].
    Supported methods: mean, max, median, topk_mean, quantile,
                       length_weighted, position_decay, logit_mean,
                       noisy_or, k_of_n, majority_vote
    """
    if not probs:
        return 0.0
    probs = _safe_probs(probs)
    n = len(probs)
    m = (method or "mean").lower()

    if m == "mean":
        return sum(probs) / n

    if m == "max":
        return max(probs)

    if m == "median":
        s = sorted(probs)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    if m == "topk_mean":
        kk = max(1, min(k, n))
        return sum(sorted(probs, reverse=True)[:kk]) / kk

    if m == "quantile":
        p = min(max(p, 0.0), 1.0)
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return sorted(probs)[idx]

    if m == "length_weighted":
        if not chunk_lengths or len(chunk_lengths) != n:
            return sum(probs) / n
        total = sum(chunk_lengths) or 1
        return sum(p_i * L_i for p_i, L_i in zip(probs, chunk_lengths)) / total

    if m == "position_decay":
        decay = float(decay)
        if decay <= 0.0 or decay > 1.0:
            decay = 0.9
        weights = [decay ** i for i in range(n)]
        sw = sum(weights) or 1e-9
        return sum(p_i * w for p_i, w in zip(probs, weights)) / sw

    if m == "logit_mean":
        logits = [math.log(p_i / (1 - p_i)) for p_i in probs]
        avg_logit = sum(logits) / n
        return 1 / (1 + math.exp(-avg_logit))

    if m == "noisy_or":
        prod_not = 1.0
        for p_i in probs:
            prod_not *= (1.0 - p_i)
        return 1.0 - prod_not

    if m == "k_of_n":
        kk = max(1, min(k, n))
        hits = sum(1 for p_i in probs if p_i >= per_chunk_thr)
        # normalize: 0..1 based on required K
        return min(1.0, max(0.0, hits / kk))

    if m == "majority_vote":
        votes = sum(1 for p_i in probs if p_i >= per_chunk_thr)
        return votes / n

    # default
    return sum(probs) / n


# ---------- Model / Inference ----------
def ensure_local_copy(repo_id: str, cache_root: Path) -> Path:
    """
    Download a hub repo into cache_root/<sanitized_repo> if not present.
    Returns the local directory path.
    """
    target_dir = cache_root / sanitized_repo_dir(repo_id)
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False
    )
    return target_dir


def resolve_model_source(
    model_name: Optional[str],
    local_dir: Optional[Path],
    cache_dir: Optional[Path]
) -> str:
    """
    Decide where to load the model from. Returns a single string:
    - explicit local_dir (if exists), else
    - DEFAULT_LOCAL_MODEL_DIR (if exists), else
    - ensured local copy inside cache_dir (if provided), else
    - hub id (model_name or DEFAULT_MODEL)
    """
    # explicit local dir wins
    if local_dir and local_dir.exists():
        return str(local_dir)

    # default local dir next
    if DEFAULT_LOCAL_MODEL_DIR.exists():
        return str(DEFAULT_LOCAL_MODEL_DIR)

    name = model_name or DEFAULT_MODEL

    # if we have a cache_dir and a hub id, download to cache_dir/<sanitized>
    if cache_dir and is_hub_id(name):
        local_copy = ensure_local_copy(name, cache_dir)
        return str(local_copy)

    # fall back to hub id; HF will use its own cache/TRANSFORMERS_CACHE if needed
    return name


@lru_cache(maxsize=8)
def get_model_cached(model_source: str):
    # Only cache the model (thread-safe). Do NOT cache Fast tokenizer.
    return AutoModelForSequenceClassification.from_pretrained(model_source)


def make_classifier(model_source: str):
    """
    Build a short-lived pipeline using a fresh tokenizer to avoid
    Rust tokenizers' concurrency issues ("Already borrowed").
    """
    tok = AutoTokenizer.from_pretrained(model_source, use_fast=not USE_SLOW_TOKENIZER)
    mdl = get_model_cached(model_source)
    return pipeline("text-classification", model=mdl, tokenizer=tok)


def _normalize_label_prob(result) -> float:
    """Normalize the model output to spam probability."""
    label = result.get("label", "").lower()
    score = float(result.get("score", 0.0))
    if label in {"spam", "label_1"}:
        return score
    elif label in {"ham", "not_spam", "label_0"}:
        return 1.0 - score
    else:
        return score if "spam" in label else 1.0 - score


def classify_text(
    text: str,
    model_name: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    local_model_dir: Optional[str] = None,
    model_cache_dir: Optional[str] = None,
    no_chunk: bool = False,             # disable chunking (single truncated pass)
    aggregate: str = DEFAULT_AGGREGATION,  # aggregation method if chunking
    topk: int = 3,
    quantile: float = 0.9,
    per_chunk_thr: float = 0.7,
    decay: float = 0.9,
) -> Dict[str, Any]:

    model_source = resolve_model_source(
        model_name,
        Path(local_model_dir) if local_model_dir else None,
        Path(model_cache_dir) if model_cache_dir else None
    )

    # Fresh tokenizer per request; cached model under the hood.
    clf = make_classifier(model_source)
    tokenizer = clf.tokenizer

    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Either short text OR explicit no-chunk -> single inference with truncation
    if no_chunk or len(tokens) <= CHUNK_SIZE:
        result = clf(text, truncation=True, max_length=CHUNK_SIZE)[0]
        spam_prob = _normalize_label_prob(result)
        decision = "spam" if spam_prob >= threshold else "ham"
        return {
            "decision": decision,
            "spam_probability": round(spam_prob, 6),
            "threshold": threshold,
            "chunks": 0 if no_chunk else 1,  # 0 indicates forced no-chunk
            "model_source": model_source,
            "chars_analyzed": len(text),
            "no_chunk": bool(no_chunk),
            "aggregation": None,
            "aggregation_params": None,
        }

    # Long text — chunk
    spam_probs: List[float] = []
    chunk_lengths: List[int] = []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + CHUNK_SIZE]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        result = clf(chunk_text, truncation=True, max_length=CHUNK_SIZE)[0]
        spam_probs.append(_normalize_label_prob(result))
        chunk_lengths.append(len(chunk_tokens))

    final_prob = aggregate_probs(
        spam_probs,
        method=aggregate,
        chunk_lengths=chunk_lengths,
        k=topk,
        p=quantile,
        per_chunk_thr=per_chunk_thr,
        decay=decay,
    )
    decision = "spam" if final_prob >= threshold else "ham"

    return {
        "decision": decision,
        "spam_probability": round(final_prob, 6),
        "threshold": threshold,
        "chunks": len(spam_probs),
        "model_source": model_source,
        "chars_analyzed": len(text),
        "no_chunk": False,
        "aggregation": aggregate,
        "aggregation_params": {
            "topk": topk,
            "quantile": quantile,
            "per_chunk_thr": per_chunk_thr,
            "decay": decay,
        },
    }


# ---------- REST Service ----------
class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    eml_base64: Optional[str] = None
    threshold: Optional[float] = None
    model: Optional[str] = None
    no_chunk: Optional[bool] = None
    aggregate: Optional[str] = None
    topk: Optional[int] = None
    quantile: Optional[float] = None
    per_chunk_thr: Optional[float] = None
    decay: Optional[float] = None


def create_app(
    default_model: Optional[str],
    default_threshold: float,
    local_dir: Optional[str],
    model_cache_dir: Optional[str]
) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError("FastAPI not installed. Install with `pip install fastapi uvicorn`.")
    app = FastAPI(title="Spam Detector (BERT)", version="1.3.1")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model")
    def model_info():
        src = resolve_model_source(
            default_model,
            Path(local_dir) if local_dir else None,
            Path(model_cache_dir) if model_cache_dir else None
        )
        return {
            "default_model": default_model,
            "resolved_source": src,
            "threshold": default_threshold,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "default_aggregation": DEFAULT_AGGREGATION,
            "use_slow_tokenizer": USE_SLOW_TOKENIZER,
        }

    @app.post("/classify")
    def classify(req: ClassifyRequest):
        if not (req.text or req.eml_base64):
            raise HTTPException(status_code=400, detail="Provide 'text' or 'eml_base64'.")

        if req.eml_base64 and not req.text:
            try:
                raw = base64.b64decode(req.eml_base64)
                tmp = Path("__tmp__.eml")
                tmp.write_bytes(raw)
                try:
                    txt = extract_text_from_eml(tmp)
                finally:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid eml_base64: {e}")
        else:
            txt = req.text or ""

        out = classify_text(
            txt,
            model_name=req.model or default_model,
            threshold=req.threshold if req.threshold is not None else default_threshold,
            local_model_dir=local_dir,
            model_cache_dir=model_cache_dir,
            no_chunk=bool(req.no_chunk),
            aggregate=(req.aggregate or DEFAULT_AGGREGATION),
            topk=int(req.topk) if req.topk is not None else 3,
            quantile=float(req.quantile) if req.quantile is not None else 0.9,
            per_chunk_thr=float(req.per_chunk_thr) if req.per_chunk_thr is not None else 0.7,
            decay=float(req.decay) if req.decay is not None else 0.9,
        )
        return out

    return app


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Spam detector (BERT) - CLI & REST")
    ap.add_argument("input", nargs="?", help="Path to .eml/.txt or raw text (omit when using --serve).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--local-model-dir", default=None, help="Path to a local model directory (for offline use).")
    ap.add_argument("--model-cache-dir", default=None, help="Directory to store/download hub models (overrides default HF cache).")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Spam threshold (default: {DEFAULT_THRESHOLD})")

    # Chunking & aggregation controls
    ap.add_argument("--no-chunk", action="store_true", help="Disable chunking (truncate to model max length).")
    ap.add_argument("--aggregate", choices=[
        "mean","max","median","topk_mean","quantile",
        "length_weighted","position_decay","logit_mean",
        "noisy_or","k_of_n","majority_vote"
    ], default=DEFAULT_AGGREGATION, help="Aggregation method for chunked inference.")
    ap.add_argument("--topk", type=int, default=3, help="K for topk_mean / k_of_n aggregation.")
    ap.add_argument("--quantile", type=float, default=0.9, help="Quantile (0..1) for quantile aggregation.")
    ap.add_argument("--per-chunk-thr", type=float, default=0.7, help="Per-chunk threshold for vote/k_of_n.")
    ap.add_argument("--decay", type=float, default=0.9, help="Decay factor (0<d<=1) for position_decay aggregation.")

    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    ap.add_argument("--serve", action="store_true", help="Run as REST service.")
    ap.add_argument("--host", default="0.0.0.0", help="REST host (default: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8000, help="REST port (default: 8000)")

    args = ap.parse_args()

    # Optionally set TRANSFORMERS_CACHE to the same directory for tokenizer/aux files
    if args.model_cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.model_cache_dir).resolve()))

    if args.serve:
        if FastAPI is None:
            print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
            sys.exit(2)
        import uvicorn
        app = create_app(args.model, args.threshold, args.local_model_dir, args.model_cache_dir)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if not args.input:
        print("Provide input text or path (or use --serve).", file=sys.stderr)
        sys.exit(2)

    try:
        text = load_text(args.input)
        if not text:
            print("No text extracted from input.", file=sys.stderr)
            sys.exit(2)
        out = classify_text(
            text,
            model_name=args.model,
            threshold=args.threshold,
            local_model_dir=args.local_model_dir,
            model_cache_dir=args.model_cache_dir,
            no_chunk=args.no_chunk,
            aggregate=args.aggregate,
            topk=args.topk,
            quantile=args.quantile,
            per_chunk_thr=args.per_chunk_thr,
            decay=args.decay,
        )
        print(json.dumps(out, indent=2 if args.pretty else None))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
