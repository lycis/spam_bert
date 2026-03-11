#!/usr/bin/env python3
"""Sweep synthetic-to-real augmentation ratios for mixed spam training."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_BASE_MODEL_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--prancyFox--tiny-bert-enron-spam" / "snapshots" / "97eef925bf7d77b00ced4e0ed00eb81eba42ebe9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-train", default="train/train_processed.csv")
    parser.add_argument("--real-val", default="train/val_processed.csv")
    parser.add_argument("--synthetic", default="train/synthetic_data.csv")
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--base-model-dir", default=str(DEFAULT_BASE_MODEL_DIR))
    parser.add_argument("--output-root", default="train/outputs/ratio_sweep")
    parser.add_argument("--data-root", default="train/ratio_sweep_data")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def slugify_ratio(ratio: float) -> str:
    text = f"{ratio:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_dataset(real_train_path: Path, synthetic_path: Path, ratio: float, output_path: Path) -> dict:
    real_df = pd.read_csv(real_train_path)
    synth_df = pd.read_csv(synthetic_path)

    synthetic_spam = synth_df[synth_df["label"] == 1].reset_index(drop=True)
    synthetic_rows = min(len(synthetic_spam), int(math.floor(len(real_df) * ratio)))
    mixed_df = pd.concat([real_df, synthetic_spam.head(synthetic_rows)], ignore_index=True)
    mixed_df = mixed_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_df.to_csv(output_path, index=False)

    counts = mixed_df["label"].value_counts().to_dict()
    return {
        "rows": len(mixed_df),
        "synthetic_rows": synthetic_rows,
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "spam_fraction": float((mixed_df["label"] == 1).mean()),
        "path": str(output_path),
    }


def run_training(train_csv: Path, val_csv: Path, base_model_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    repo_root = Path.cwd()
    env["HF_HOME"] = str(repo_root / ".hf_cache" / "huggingface")
    env["HF_METRICS_CACHE"] = str(repo_root / ".hf_cache" / "metrics")
    env["HF_DATASETS_CACHE"] = str(repo_root / ".hf_cache" / "datasets")
    env["HF_MODULES_CACHE"] = str(Path.home() / ".cache" / "huggingface" / "modules")
    env["TOKENIZERS_PARALLELISM"] = "false"

    cmd = [
        sys.executable,
        "train/train_tinybert.py",
        "--train",
        str(train_csv),
        "--val",
        str(val_csv),
        "--local_model_dir",
        str(base_model_dir),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_length",
        str(args.max_length),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, env=env)


def evaluate_model(model_dir: Path, data_path: Path, max_length: int) -> dict:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    preds = []
    probs = []
    batch_size = 32

    for index in range(0, len(texts), batch_size):
        batch = texts[index:index + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        probs.extend(batch_probs)
        preds.extend(batch_preds)

    accuracy = accuracy_score(labels, preds)
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "rows": len(df),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "ham": {
            "precision": float(prec[0]),
            "recall": float(rec[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "spam": {
            "precision": float(prec[1]),
            "recall": float(rec[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix_labels_[ham,spam]": cm.tolist(),
        "avg_spam_probability": float(sum(probs) / len(probs)),
    }


def main() -> None:
    args = parse_args()
    real_train_path = Path(args.real_train)
    real_val_path = Path(args.real_val)
    synthetic_path = Path(args.synthetic)
    base_model_dir = Path(args.base_model_dir)
    output_root = Path(args.output_root)
    data_root = Path(args.data_root)

    output_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    all_results = []
    for ratio in args.ratios:
        ratio_slug = slugify_ratio(ratio)
        train_csv = data_root / f"mixed_ratio_{ratio_slug}.csv"
        output_dir = output_root / f"ratio_{ratio_slug}"
        dataset_info = build_dataset(real_train_path, synthetic_path, ratio, train_csv)
        run_training(train_csv, real_val_path, base_model_dir, output_dir, args)
        evaluation = evaluate_model(output_dir, real_val_path, args.max_length)
        record = {
            "ratio": ratio,
            "dataset": dataset_info,
            "validation": evaluation,
            "output_dir": str(output_dir),
        }
        all_results.append(record)

    ranked = sorted(
        all_results,
        key=lambda item: (
            item["validation"]["macro_f1"],
            item["validation"]["spam"]["f1"],
            item["validation"]["accuracy"],
        ),
        reverse=True,
    )

    summary = {
        "best_ratio": ranked[0]["ratio"],
        "best_output_dir": ranked[0]["output_dir"],
        "ranked_results": ranked,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
