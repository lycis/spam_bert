#!/usr/bin/env python3
"""Tune classification thresholds for a trained spam model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def predict_probs(model_dir: Path, data_path: Path, max_length: int) -> tuple[list[int], list[float]]:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    probs: list[float] = []
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
        probs.extend(batch_probs)

    return labels, probs


def metrics_at_threshold(labels: list[int], probs: list[float], threshold: float) -> dict:
    preds = [1 if prob >= threshold else 0 for prob in probs]
    accuracy = accuracy_score(labels, preds)
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "threshold": threshold,
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
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    data_path = Path(args.data)
    output_path = Path(args.output) if args.output else model_dir / "threshold_tuning.json"

    labels, probs = predict_probs(model_dir, data_path, args.max_length)

    thresholds = sorted({round(i / 100, 2) for i in range(5, 96)} | {round(prob, 6) for prob in probs})
    results = [metrics_at_threshold(labels, probs, threshold) for threshold in thresholds]

    best_spam_f1 = max(results, key=lambda item: (item["spam"]["f1"], item["spam"]["recall"], -item["false_positives"]))
    best_macro_f1 = max(results, key=lambda item: (item["macro_f1"], item["spam"]["f1"]))

    fp_le_one = [item for item in results if item["false_positives"] <= 1]
    best_low_fp = max(
        fp_le_one,
        key=lambda item: (item["spam"]["recall"], item["spam"]["f1"], item["accuracy"]),
    ) if fp_le_one else None

    fp_le_two = [item for item in results if item["false_positives"] <= 2]
    best_fp_two = max(
        fp_le_two,
        key=lambda item: (item["spam"]["recall"], item["spam"]["f1"], item["accuracy"]),
    ) if fp_le_two else None

    summary = {
        "model_dir": str(model_dir),
        "data": str(data_path),
        "rows": len(labels),
        "spam_rows": int(sum(labels)),
        "ham_rows": int(len(labels) - sum(labels)),
        "best_spam_f1": best_spam_f1,
        "best_macro_f1": best_macro_f1,
        "best_with_false_positives_le_1": best_low_fp,
        "best_with_false_positives_le_2": best_fp_two,
        "all_thresholds": results,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
