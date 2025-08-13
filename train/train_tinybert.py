#!/usr/bin/env python3
"""
Fine-tune TinyBERT (4L-312D) for spam vs ham.

Input data:
- CSV or JSON Lines with columns: text, label
  - label can be {spam, ham} or {1, 0} (case-insensitive)

Examples:
  python train_tinybert.py \
    --train data/train.csv --val data/val.csv \
    --output_dir outputs/tinybert-spam \
    --epochs 3 --batch_size 32 --lr 3e-5 --max_length 256

Optional:
  --model_cache_dir ./models_cache
  --local_model_dir ./models/tinybert     # offline, previously downloaded
  --use_focal_loss
  --class_weight 1.0  # weight for positive class (spam=1)
  --push_to_hub --hub_repo youruser/spam-tinybert
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset, ClassLabel
from evaluate import load as load_metric

DEFAULT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"
LABELS = ["ham", "spam"]  # index 0 -> ham, 1 -> spam


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True, help="train file (csv or jsonl)")
    ap.add_argument("--val", type=str, default=None, help="val file (csv or jsonl). If omitted, we split from train.")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")

    ap.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--local_model_dir", type=str, default=None, help="Path to local model dir (offline).")
    ap.add_argument("--model_cache_dir", type=str, default=None, help="HF cache dir for downloads.")

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_focal_loss", action="store_true")
    ap.add_argument("--class_weight", type=float, default=None, help="Weight for positive class (spam=1). If not set, computed from data.")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo", type=str, default=None)
    return ap.parse_args()


def read_any(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".json", ".jsonl"]:
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"Unsupported file extension for {path}")

    for col in [text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}. Found: {list(df.columns)}")

    # normalize labels
    def norm_label(v):
        if isinstance(v, str):
            v = v.strip().lower()
            if v in {"spam", "1", "true", "yes"}:
                return 1
            if v in {"ham", "0", "false", "no"}:
                return 0
        try:
            iv = int(v)
            return 1 if iv == 1 else 0
        except Exception:
            pass
        raise ValueError(f"Unrecognized label value: {v}")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].map(norm_label)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    return df


def make_datasets(args) -> Tuple[Dataset, Dataset]:
    df_train = read_any(args.train, args.text_col, args.label_col)

    if args.val:
        df_val = read_any(args.val, args.text_col, args.label_col)
    else:
        # stratified split
        df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=args.seed, stratify=df_train["label"])

    # to HF datasets
    train_ds = Dataset.from_pandas(df_train, preserve_index=False)
    val_ds = Dataset.from_pandas(df_val, preserve_index=False)

    # attach ClassLabel feature for neatness
    cl = ClassLabel(names=LABELS)
    train_ds = train_ds.cast_column("label", cl)
    val_ds = val_ds.cast_column("label", cl)
    return train_ds, val_ds


def get_model_and_tokenizer(model_name_or_path: str, cache_dir: Optional[str], gradient_checkpointing=False):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    cfg = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
        cache_dir=cache_dir
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, config=cfg, cache_dir=cache_dir
    )
    if gradient_checkpointing:
        mdl.gradient_checkpointing_enable()
    return mdl, tok


def tokenize_function(examples, tokenizer, max_length: int):
    # If Subject exists, concatenate it for more context
    if "Subject" in examples:
        texts = [
            ((s or "") + "\n" + (t or "")).strip()
            for s, t in zip(examples.get("Subject", [""]*len(examples["text"])), examples["text"])
        ]
    else:
        texts = examples["text"]
    return tokenizer(texts, truncation=True, max_length=max_length)


class WeightedOrFocalLossTrainer(Trainer):
    def __init__(self, use_focal=False, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.use_focal = use_focal
        self.class_weight = class_weight  # tensor([w0, w1]) or None
        if self.class_weight is not None:
            self.class_weight = self.class_weight.to(self.model.device)

    # accept **kwargs to be compatible with newer Trainer (e.g., num_items_in_batch)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        if self.use_focal:
            # gamma=2 focal loss
            ce = nn.CrossEntropyLoss(reduction="none")(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            gamma = 2.0
            fl = (1 - pt) ** gamma * ce
            loss = fl.mean()
        else:
            if self.class_weight is not None:
                loss = nn.CrossEntropyLoss(weight=self.class_weight)(logits, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_fn():
    acc = load_metric("accuracy")
    prec = load_metric("precision")
    rec = load_metric("recall")
    f1 = load_metric("f1")

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "precision": prec.compute(predictions=preds, references=labels, average="binary")["precision"],
            "recall": rec.compute(predictions=preds, references=labels, average="binary")["recall"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }
    return compute


def main():
    args = parse_args()
    set_seed(args.seed)

    # allow explicit cache override
    if args.model_cache_dir:
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.model_cache_dir).resolve()))

    model_source = args.local_model_dir if args.local_model_dir and Path(args.local_model_dir).exists() else args.model_name

    train_ds, val_ds = make_datasets(args)
    model, tokenizer = get_model_and_tokenizer(model_source, args.model_cache_dir, args.gradient_checkpointing)

    # tokenize
    train_tok = train_ds.map(lambda e: tokenize_function(e, tokenizer, args.max_length), batched=True, remove_columns=["text"])
    val_tok = val_ds.map(lambda e: tokenize_function(e, tokenizer, args.max_length), batched=True, remove_columns=["text"])

    # class weights (if not focal)
    cw_tensor = None
    if not args.use_focal_loss:
        if args.class_weight is not None:
            # weight vector is [w_ham, w_spam]; allow user to pass single positive class weight
            cw_tensor = torch.tensor([1.0, float(args.class_weight)], dtype=torch.float)
        else:
            # compute from training labels
            y = np.array(train_ds["label"])
            weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
            cw_tensor = torch.tensor(weights, dtype=torch.float)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    # ===== Compatibility shim for TrainingArguments =====
    import inspect
    ta_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size // 2),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",   # new API
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        report_to=["none"],  # set to ["wandb"] if you like
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo if args.hub_repo else None,
    )
    sig = inspect.signature(TrainingArguments)

    # If 'evaluation_strategy' isn't supported, fall back to legacy options
    if "evaluation_strategy" not in sig.parameters:
        ta_kwargs.pop("evaluation_strategy", None)
        ta_kwargs.setdefault("eval_steps", 200)
        ta_kwargs["evaluate_during_training"] = True  # legacy flag for older versions
        # prune newer-only fields old versions don't know
        for k in ("save_strategy", "metric_for_best_model", "greater_is_better",
                  "load_best_model_at_end", "report_to", "hub_model_id"):
            ta_kwargs.pop(k, None)

    # Finally, drop anything else the installed version doesn't accept
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig.parameters}
    training_args = TrainingArguments(**ta_kwargs)

    # --- Ensure EarlyStopping has a metric + an evaluation & save schedule ---
    # 1) Metric to watch
    if not hasattr(training_args, "metric_for_best_model") or training_args.metric_for_best_model is None:
        setattr(training_args, "metric_for_best_model", "eval_f1")
    if not hasattr(training_args, "greater_is_better"):
        setattr(training_args, "greater_is_better", True)

    # 2) Evaluation schedule (cover old/new APIs)
    try:
        from transformers.trainer_utils import IntervalStrategy
        if hasattr(training_args, "evaluation_strategy") and str(training_args.evaluation_strategy).lower() in {"no", "none"}:
            training_args.evaluation_strategy = IntervalStrategy.STEPS
        if hasattr(training_args, "eval_strategy") and str(training_args.eval_strategy).lower() in {"no", "none"}:
            training_args.eval_strategy = IntervalStrategy.STEPS
        if hasattr(training_args, "eval_strategy") and str(training_args.eval_strategy).lower() not in {"steps", "epoch"}:
            training_args.eval_strategy = IntervalStrategy.STEPS
    except Exception:
        if hasattr(training_args, "evaluation_strategy"):
            setattr(training_args, "evaluation_strategy", "steps")
        if hasattr(training_args, "eval_strategy"):
            setattr(training_args, "eval_strategy", "steps")

    if not hasattr(training_args, "eval_steps") or not training_args.eval_steps:
        setattr(training_args, "eval_steps", 200)

    # Align saving with evaluation if needed
    if hasattr(training_args, "save_strategy") and str(getattr(training_args, "save_strategy")).lower() in {"no", "none"}:
        try:
            from transformers.trainer_utils import IntervalStrategy
            training_args.save_strategy = IntervalStrategy.STEPS
        except Exception:
            setattr(training_args, "save_strategy", "steps")
    if not hasattr(training_args, "save_steps") or not training_args.save_steps:
        setattr(training_args, "save_steps", training_args.eval_steps)

    # Ensure best model restoration (if supported)
    if hasattr(training_args, "load_best_model_at_end") and not training_args.load_best_model_at_end:
        training_args.load_best_model_at_end = True

    # Keep logs happening
    if not hasattr(training_args, "logging_steps") or not training_args.logging_steps:
        setattr(training_args, "logging_steps", max(50, training_args.eval_steps // 4))
    # ===== end shim =====

    trainer = WeightedOrFocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(),
        use_focal=args.use_focal_loss,
        class_weight=cw_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0005)],
    )

    trainer.train()

    # Save artifacts
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(Path(args.output_dir) / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": {0: "ham", 1: "spam"}, "label2id": {"ham": 0, "spam": 1}}, f, indent=2)

    # Quick final eval
    metrics = trainer.evaluate()
    with open(Path(args.output_dir) / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("== Final metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
