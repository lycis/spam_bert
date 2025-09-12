#!/usr/bin/env python3
"""
Finetune prancyFox/tiny-bert-enron-spam for spam detection using a custom CSV with subject, body, and spam/ham columns.
"""

import argparse
import pandas as pd
from train_tinybert import main as train_main
from sklearn.model_selection import train_test_split

def preprocess_data(input_csv, output_csv):
    """Preprocess your CSV: combine subject/body, map spam/ham to 1/0."""
    df = pd.read_csv(input_csv)
    df["text"] = df["subject"] + " " + df["body"]
    df["label"] = df["spam/ham"].map({"spam": 1, "ham": 0})
    df[["text", "label"]].to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True, help="Path to your input CSV (subject, body, spam/ham).")
    ap.add_argument("--output_csv", type=str, default="processed_data.csv", help="Path to save processed CSV.")
    ap.add_argument("--train", type=str, default=None, help="Path to processed train CSV (auto-split if omitted).")
    ap.add_argument("--val", type=str, default=None, help="Path to processed val CSV.")
    ap.add_argument("--model_name", type=str, default="prancyFox/tiny-bert-enron-spam", help="Model to finetune.")
    ap.add_argument("--output_dir", type=str, default="outputs/finetuned-model", help="Output directory.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hub_repo", type=str, default=None, help="Hub repo for pushing.")
    args = ap.parse_args()

    # Preprocess data
    preprocess_data(args.input_csv, args.output_csv)

    # Split into train/val if needed
    if not args.train:
        df = pd.read_csv(args.output_csv)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv("train_processed.csv", index=False)
        val_df.to_csv("val_processed.csv", index=False)
        args.train = "train_processed.csv"
        args.val = "val_processed.csv"

    # Call the training script
    train_args = [
        "--train", args.train,
        "--val", args.val,
        "--model_name", args.model_name,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--max_length", str(args.max_length),
    ]
    if args.push_to_hub:
        train_args.extend(["--push_to_hub", "--hub_repo", args.hub_repo])

    # Convert to sys.argv and call main
    import sys
    sys.argv = ["train_tinybert.py"] + train_args
    train_main()

if __name__ == "__main__":
    main()
