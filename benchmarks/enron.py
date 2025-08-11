#!/usr/bin/env python3
"""
Parallel Enron benchmark with pip-style progress (Rich), no duplicated lines.

CSV columns expected:
  [Index], Subject, Message, Spam/Ham, Date
"""

from __future__ import annotations
import argparse, os, sys, time
from math import ceil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from huggingface_hub import snapshot_download
from spam_bert import classify_text, resolve_model_source

# Pretty output + progress (like pip)
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.progress import (
        Progress, TextColumn, BarColumn, MofNCompleteColumn,
        TimeRemainingColumn, TaskProgressColumn, SpinnerColumn
    )
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

# ---------------- per-process state ----------------
_WORKER_MODEL_SOURCE: str | None = None

def _worker_init(resolved_model_source: str):
    global _WORKER_MODEL_SOURCE
    _WORKER_MODEL_SOURCE = resolved_model_source
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

def _classify_chunk(chunk_rows, threshold: float, worker_id: int, q=None):
    """
    Returns list of (idx, y_true, y_pred, prob, chunks, text_len).
    If q is provided (a multiprocessing.Manager().Queue()), send q.put((worker_id, 1)) per row.
    """
    assert _WORKER_MODEL_SOURCE is not None, "worker not initialized"
    out_rows = []
    for idx, row in chunk_rows:
        subj = str(row.get("Subject", "") or "")
        msg  = str(row.get("Message", "") or "")
        text = (subj + " " + msg).strip()
        y_true = 1 if str(row.get("Spam/Ham", "")).strip().lower() == "spam" else 0
        out = classify_text(
            text,
            model_name=_WORKER_MODEL_SOURCE,
            threshold=threshold,
            local_model_dir=None,
            model_cache_dir=None,
        )
        y_pred = 1 if out["decision"] == "spam" else 0
        prob   = float(out["spam_probability"])
        out_rows.append((idx, y_true, y_pred, prob, out.get("chunks", 1), len(text)))
        if q is not None:
            q.put((worker_id, 1))  # tell main to advance this worker + overall
    # signal worker finished (optional)
    if q is not None:
        q.put((worker_id, 0))
    return out_rows

def _make_progress():
    return Progress(
        TextColumn("[bold]Benchmark[/bold]"),
        SpinnerColumn(),
        TextColumn("{task.fields[label]}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=True,  # hide progress after completion
        refresh_per_second=8,
    )

def print_criteria(metrics: dict, thresholds: dict) -> bool:
    rows = [
        ("Macro F1",      metrics["macro_f1"],      thresholds["macro_f1"]),
        ("Spam Recall",   metrics["spam_recall"],   thresholds["spam_recall"]),
        ("Ham Precision", metrics["ham_precision"], thresholds["ham_precision"]),
        ("ROC-AUC",       metrics["roc_auc"],       thresholds["roc_auc"]),
    ]
    overall_ok = True
    if RICH:
        table = Table(title="Benchmark Criteria", box=box.SIMPLE_HEAVY)
        table.add_column("Metric"); table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right"); table.add_column("Result", justify="center")
        for name, val, thr in rows:
            ok = (val >= thr)
            overall_ok &= ok
            table.add_row(name, f"{val:.4f}", f"{thr:.4f}", "✅" if ok else "❌")
        console.print(table)
    else:
        print("\n== Benchmark Criteria ==")
        for name, val, thr in rows:
            ok = (val >= thr); overall_ok &= ok
            print(f"{name:>14}: {val:.4f} | thr {thr:.4f} {'✅' if ok else '❌'}")
    return overall_ok

def main():
    ap = argparse.ArgumentParser(description="Parallel Enron benchmark (pip-style progress)")
    ap.add_argument("--csv", required=True, help="Path to Enron CSV (Subject, Message, Spam/Ham, Date)")
    ap.add_argument("--model", default="AntiSpamInstitute/spam-detector-bert-MoE-v2.2")
    ap.add_argument("--local-model-dir", default=None)
    ap.add_argument("--model-cache-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-csv", default="enron_predictions.csv")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress UI")

    # Criteria (tweak to taste)
    ap.add_argument("--crit-macro-f1", type=float, default=0.94)
    ap.add_argument("--crit-spam-recall", type=float, default=0.95)
    ap.add_argument("--crit-ham-precision", type=float, default=0.98)
    ap.add_argument("--crit-roc-auc", type=float, default=0.98)
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any criterion fails")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.limit > 0:
        df = df.iloc[:args.limit].copy()

    # Resolve model once (and optionally pre-download)
    local_dir = Path(args.local_model_dir) if args.local_model_dir else None
    cache_dir = Path(args.model_cache_dir) if args.model_cache_dir else None
    resolved = resolve_model_source(args.model, local_dir, cache_dir)

    if cache_dir and os.path.isdir(resolved) is False and "/" in resolved:
        target = cache_dir / resolved.replace("/", "__")
        snapshot_download(repo_id=resolved, local_dir=str(target), local_dir_use_symlinks=False)
        resolved = str(target)

    rows_iter = list(df[["Subject", "Message", "Spam/Ham"]].iterrows())
    if not rows_iter:
        print("No rows to process."); sys.exit(2)

    W = max(1, int(args.workers))
    chunk_sz = ceil(len(rows_iter) / W)
    chunks = [rows_iter[i:i+chunk_sz] for i in range(0, len(rows_iter), chunk_sz)]
    # In case of more workers than rows
    chunks += [[]] * max(0, W - len(chunks))

    results = { "idx": [], "y_true": [], "y_pred": [], "spam_prob": [], "chunks": [], "text_len": [] }

    use_progress = RICH and (not args.no_progress) and console.is_terminal

    with Manager() as mgr:
        q = mgr.Queue(maxsize=1000) if use_progress else None

        with ProcessPoolExecutor(max_workers=W, initializer=_worker_init, initargs=(resolved,)) as ex:
            futures = {
                ex.submit(_classify_chunk, chunks[wid], args.threshold, wid, q): wid
                for wid in range(W)
            }

            if use_progress:
                with _make_progress() as progress:
                    # Build tasks: one per worker + overall
                    overall = progress.add_task("", label="Overall", total=len(rows_iter))
                    worker_tasks = []
                    for wid in range(W):
                        total = len(chunks[wid])
                        worker_tasks.append(
                            progress.add_task("", label=f"Worker {wid}", total=total if total > 0 else 1)
                        )

                    done_workers = 0
                    done_rows = 0
                    # Main UI loop: advance bars on queue messages while gathering results
                    while done_workers < W:
                        # drain progress updates
                        drained = False
                        while q is not None:
                            try:
                                wid, inc = q.get_nowait()
                                drained = True
                                if inc > 0:
                                    progress.advance(worker_tasks[wid], inc)
                                    progress.advance(overall, inc)
                                    done_rows += inc
                                else:
                                    # worker sent 'done' signal
                                    pass
                            except Exception:
                                break
                        # collect any completed futures non-blocking
                        for fut in list(f for f in futures if futures[f] is not None and f.done()):
                            wid = futures[fut]
                            futures[fut] = None  # mark consumed
                            try:
                                for idx, y_true, y_pred, prob, ch, tlen in fut.result():
                                    results["idx"].append(idx)
                                    results["y_true"].append(y_true)
                                    results["y_pred"].append(y_pred)
                                    results["spam_prob"].append(prob)
                                    results["chunks"].append(ch)
                                    results["text_len"].append(tlen)
                            finally:
                                done_workers += 1
                        # small sleep to reduce CPU when idle
                        time.sleep(0.05 if not drained else 0.0)
            else:
                # no progress UI; just collect
                for fut in as_completed(futures):
                    for idx, y_true, y_pred, prob, ch, tlen in fut.result():
                        results["idx"].append(idx)
                        results["y_true"].append(y_true)
                        results["y_pred"].append(y_pred)
                        results["spam_prob"].append(prob)
                        results["chunks"].append(ch)
                        results["text_len"].append(tlen)

    # Reassemble & save
    out_df = pd.DataFrame(results).sort_values("idx")
    out = df.copy()
    out["y_true"] = out_df["y_true"].values
    out["y_pred"] = out_df["y_pred"].values
    out["spam_prob"] = out_df["spam_prob"].values
    out["chunks"] = out_df["chunks"].values
    out["text_len"] = out_df["text_len"].values
    out.to_csv(args.out_csv, index=False)

    # Metrics
    y_true = out["y_true"].tolist()
    y_pred = out["y_pred"].tolist()
    probs  = out["spam_prob"].tolist()

    rep = classification_report(y_true, y_pred, target_names=["ham","spam"], digits=4, output_dict=True)
    macro_f1 = rep["macro avg"]["f1-score"]
    spam_recall = rep["spam"]["recall"]
    ham_precision = rep["ham"]["precision"]
    try:
        roc = roc_auc_score(y_true, probs)
    except Exception:
        roc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    # Pretty report
    if RICH:
        table = Table(title="Classification Report", box=box.SIMPLE)
        table.add_column("Class"); table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right"); table.add_column("F1", justify="right")
        for cls in ("ham","spam"):
            table.add_row(cls, f"{rep[cls]['precision']:.4f}", f"{rep[cls]['recall']:.4f}", f"{rep[cls]['f1-score']:.4f}")
        table.add_row("macro avg", f"{rep['macro avg']['precision']:.4f}", f"{rep['macro avg']['recall']:.4f}", f"{rep['macro avg']['f1-score']:.4f}")
        console.print(table)
        console.print(f"ROC-AUC: [bold]{roc:.4f}[/bold]")
        console.print(f"Confusion matrix:\n{cm}")
    else:
        print("\n=== Classification report ===")
        print(f"ham   P={rep['ham']['precision']:.4f} R={rep['ham']['recall']:.4f} F1={rep['ham']['f1-score']:.4f}")
        print(f"spam  P={rep['spam']['precision']:.4f} R={rep['spam']['recall']:.4f} F1={rep['spam']['f1-score']:.4f}")
        print(f"macro F1 = {macro_f1:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print("Confusion matrix:\n", cm)

    # Criteria
    thresholds = {
        "macro_f1": args.crit_macro_f1,
        "spam_recall": args.crit_spam_recall,
        "ham_precision": args.crit_ham_precision,
        "roc_auc": args.crit_roc_auc,
    }
    metrics = {
        "macro_f1": macro_f1,
        "spam_recall": spam_recall,
        "ham_precision": ham_precision,
        "roc_auc": roc,
    }
    ok = print_criteria(metrics, thresholds)
    if args.strict and not ok:
        if RICH: console.print("[bold red]❌ Benchmark did not meet acceptance criteria (--strict).[/bold red]")
        else:    print("❌ Benchmark did not meet acceptance criteria (--strict).")
        sys.exit(1)
    else:
        if RICH: console.print("[bold green]✅ Benchmark criteria satisfied.[/bold green]")
        else:    print("✅ Benchmark criteria satisfied.")

if __name__ == "__main__":
    main()
