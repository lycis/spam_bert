# 📧 Spam BERT Detector

![Logo](spambert_logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Build Status](https://github.com/lycis/spam_bert/actions/workflows/ci.yml/badge.svg)](https://github.com/lycis/spam_bert/actions)
[![codecov](https://codecov.io/gh/lycis/spam_bert/branch/main/graph/badge.svg)](https://codecov.io/gh/lycis/spam_bert)
![Powered by BERT](https://img.shields.io/badge/powered%20by-BERT-orange)
![Container](https://img.shields.io/badge/container-GHCR-blue)

---

**Spam BERT Detector** is a flexible spam/ham classifier powered by [Hugging Face Transformers](https://huggingface.co/).  
It can run from the **command line**, as a **REST API**, or inside **Docker**, making it ideal for email gateways, webhook filters, or automated pipelines.

---

## 📚 Table of Contents
- [✨ Key Features](#-key-features)
- [🚀 CLI Usage](#-cli-usage)
- [📡 REST API Mode](#-rest-api-mode)
  - [Example REST Request](#example-rest-request)
- [🔄 Aggregation Strategies for Chunked Inference](#-aggregation-strategies-for-chunked-inference)
- [🛠️ Installation](#️-installation)
- [🐾 Default Model](#-default-model)
- [📦 Packaging](#-packaging)
- [📜 License](#-license)
- [🛣 Roadmap](#-roadmap)

---

## ✨ Key Features

* **Model flexibility** – Use any Hugging Face `text-classification` model or a local fine-tuned model.
* **Configurable caching** – Choose where models are stored (`--model-cache-dir`) or run fully offline (`--local-model-dir`).
* **Long-email support** – Splits long emails into overlapping 512-token chunks and aggregates results.
* **Multiple aggregation strategies** – Choose how to combine chunk predictions (`--aggregation`).
* **EML parsing** – Extracts plain text or HTML body for classification.
* **One-file deploy** – Package into a single executable with PyInstaller.
* **Docker-ready** – Prebuilt images for CI, nightly builds, and tagged releases.
* **CI/CD Integration** – Automated builds, tests, and code coverage reporting.

---

## 🚀 CLI Usage

```bash
python spam_bert.py <input> [options]
````

### **Positional Arguments**

| Argument | Description                                                      |
| -------- | ---------------------------------------------------------------- |
| `input`  | Path to a `.eml` or `.txt` file, or raw text string to classify. |

### **Options**

| Option                   | Description                                                                                                                                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model MODEL`          | Hugging Face model ID or local directory path. Default: `mshenoda/roberta-spam`.                                                                                                                                                                                                                         |
| `--local-model-dir PATH` | Path to a local model directory for offline use.                                                                                                                                                                                                                                                         |
| `--model-cache-dir PATH` | Directory to store/download models from Hugging Face (avoids default hidden cache).                                                                                                                                                                                                                      |
| `--threshold FLOAT`      | Spam probability threshold. Default: `0.6`.                                                                                                                                                                                                                                                              |
| `--no-chunk`             | Disable chunking for long texts (truncate to model max length instead).                                                                                                                                                                                                                                  |
| `--aggregation METHOD`   | How to combine spam probabilities from multiple chunks|
| `--pretty`               | Pretty-print JSON output.                                                                                                                                                                                                                                                                                |
| `--serve`                | Run as a REST API service instead of CLI mode.                                                                                                                                                                                                                                                           |
| `--host HOST`            | REST API host (default: `0.0.0.0`).                                                                                                                                                                                                                                                                      |
| `--port PORT`            | REST API port (default: `8000`).                                                                                                                                                                                                                                                                         |

---

## 📡 REST API Mode

```bash
python spam_bert.py --serve --port 9000 \
  --model AntiSpamInstitute/spam-detector-bert-MoE-v2.2
```

**Endpoints**:

* `GET /health` – Health check
* `GET /model` – Current model info
* `POST /classify` – Classify raw text or base64-encoded `.eml` file

---

### Example REST Request

```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE gift card – click here!", "aggregation": "max"}'
```

**Response:**

```json
{
  "decision": "spam",
  "spam_probability": 0.9871,
  "threshold": 0.6,
  "chunks": 1,
  "aggregation": "max",
  "model_source": "AntiSpamInstitute/spam-detector-bert-MoE-v2.2",
  "chars_analyzed": 43
}
```

---
## 🔄 Aggregation Strategies for Chunked Inference

When an email exceeds the model’s **512-token limit**, it is split into overlapping chunks.
Each chunk is classified separately, producing a spam probability.
The **aggregation strategy** determines how these per-chunk scores are combined into a single decision.

Use via CLI with `--aggregate <method>` or API with `"aggregate": "<method>"`.

**Available methods:**

| Method            | Description                                                                                 | Parameters                            |
| ----------------- | ------------------------------------------------------------------------------------------- | ------------------------------------- |
| `mean`            | Average spam probability across chunks.                                                     | —                                     |
| `max`             | Takes the highest spam probability among chunks.                                            | —                                     |
| `median`          | Middle value when chunk scores are sorted.                                                  | —                                     |
| `topk_mean`       | Average of the top-K highest scores.                                                        | `--topk` (default `3`)                |
| `quantile`        | Selects the score at a given quantile.                                                      | `--quantile` (0.0–1.0, default `0.9`) |
| `length_weighted` | Weighted average, weighting each chunk by its token length.                                 | —                                     |
| `position_decay`  | Earlier chunks weigh more, later chunks decay by a factor.                                  | `--decay` (0.0–1.0, default `0.9`)    |
| `logit_mean`      | Mean of logits (log-odds) instead of probabilities; reduces probability saturation effects. | —                                     |
| `noisy_or`        | Probability that **any** chunk is spam, assuming independence.                              | —                                     |
| `k_of_n`          | Score based on proportion of chunks exceeding a threshold, normalized by K.                 | `--topk` and `--per-chunk-thr`        |
| `majority_vote`   | Fraction of chunks voting “spam” (above threshold).                                         | `--per-chunk-thr`                     |

**Example:**

```bash
python -m spam_bert email.eml \
  --aggregate topk_mean \
  --topk 5 \
  --per-chunk-thr 0.7
```

This reads the email, splits it into chunks, takes the top 5 spam scores, averages them,
and decides “spam” if the final score exceeds the global threshold.
---

## 🛠️ Installation

### From source

```bash
pip install -r requirements.txt
pip install -e .
```

### From Docker

```bash
docker run -p 8000:8000 ghcr.io/lycis/spam_bert:latest
```

You can also pull:

* `:ci` – Latest CI build
* `:nightly` – Nightly builds
* `:vX.Y.Z` – Specific release

---

Got it — here’s the section updated to make it clear that the exact script in `train/` was used to train and fine-tune the default model:

---

## 🐾 Default Model
By default, **Spam BERT Detector** ships with [`prancyFox/tiny-bert-enron-spam`](https://huggingface.co/prancyFox/tiny-bert-enron-spam),
a **TinyBERT**-based spam/ham classifier fine-tuned on the **Enron Email Spam Dataset**.

**Why this model?**

* **Lightweight** – TinyBERT architecture for fast inference (<50 ms typical per email) with low RAM use.
* **Domain-tuned** – Trained on thousands of real corporate emails (ham and spam) from the Enron corpus.
* **Balanced** – High spam recall without sacrificing ham precision.
* **Offline-ready** – Can be bundled locally with `--local-model-dir` for air-gapped deployments.

**Performance (Evaluation on held-out Enron test set):**

| Class         | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| Ham           | 0.6875     | 0.9973     | 0.8139     |
| Spam          | 0.9954     | 0.5632     | 0.7194     |
| **Macro Avg** | **0.8414** | **0.7802** | **0.7666** |

**Additional Metrics:**

* **ROC-AUC**: 0.9977
* **Confusion Matrix**:

  ```
  [[16500    45]   # ham correctly classified vs. ham misclassified
   [ 7500  9671]]  # spam misclassified as ham vs. spam correctly classified
  ```

**Default behavior:**

* If `--model` is not specified, the detector loads `prancyFox/tiny-bert-enron-spam` from Hugging Face.
* Override with **any** Hugging Face `text-classification` model or a path to a local fine-tuned model.

**Example:**

```bash
# Uses default model
python spam_bert.py email.eml

# Override with a different model
python spam_bert.py email.eml --model my-org/spam-detector-v2
```

**Training your own model:**
The exact [`train/train.py`](train/train.py) script in this repository was used to fine-tune
`prancyFox/tiny-bert-enron-spam`. You can use the same script to train a replacement model
on your own dataset. It supports:

* Preprocessing for `.eml`, `.txt`, and `.csv` spam datasets
* Fine-tuning of any `transformers` text-classification backbone
* Evaluation with precision, recall, F1, and ROC-AUC
* Optional Hugging Face model upload

---

## 📦 Packaging

To create a standalone executable:

```bash
pyinstaller --onefile spam_bert.py --name spam-bert
```

---

## 📜 License

Licensed under the [MIT License](LICENSE).

---

## 🛣 Roadmap

### 0.2.x — Benchmarking & API Hardening

- [x] Reproducible benchmark suite using Enron Spam Dataset
- [ ] Report Precision / Recall / F1 / ROC-AUC
- [ ] API key authentication & per-client rate limiting
- [ ] Request size limits, CORS config, and trusted host checks

### 0.3.x — Deployment & Distribution

* Pre-built Docker images (CI / nightly / tagged release variants)
* One-file binaries via PyInstaller
* Improved HF model caching in container environments

### 0.4.x — Advanced Features

* Benchmarking across datasets (SMS Spam, Ling-Spam)
* Support for additional backends (DistilBERT, RoBERTa)
* Configurable multi-model voting ensemble

### 0.5.x — Production Maturity

* Monitoring & metrics export (Prometheus)
* Security scanning (SAST/DAST)
* Optional web dashboard for testing
