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

## ✨ Key Features

* **Model flexibility** – Use any Hugging Face `text-classification` model or a local fine-tuned model.
* **Configurable caching** – Choose where models are stored (`--model-cache-dir`) or run fully offline (`--local-model-dir`).
* **Long-email support** – Splits long emails into overlapping 512-token chunks and aggregates results.
* **EML parsing** – Extracts plain text or HTML body for classification.
* **One-file deploy** – Package into a single executable with PyInstaller.
* **Docker-ready** – Prebuilt images for CI, nightly builds, and tagged releases.
* **CI/CD Integration** – Automated builds, tests, and code coverage reporting.

---

## 🚀 Usage

### Command-Line Mode

```bash
python -m spam_bert email.eml --pretty
python -m spam_bert "You won a FREE iPhone!" --threshold 0.7
```

### REST API Mode

```bash
python -m spam_bert --serve --port 9000 \
  --model AntiSpamInstitute/spam-detector-bert-MoE-v2.2
```

**Endpoints**:

* `GET /health` – Health check
* `GET /model` – Current model info
* `POST /classify` – Classify raw text or base64-encoded `.eml` file

---

## 📡 Example REST Call

```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE gift card – click here!"}'
```

**Response:**

```json
{
  "decision": "spam",
  "spam_probability": 0.9871,
  "threshold": 0.6,
  "chunks": 1,
  "model_source": "AntiSpamInstitute/spam-detector-bert-MoE-v2.2",
  "chars_analyzed": 43
}
```

---

## 🛠️ Installation

### From source

```bash
pip install -r requirements.txt
pip install -e .
```

### From Docker

```bash
docker run -p 9000:9000 ghcr.io/lycis/spam_bert:latest
```

You can also pull:

* `:ci` for the latest CI build
* `:nightly` for nightly builds
* version tags like `:v0.1.0` for specific releases

---

## 📦 Packaging

To create a standalone executable:

```bash
pyinstaller --onefile src/spam_bert/main.py --name spam-bert
```

---

## 📜 License

Licensed under the [MIT License](LICENSE).
