# Spam BERT Detector
![Logo](spambert_logo.png)

A flexible spam/ham classifier powered by Hugging Face Transformers.  
Supports both **command-line** and **REST API** usage, with:

- **Model flexibility** – use any `text-classification` model from Hugging Face or a local fine-tuned model.
- **Configurable caching** – choose where models are downloaded and stored (`--model-cache-dir`), or run fully offline (`--local-model-dir`).
- **Long-email support** – intelligently splits long emails into overlapping 512-token chunks and aggregates results.
- **.eml parsing** – extracts plain text or HTML body for classification.
- **One-file deploy** – package into a single executable with PyInstaller.

Perfect for integrating spam detection into email gateways, webhooks, or automation pipelines.

## Features

- **CLI mode**
  ```bash
  python spam_bert.py email.eml --pretty
  python spam_bert.py "You won a FREE iPhone!" --threshold 0.7

- REST API mode
```bash
python spam_bert.py --serve --port 9000 --model AntiSpamInstitute/spam-detector-bert-MoE-v2.2
```

- Endpoints 
  - GET /health – Health check 
  - GET /model – Current model info 
  - POST /classify – Classify raw text or base64-encoded .eml

## Example REST call
```bash
curl -X POST http://localhost:9000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "FREE gift card – click here!"}'
```
**Response**:
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

## License
[MIT License](LICENSE.md)