# Fake News Detector

An NLP-based fake news classification system built with **DistilBERT**, **PyTorch**, **FastAPI**, and a lightweight web frontend.

The system analyzes news text and classifies it as:

- **FAKE**
- **REAL**
- **UNCERTAIN**

> **Important:** This is a text-style classifier, not a fact-checking system. It does not independently verify whether a claim is true.

---

## Demo

![Fake News Detector Demo](frontend-demo.png)

---

## Features

- DistilBERT-based text classification
- FAKE / REAL / UNCERTAIN prediction
- Confidence score
- Fake and real probability scores
- Inference latency measurement
- 5,000-character input limit
- 128-token model input limit
- FastAPI REST API
- Browser-based frontend
- CPU inference support
- Dynamic INT8 CPU benchmarking
- Automated pytest test suite
- GitHub Actions CI
- Model kept outside the Git repository

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │     Web Frontend    │
                    │      index.html     │
                    └──────────┬──────────┘
                               │
                               │ POST /predict
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │      src/app.py     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Tokenizer      │
                    │     DistilBERT      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Classification    │
                    │      Model          │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌──────────────────────────────────┐
             │ FAKE / REAL / UNCERTAIN          │
             │ Probability + Confidence + ms    │
             └──────────────────────────────────┘
