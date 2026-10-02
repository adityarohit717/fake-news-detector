# Fake News Detector

An NLP-based fake news detection system built using **DistilBERT**, **PyTorch**, and **FastAPI**.

The system accepts a news headline or article text and predicts whether the text resembles **FAKE**, **REAL**, or **UNCERTAIN** based on the model's confidence.

> **Important:** This is a text classifier, not a fact-checking system. It does not independently verify whether a claim is true or false.

---

## Features

- DistilBERT-based text classification
- FAKE / REAL / UNCERTAIN predictions
- Confidence score
- Fake and real probability scores
- Configurable confidence threshold
- CPU inference support
- Dynamic INT8 CPU benchmarking
- FastAPI REST API
- Browser-based frontend
- Input validation
- Inference latency reporting
- Automated API tests with pytest
- Evaluation on held-out and source-specific datasets

---

## System Architecture

```text
User
 │
 ▼
Frontend (HTML/CSS/JavaScript)
 │
 ▼
FastAPI
 │
 ▼
Input Validation
 │
 ▼
DistilBERT Tokenizer
 │
 ▼
DistilBERT Classification Model
 │
 ▼
Softmax Probabilities
 │
 ├── Fake Probability
 └── Real Probability
 │
 ▼
Confidence Threshold
 │
 ├── FAKE
 ├── REAL
 └── UNCERTAIN
 │
 ▼
JSON Response / Frontend Result