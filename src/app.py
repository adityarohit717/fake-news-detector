import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("fakenews")


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-final"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# Hugging Face model repository
HF_MODEL_ID = "adityakumarrohit/fake-news-detector-distilbert"


# ============================================================
# Configuration
# ============================================================

THRESHOLD = 0.65
MAX_LENGTH = 128
MAX_CHARS = 5000
NUM_THREADS = 4


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(NUM_THREADS)


# ============================================================
# Select model source
# ============================================================

if LOCAL_MODEL_DIR.exists():
    MODEL_SOURCE = LOCAL_MODEL_DIR
    log.info(
        "Local model found. Loading from: %s",
        LOCAL_MODEL_DIR,
    )
else:
    MODEL_SOURCE = HF_MODEL_ID
    log.info(
        "Local model not found. Loading from Hugging Face: %s",
        HF_MODEL_ID,
    )


# ============================================================
# Load tokenizer and model
# ============================================================

tokenizer = None
model = None

try:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SOURCE
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_SOURCE
    )

    model.to(device)
    model.eval()

    log.info(
        "Model loaded successfully. Device: %s",
        device,
    )

except Exception:
    log.exception(
        "Failed to load model from %s",
        MODEL_SOURCE,
    )


# ============================================================
# Model inference
# ============================================================

def run_model(text: str):

    if tokenizer is None or model is None:
        raise RuntimeError(
            "Model is not available."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        logits = model(
            **inputs
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    return probabilities


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Fake News Detection API",
    version="1.2.0",
    description=(
        "NLP-based fake news classification API "
        "using DistilBERT."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request schema
# ============================================================

class NewsRequest(BaseModel):
    text: str


# ============================================================
# Frontend
# ============================================================

@app.get(
    "/",
    include_in_schema=False,
)
def index():

    frontend_file = FRONTEND_DIR / "index.html"

    if not frontend_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend not found",
        )

    return FileResponse(
        frontend_file
    )


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
        "model_source": (
            "local"
            if LOCAL_MODEL_DIR.exists()
            else "huggingface"
        ),
        "model_id": HF_MODEL_ID,
        "threshold": THRESHOLD,
        "max_tokens": MAX_LENGTH,
        "max_chars": MAX_CHARS,
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(request: NewsRequest):

    text = request.text.strip()

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not text:

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty",
        )

    if len(text) > MAX_CHARS:

        raise HTTPException(
            status_code=413,
            detail=(
                f"Text too long "
                f"(max {MAX_CHARS} characters)"
            ),
        )

    # --------------------------------------------------------
    # Model availability
    # --------------------------------------------------------

    if tokenizer is None or model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start = time.perf_counter()

    try:

        probabilities = run_model(text)

    except Exception:

        log.exception(
            "Inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    fake_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if fake_probability >= THRESHOLD:

        prediction = "FAKE"
        confidence = fake_probability

    elif real_probability >= THRESHOLD:

        prediction = "REAL"
        confidence = real_probability

    else:

        prediction = "UNCERTAIN"
        confidence = max(
            fake_probability,
            real_probability,
        )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = (
        time.perf_counter() - start
    ) * 1000

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log.info(
        "pred=%s conf=%.1f chars=%d latency=%.1fms",
        prediction,
        confidence * 100,
        len(text),
        latency,
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": round(
            confidence * 100,
            2,
        ),
        "fake_probability": round(
            fake_probability * 100,
            2,
        ),
        "real_probability": round(
            real_probability * 100,
            2,
        ),
        "latency_ms": round(
            latency,
            1,
        ),
    }