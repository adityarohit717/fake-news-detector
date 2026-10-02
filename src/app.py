import logging
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

MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-final"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


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
# Model
# ============================================================

tokenizer = None
model = None


if MODEL_DIR.exists():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        model.to(device)
        model.eval()

        log.info(
            "Model loaded successfully from %s",
            MODEL_DIR,
        )

    except Exception:
        log.exception("Failed to load model")

else:
    log.warning(
        "Model directory not found: %s",
        MODEL_DIR,
    )


# ============================================================
# Model inference
# ============================================================

def run_model(text: str):
    """
    Run fake-news classification.

    The real DistilBERT model is used when the model directory
    exists. Tests can replace this function with a mock model.
    """

    if tokenizer is None or model is None:
        raise RuntimeError(
            "Model is not available. "
            "Please make sure models/distilbert-final exists."
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
        logits = model(**inputs).logits

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
    version="1.1.0",
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
    """
    Serve the browser frontend.
    """

    frontend_file = FRONTEND_DIR / "index.html"

    if not frontend_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found",
        )

    return FileResponse(frontend_file)


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():
    """
    Return API and model configuration information.
    """

    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
        "max_tokens": MAX_LENGTH,
        "max_chars": MAX_CHARS,
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(request: NewsRequest):
    """
    Classify submitted news text as:

    FAKE
    REAL
    UNCERTAIN
    """

    text = request.text.strip()

    # --------------------------------------------------------
    # Validate input
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
    # Run inference
    # --------------------------------------------------------

    start = time.perf_counter()

    try:
        probabilities = run_model(text)

    except RuntimeError as exc:
        log.error("Model unavailable: %s", exc)

        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    except Exception:
        log.exception("Inference failed")

        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        )

    # --------------------------------------------------------
    # Extract probabilities
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