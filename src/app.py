import logging
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fakenews")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-final"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

THRESHOLD = 0.65
MAX_LENGTH = 128      # tokens the model reads
MAX_CHARS = 5000      # input size limit
NUM_THREADS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    torch.set_num_threads(NUM_THREADS)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def run_model(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0]


for _ in range(3):  # warm-up
    run_model("warm up")
print(f"Model loaded successfully! Device: {device}")

app = FastAPI(title="Fake News Detection API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsRequest(BaseModel):
    text: str

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "threshold": THRESHOLD,
        "max_tokens": MAX_LENGTH,
        "max_chars": MAX_CHARS,
    }


@app.post("/predict")
def predict(request: NewsRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail=f"Text too long (max {MAX_CHARS} characters)")

    start = time.perf_counter()
    try:
        probs = run_model(text)
    except Exception:
        log.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")

    fake_p, real_p = probs[0].item(), probs[1].item()
    if fake_p >= THRESHOLD:
        prediction, confidence = "FAKE", fake_p
    elif real_p >= THRESHOLD:
        prediction, confidence = "REAL", real_p
    else:
        prediction, confidence = "UNCERTAIN", max(fake_p, real_p)

    latency = (time.perf_counter() - start) * 1000
    log.info("pred=%s conf=%.1f chars=%d latency=%.1fms", prediction, confidence * 100, len(text), latency)

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(fake_p * 100, 2),
        "real_probability": round(real_p * 100, 2),
        "latency_ms": round(latency, 1),
    }