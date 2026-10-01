import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "distilbert-final"
TEXT = "The government announced a new policy affecting millions of citizens."

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(device, half=False, runs=50):
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
    if half:
        model = model.half()
    sync(device)
    load = time.perf_counter() - t0

    inputs = tok(TEXT, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    def one():
        s = time.perf_counter()
        with torch.inference_mode():
            model(**inputs)
        sync(device)
        return (time.perf_counter() - s) * 1000

    first = one()
    for _ in range(5):
        one()  # warm-up
    times = [one() for _ in range(runs)]
    avg = sum(times) / len(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    name = f"{device.type}{' fp16' if half else ''}"
    print(f"{name:10} load {load:5.1f}s | first {first:7.1f} ms | avg {avg:6.1f} ms | p95 {p95:6.1f} ms")


bench(torch.device("cpu"))
if torch.cuda.is_available():
    bench(torch.device("cuda"))
    bench(torch.device("cuda"), half=True)