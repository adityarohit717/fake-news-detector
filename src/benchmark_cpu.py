import copy, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilbert-final"
TEXT = "The government announced a new policy affecting millions of citizens."

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
int8 = torch.ao.quantization.quantize_dynamic(
    copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8
)


def bench(model, threads, runs=50):
    torch.set_num_threads(threads)
    inputs = tok(TEXT, return_tensors="pt", truncation=True, max_length=128)
    def one():
        s = time.perf_counter()
        with torch.inference_mode():
            model(**inputs)
        return (time.perf_counter() - s) * 1000
    for _ in range(6):
        one()  # warm-up
    t = sorted(one() for _ in range(runs))
    return sum(t) / len(t), t[int(0.95 * runs) - 1]


default_threads = torch.get_num_threads()
print("Default threads:", default_threads)
print(f"{'model':6} {'threads':>7} {'avg ms':>8} {'p95 ms':>8}")
for th in sorted({2, 4, default_threads}):
    for name, m in (("fp32", fp32), ("int8", int8)):
        avg, p95 = bench(m, th)
        print(f"{name:6} {th:7d} {avg:8.1f} {p95:8.1f}")

# ---- correctness check: does int8 change the predictions? ----
torch.set_num_threads(default_threads)
texts = [
    "Scientists discovered a new method for producing clean energy after conducting several controlled experiments.",
    "A viral social media post claims that drinking a certain liquid can instantly cure every disease.",
    "The government announced a new policy affecting millions of citizens.",
    "Breaking: celebrity secretly arrested, media refuses to report it.",
    "The central bank kept interest rates unchanged on Wednesday, citing stable inflation.",
    "You won't believe what this politician did, doctors hate this trick.",
]
try:
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "test.csv").sample(300, random_state=1)
    col = "text" if "text" in df.columns else df.columns[0]
    texts += df[col].astype(str).tolist()
    print(f"\nChecking on {len(texts)} texts (incl. 300 from data/test.csv, column '{col}')")
except Exception as e:
    print(f"\nChecking on {len(texts)} texts (test.csv not used: {e})")


def fake_probs(model):
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            out.append(torch.softmax(model(**enc).logits, dim=-1)[:, 0])
    return torch.cat(out)


def label(p, th=0.65):
    return ["FAKE" if x >= th else "REAL" if 1 - x >= th else "UNCERTAIN" for x in p.tolist()]


a, b = fake_probs(fp32), fake_probs(int8)
agree = sum(x == y for x, y in zip(label(a), label(b))) / len(texts)
print("Max prob difference :", round((a - b).abs().max().item(), 4))
print("Mean prob difference:", round((a - b).abs().mean().item(), 4))
print("Label agreement     :", f"{agree:.1%}")