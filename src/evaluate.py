import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilbert-final"
PROC = ROOT / "data" / "processed"
THRESHOLD = 0.65
N_PER_SOURCE = 3000

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


def fake_prob(texts, bs=64):
    """P(fake) for each text. Sorted by length so batches need less padding."""
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt",
                  padding=True, truncation=True, max_length=128)
        with torch.inference_mode():
            out[idx] = torch.softmax(model(**enc).logits, dim=-1)[:, 0].numpy()
    return out


def report(name, df):
    texts = df["text"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()
    t0 = time.time()
    pf = fake_prob(texts)
    pred = np.where(pf >= 0.5, 0, 1)                      # 0 = FAKE, 1 = REAL
    decided = (pf >= THRESHOLD) | ((1 - pf) >= THRESHOLD)
    majority = max((y == 0).mean(), (y == 1).mean())
    print(f"\n=== {name} (n={len(df)}, {time.time() - t0:.0f}s) ===")
    print(f"Accuracy {accuracy_score(y, pred):.4f} | macro F1 {f1_score(y, pred, average='macro'):.4f}"
          f" | always-majority baseline {majority:.4f}")
    if decided.sum():
        print(f"UNCERTAIN rate at {THRESHOLD}: {1 - decided.mean():.1%}"
              f" | accuracy on the decided rest: {accuracy_score(y[decided], pred[decided]):.4f}")
    print("Confusion [[FAKE->FAKE, FAKE->REAL], [REAL->FAKE, REAL->REAL]]:",
          confusion_matrix(y, pred, labels=[0, 1]).tolist())


train_texts = set(pd.read_csv(PROC / "transformer_train.csv")["text"].astype(str))


def unseen_only(df):
    seen = df["text"].astype(str).isin(train_texts)
    print(f"  rows also in transformer_train.csv: {seen.sum()} of {len(df)} ({seen.mean():.1%}) -> dropped")
    return df[~seen]


# 1. The held-out set built for this model
tt = pd.read_csv(PROC / "transformer_test.csv").dropna(subset=["text", "label"])
report("transformer_test.csv", tt)

# 2. Big test split, per source, rows the model never saw
full = pd.read_csv(PROC / "test.csv", usecols=["text", "label", "source_dataset"]).dropna()
print("\ntest.csv rows per source:\n", full["source_dataset"].value_counts().to_string())
for src, g in full.groupby("source_dataset"):
    print(f"\n[{src}]")
    g = unseen_only(g)
    if len(g) > N_PER_SOURCE:
        g = g.sample(N_PER_SOURCE, random_state=1)
    if len(g):
        report(f"test.csv / {src}", g)

# 3. LIAR as an out-of-domain check
liar = pd.read_csv(PROC / "liar_standardized.csv", usecols=["text", "label"]).dropna()
print("\n[liar_standardized.csv]")
liar = unseen_only(liar)
if len(liar) > N_PER_SOURCE:
    liar = liar.sample(N_PER_SOURCE, random_state=1)
if len(liar):
    report("liar_standardized.csv", liar)