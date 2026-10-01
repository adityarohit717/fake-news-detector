from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilbert-final"
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

cols = ["id", "label", "statement", "subject", "speaker", "job", "state",
        "party", "c1", "c2", "c3", "c4", "c5", "context"]
df = pd.read_csv(ROOT / "data" / "raw" / "liar" / "test.tsv", sep="\t", header=None, names=cols)
FAKE = {"pants-fire", "false", "barely-true"}
y = df["label"].apply(lambda l: 0 if l in FAKE else 1).to_numpy()
texts = df["statement"].astype(str).tolist()

pf = np.zeros(len(texts))
order = np.argsort([len(t) for t in texts])
for i in range(0, len(texts), 64):
    idx = order[i:i + 64]
    enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
              truncation=True, max_length=128)
    with torch.inference_mode():
        pf[idx] = torch.softmax(model(**enc).logits, dim=-1)[:, 0].numpy()

pred = np.where(pf >= 0.5, 0, 1)
print("LIAR test rows:", len(df))
print("Accuracy at 0.5:", round(accuracy_score(y, pred), 4))
print("ROC-AUC:", round(roc_auc_score(y == 0, pf), 4))
print("Predicted FAKE:", int((pred == 0).sum()), "of", len(df))