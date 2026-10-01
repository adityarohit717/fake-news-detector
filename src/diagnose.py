from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "distilbert-final"
torch.set_num_threads(4)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
print("id2label:", model.config.id2label)

df = pd.read_csv(ROOT / "data" / "processed" / "transformer_test.csv").dropna()
df = df.sample(2000, random_state=1)
y = df["label"].astype(int).to_numpy()
print("Label counts:", df["label"].value_counts().to_dict())

pd.set_option("display.max_colwidth", 100)
for lab in (0, 1):
    print(f"\n--- 6 samples with label={lab} ---")
    print(df.loc[y == lab, "text"].head(6).to_string(index=False))

texts = df["text"].astype(str).tolist()
pf = np.zeros(len(texts))
order = np.argsort([len(t) for t in texts])
for i in range(0, len(texts), 64):
    idx = order[i:i + 64]
    enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
              truncation=True, max_length=128)
    with torch.inference_mode():
        pf[idx] = torch.softmax(model(**enc).logits, dim=-1)[:, 0].numpy()

print("\nMean P(fake) when label=0:", round(pf[y == 0].mean(), 3),
      "| when label=1:", round(pf[y == 1].mean(), 3))
print("ROC-AUC (P(fake) vs label==0):", round(roc_auc_score(y == 0, pf), 4))