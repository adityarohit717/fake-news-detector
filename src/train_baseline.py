import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/validation.csv"
MODEL_PATH = "models/tfidf_logistic.pkl"


print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

X_train = train_df["text"].fillna("")
y_train = train_df["label"].astype(int)

X_val = val_df["text"].fillna("")
y_val = val_df["label"].astype(int)

print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")


print("\nBuilding TF-IDF + Logistic Regression pipeline...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=300_000,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
    ),
])


print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


print("\nEvaluating validation set...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)

print("\n========== RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        target_names=["FAKE", "REAL"],
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))


print("\nSaving model...")

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")