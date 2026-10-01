from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FAKE NEWS DETECTOR
# PHASE 12.4.2 — LABEL DISTRIBUTION ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "liar" / "train.tsv"

COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    header=None,
    names=COLUMNS,
    quoting=3,
)


# ------------------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------------------

print("=" * 70)
print("LIAR DATASET — LABEL ANALYSIS")
print("=" * 70)

print(f"\nTotal training examples: {len(df):,}")


# ------------------------------------------------------------
# LABEL COUNTS
# ------------------------------------------------------------

label_counts = df["label"].value_counts()

print("\n" + "=" * 70)
print("LABEL COUNTS")
print("=" * 70)

print(label_counts)


# ------------------------------------------------------------
# LABEL PERCENTAGES
# ------------------------------------------------------------

label_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n" + "=" * 70)
print("LABEL PERCENTAGES")
print("=" * 70)

for label, percentage in label_percentages.items():
    count = label_counts[label]

    print(
        f"{label:15} "
        f"{count:5} samples "
        f"({percentage:6.2f}%)"
    )


# ------------------------------------------------------------
# CHECK FOR MISSING LABELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING LABELS")
print("=" * 70)

print(
    f"Missing labels: {df['label'].isna().sum()}"
)


# ------------------------------------------------------------
# CHECK UNIQUE LABELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNIQUE LABELS")
print("=" * 70)

for label in sorted(df["label"].dropna().unique()):
    print("-", label)


# ------------------------------------------------------------
# CLASS IMBALANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS IMBALANCE")
print("=" * 70)

largest_class = label_counts.max()
smallest_class = label_counts.min()

imbalance_ratio = largest_class / smallest_class

print(f"Largest class : {largest_class}")
print(f"Smallest class: {smallest_class}")
print(f"Ratio         : {imbalance_ratio:.2f}:1")


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

label_counts.plot(kind="bar")

plt.title("LIAR Training Dataset — Label Distribution")
plt.xlabel("Truthfulness Label")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)

plt.tight_layout()

output_dir = PROJECT_ROOT / "data" / "processed"

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_file = output_dir / "liar_label_distribution.png"

plt.savefig(output_file, dpi=200)

print("\nChart saved to:")
print(output_file)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 12.4.2 COMPLETE")
print("=" * 70)