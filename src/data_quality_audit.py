from pathlib import Path
import pandas as pd


# ============================================================
# FAKE NEWS DETECTOR
# PHASE 12.4.3 — DATA QUALITY AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIAR_DIR = PROJECT_ROOT / "data" / "raw" / "liar"

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


def load_split(filename):

    path = LIAR_DIR / filename

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=3,
    )


# ------------------------------------------------------------
# LOAD ALL SPLITS
# ------------------------------------------------------------

train = load_split("train.tsv")
valid = load_split("valid.tsv")
test = load_split("test.tsv")


print("=" * 70)
print("LIAR DATASET — DATA QUALITY AUDIT")
print("=" * 70)


# ------------------------------------------------------------
# BASIC SHAPES
# ------------------------------------------------------------

print("\nDATASET SIZES")

print(f"Train      : {train.shape}")
print(f"Validation : {valid.shape}")
print(f"Test       : {test.shape}")


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES — TRAIN")
print("=" * 70)

missing = train.isna().sum()

print(
    missing[
        missing > 0
    ].sort_values(
        ascending=False
    )
)


# ------------------------------------------------------------
# DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print(f"Train duplicates: {train.duplicated().sum()}")
print(f"Valid duplicates: {valid.duplicated().sum()}")
print(f"Test duplicates : {test.duplicated().sum()}")


# ------------------------------------------------------------
# DUPLICATE STATEMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE STATEMENTS")
print("=" * 70)

print(
    f"Train duplicate statements: "
    f"{train['statement'].duplicated().sum()}"
)

print(
    f"Validation duplicate statements: "
    f"{valid['statement'].duplicated().sum()}"
)

print(
    f"Test duplicate statements: "
    f"{test['statement'].duplicated().sum()}"
)


# ------------------------------------------------------------
# DUPLICATE IDs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE IDS")
print("=" * 70)

print(
    f"Train duplicate IDs: "
    f"{train['id'].duplicated().sum()}"
)

print(
    f"Validation duplicate IDs: "
    f"{valid['id'].duplicated().sum()}"
)

print(
    f"Test duplicate IDs: "
    f"{test['id'].duplicated().sum()}"
)


# ------------------------------------------------------------
# EMPTY STATEMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EMPTY / VERY SHORT STATEMENTS")
print("=" * 70)

empty_train = train["statement"].fillna("").str.strip().eq("")

short_train = (
    train["statement"]
    .fillna("")
    .str.split()
    .str.len()
    .lt(3)
)

print(f"Empty statements: {empty_train.sum()}")
print(f"Statements < 3 words: {short_train.sum()}")


# ------------------------------------------------------------
# UNIQUE LABELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL VALIDATION")
print("=" * 70)

expected_labels = {
    "true",
    "mostly-true",
    "half-true",
    "barely-true",
    "false",
    "pants-fire",
}

actual_labels = set(
    train["label"].dropna().unique()
)

print("Unexpected labels:")
print(actual_labels - expected_labels)


# ------------------------------------------------------------
# CROSS-SPLIT STATEMENT OVERLAP
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-SPLIT STATEMENT OVERLAP")
print("=" * 70)

train_statements = set(
    train["statement"].dropna()
)

valid_statements = set(
    valid["statement"].dropna()
)

test_statements = set(
    test["statement"].dropna()
)

train_valid_overlap = (
    train_statements & valid_statements
)

train_test_overlap = (
    train_statements & test_statements
)

valid_test_overlap = (
    valid_statements & test_statements
)

print(
    f"Train ∩ Validation: "
    f"{len(train_valid_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(valid_test_overlap)}"
)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 12.4.3 COMPLETE")
print("=" * 70)