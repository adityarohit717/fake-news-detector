from pathlib import Path
import pandas as pd


# ============================================================
# FAKE NEWS DETECTOR
# PHASE 12.4 — DATASET FORENSICS
# Dataset: LIAR
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIAR_DIR = PROJECT_ROOT / "data" / "raw" / "liar"

TRAIN_FILE = LIAR_DIR / "train.tsv"
VALID_FILE = LIAR_DIR / "valid.tsv"
TEST_FILE = LIAR_DIR / "test.tsv"


# ------------------------------------------------------------
# 2. LIAR DATASET COLUMN NAMES
# ------------------------------------------------------------

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
# 3. LOAD DATASET
# ------------------------------------------------------------

def load_dataset(file_path: Path) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=COLUMNS,
        quoting=3,
    )

    return df


# ------------------------------------------------------------
# 4. DATASET OVERVIEW
# ------------------------------------------------------------

def show_dataset_overview(name: str, df: pd.DataFrame):

    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for index, column in enumerate(df.columns):
        print(f"{index:2}. {column}")


# ------------------------------------------------------------
# 5. LABEL INFORMATION
# ------------------------------------------------------------

def show_labels(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print(df["label"].value_counts(dropna=False))

    print("\nPercentage distribution:")
    print(
        (df["label"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )


# ------------------------------------------------------------
# 6. SAMPLE RECORDS
# ------------------------------------------------------------

def show_samples(df: pd.DataFrame):

    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)

    columns_to_show = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "party",
        "context",
    ]

    print(
        df[columns_to_show]
        .head(5)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():

    print("\n")
    print("=" * 70)
    print("FAKE NEWS DETECTOR — DATASET FORENSICS")
    print("Dataset: LIAR")
    print("=" * 70)

    train_df = load_dataset(TRAIN_FILE)
    valid_df = load_dataset(VALID_FILE)
    test_df = load_dataset(TEST_FILE)

    show_dataset_overview("Training", train_df)
    show_dataset_overview("Validation", valid_df)
    show_dataset_overview("Testing", test_df)

    show_labels(train_df)

    show_samples(train_df)

    print("\n" + "=" * 70)
    print("DATASET FORENSICS 12.4.1 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()