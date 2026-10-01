from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# STANDARD SCHEMA
# ============================================================

STANDARD_COLUMNS = [
    "id",
    "text",
    "label",
    "source_dataset",
    "original_label",
    "title",
    "author",
    "source",
    "url",
    "date",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def empty_standard_dataframe():
    """Create an empty dataframe with the standard schema."""
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def clean_text(text):
    """Basic text normalization."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def standardize_dataframe(
    df,
    dataset_name,
    text_column,
    label_column,
    title_column=None,
    author_column=None,
    source_column=None,
    url_column=None,
    date_column=None,
):
    """
    Convert a dataset into our common schema.
    """

    result = empty_standard_dataframe()

    result["text"] = df[text_column].apply(clean_text)
    result["original_label"] = df[label_column].astype(str)

    result["source_dataset"] = dataset_name

    # Preserve optional information
    if title_column and title_column in df.columns:
        result["title"] = df[title_column].fillna("").astype(str)

    if author_column and author_column in df.columns:
        result["author"] = df[author_column].fillna("").astype(str)

    if source_column and source_column in df.columns:
        result["source"] = df[source_column].fillna("").astype(str)

    if url_column and url_column in df.columns:
        result["url"] = df[url_column].fillna("").astype(str)

    if date_column and date_column in df.columns:
        result["date"] = df[date_column].fillna("").astype(str)

    # Temporary unique ID
    result["id"] = [
        f"{dataset_name}_{i}"
        for i in range(len(result))
    ]

    return result


# ============================================================
# LIAR DATASET
# ============================================================

def load_liar():
    print("\n" + "=" * 70)
    print("LOADING LIAR")
    print("=" * 70)

    LIAR_DIR = RAW_DIR / "liar"

    train_path = LIAR_DIR / "train.tsv"
    valid_path = LIAR_DIR / "valid.tsv"
    test_path = LIAR_DIR / "test.tsv"

    columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job",
        "state",
        "party",
        "barely_true",
        "false",
        "half_true",
        "mostly_true",
        "pants_fire",
        "context",
    ]

    frames = []

    for split_name, path in [
        ("train", train_path),
        ("valid", valid_path),
        ("test", test_path),
    ]:

        if not path.exists():
            print(f"WARNING: Missing {path}")
            continue

        print(f"Reading {split_name}: {path.name}")

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=columns,
        )

        df["split"] = split_name

        frames.append(df)

        print(f"  Rows: {len(df):,}")

    if not frames:
        print("LIAR dataset not found.")
        return empty_standard_dataframe()

    df = pd.concat(frames, ignore_index=True)

    result = standardize_dataframe(
        df=df,
        dataset_name="LIAR",
        text_column="statement",
        label_column="label",
        source_column="speaker",
    )

    print(f"LIAR standardized rows: {len(result):,}")

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("FAKE NEWS DETECTOR")
    print("UNIFIED DATASET PIPELINE")
    print("=" * 70)

    liar = load_liar()

    print("\n" + "=" * 70)
    print("LIAR PREVIEW")
    print("=" * 70)

    print(liar.head(10).to_string())

    print("\n" + "=" * 70)
    print("LABEL DISTRIBUTION")
    print("=" * 70)

    print(liar["original_label"].value_counts())

    output_path = PROCESSED_DIR / "liar_standardized.csv"

    liar.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\nSaved:")
    print(output_path)

    print("\nRows:", len(liar))
    print("Columns:", len(liar.columns))

    print("\nPipeline stage completed successfully.")


if __name__ == "__main__":
    main()