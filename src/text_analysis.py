from pathlib import Path
import re
import pandas as pd
import numpy as np


# ============================================================
# FAKE NEWS DETECTOR
# PHASE 12.4.4 — NLP TEXT ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIAR_DIR = PROJECT_ROOT / "data" / "raw" / "liar"

TRAIN_FILE = LIAR_DIR / "train.tsv"


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
# BASIC TEXT CLEANING FOR ANALYSIS
# ------------------------------------------------------------

df["statement"] = df["statement"].fillna("").astype(str)

df["word_count"] = (
    df["statement"]
    .str.split()
    .str.len()
)

df["character_count"] = (
    df["statement"]
    .str.len()
)

df["sentence_count"] = (
    df["statement"]
    .apply(
        lambda x: max(
            1,
            len(
                re.findall(
                    r"[.!?]+",
                    x
                )
            )
        )
    )
)


# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------

print("=" * 70)
print("LIAR DATASET — NLP TEXT ANALYSIS")
print("=" * 70)

print(f"\nTotal statements: {len(df):,}")


print("\n" + "=" * 70)
print("WORD COUNT STATISTICS")
print("=" * 70)

print(
    df["word_count"].describe().round(2)
)


print("\n" + "=" * 70)
print("CHARACTER COUNT STATISTICS")
print("=" * 70)

print(
    df["character_count"].describe().round(2)
)


print("\n" + "=" * 70)
print("SENTENCE COUNT STATISTICS")
print("=" * 70)

print(
    df["sentence_count"].describe().round(2)
)


# ------------------------------------------------------------
# EXTREMELY SHORT / LONG STATEMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERY SHORT STATEMENTS")
print("=" * 70)

short = df.nsmallest(
    10,
    "word_count"
)

print(
    short[
        [
            "label",
            "word_count",
            "statement"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("VERY LONG STATEMENTS")
print("=" * 70)

long = df.nlargest(
    10,
    "word_count"
)

print(
    long[
        [
            "label",
            "word_count",
            "statement"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# WORD LENGTH DISTRIBUTION BY LABEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AVERAGE TEXT LENGTH BY LABEL")
print("=" * 70)

length_by_label = (
    df.groupby("label")
    .agg(
        statements=("statement", "count"),
        avg_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        avg_characters=("character_count", "mean"),
    )
    .sort_values(
        "avg_words",
        ascending=False
    )
)

print(
    length_by_label.round(2)
)


# ------------------------------------------------------------
# PUNCTUATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PUNCTUATION / SPECIAL CHARACTER ANALYSIS")
print("=" * 70)


df["question_marks"] = (
    df["statement"].str.count(r"\?")
)

df["exclamation_marks"] = (
    df["statement"].str.count(r"!")
)

df["numbers"] = (
    df["statement"].str.count(r"\d")
)

df["uppercase_letters"] = (
    df["statement"].apply(
        lambda x: sum(
            1 for c in x
            if c.isupper()
        )
    )
)


print(
    f"Total question marks : "
    f"{df['question_marks'].sum():,}"
)

print(
    f"Total exclamation marks: "
    f"{df['exclamation_marks'].sum():,}"
)

print(
    f"Total numeric characters: "
    f"{df['numbers'].sum():,}"
)

print(
    f"Total uppercase letters: "
    f"{df['uppercase_letters'].sum():,}"
)


# ------------------------------------------------------------
# URL DETECTION
# ------------------------------------------------------------

url_pattern = r"https?://\S+|www\.\S+"

df["contains_url"] = (
    df["statement"]
    .str.contains(
        url_pattern,
        regex=True,
        case=False,
        na=False
    )
)

print("\nStatements containing URLs:")
print(
    df["contains_url"].sum()
)


# ------------------------------------------------------------
# WORD VOCABULARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VOCABULARY ANALYSIS")
print("=" * 70)

all_words = []

for text in df["statement"]:

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    all_words.extend(words)


unique_words = set(all_words)

print(
    f"Total word occurrences : "
    f"{len(all_words):,}"
)

print(
    f"Unique vocabulary words: "
    f"{len(unique_words):,}"
)

if len(all_words) > 0:

    type_token_ratio = (
        len(unique_words) /
        len(all_words)
    )

    print(
        f"Type-token ratio       : "
        f"{type_token_ratio:.4f}"
    )


# ------------------------------------------------------------
# MOST COMMON WORDS
# ------------------------------------------------------------

from collections import Counter

word_frequency = Counter(all_words)

print("\n" + "=" * 70)
print("MOST COMMON WORDS")
print("=" * 70)

for word, count in word_frequency.most_common(30):

    print(
        f"{word:20} {count:,}"
    )


# ------------------------------------------------------------
# MOST RARE WORDS
# ------------------------------------------------------------

rare_words = [
    word
    for word, count
    in word_frequency.items()
    if count == 1
]

print("\n" + "=" * 70)
print("RARE VOCABULARY")
print("=" * 70)

print(
    f"Words appearing exactly once: "
    f"{len(rare_words):,}"
)


# ------------------------------------------------------------
# TEXT LENGTH EXTREMES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEXT LENGTH EXTREMES")
print("=" * 70)

print(
    "Shortest statement:",
    df.loc[
        df["word_count"].idxmin(),
        "statement"
    ]
)

print(
    "\nLongest statement:",
    df.loc[
        df["word_count"].idxmax(),
        "statement"
    ]
)


# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 12.4.4 COMPLETE")
print("=" * 70)