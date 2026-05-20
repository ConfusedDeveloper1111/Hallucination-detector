"""
Stage 1: Data Preparation
─────────────────────────
Streams HaluEval QA dataset, cleans schema, saves as local Parquet.

Usage:
    python src/data_prep.py

Output:
    data/halueval_clean.parquet
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from datasets import load_dataset
from configs.training_config import DATA_PATH


def download_and_clean():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        df = pl.read_parquet(DATA_PATH)
        print(f"✓ Dataset already exists: {df.shape[0]} rows")
        print(df.head(2))
        return df

    print("Streaming HaluEval QA dataset...")
    dataset = load_dataset(
        "pminervini/HaluEval", "qa_samples",
        split="data", streaming=True
    )

    rows = []
    for row in dataset:
        rows.append({
            "context":  row["knowledge"],
            "question": row["question"],
            "response": row["answer"],
            "label":    1 if row["hallucination"] == "yes" else 0
        })

    df = pl.DataFrame(rows)

    print(f"\n✓ Rows: {df.shape[0]}")
    print(f"✓ Columns: {df.columns}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())

    df.write_parquet(DATA_PATH)
    print(f"\n✓ Saved → {DATA_PATH}")
    return df


if __name__ == "__main__":
    download_and_clean()
