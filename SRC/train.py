"""
Stage 2: Model Training
────────────────────────
Fine-tunes roberta-base as a binary hallucination classifier on HaluEval.

Usage:
    python src/train.py

Output:
    models/final_model/   ← trained classifier
    models/checkpoints/   ← epoch checkpoints
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from configs.training_config import *


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1":        f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall":    recall_score(labels, preds),
        "accuracy":  accuracy_score(labels, preds)
    }


def prepare_splits():
    if os.path.exists(SPLITS_PATH):
        splits = load_from_disk(SPLITS_PATH)
        print("✓ Splits loaded from cache")
        return splits

    df = pl.read_parquet(DATA_PATH)
    df_pd = df.select(["context", "question", "response", "label"]).to_pandas()

    train_df, val_df = train_test_split(
        df_pd, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_pd["label"]
    )

    splits = DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "val":   Dataset.from_pandas(val_df,   preserve_index=False)
    })
    splits.save_to_disk(SPLITS_PATH)
    print(f"✓ Splits created: train={len(splits['train'])} val={len(splits['val'])}")
    return splits


def tokenize_splits(splits):
    if os.path.exists(TOKENIZED_PATH):
        tokenized = load_from_disk(TOKENIZED_PATH)
        print("✓ Tokenized dataset loaded from cache")
        return tokenized

    os.makedirs(TOKENIZER_CACHE, exist_ok=True)
    if os.path.exists(TOKENIZER_CACHE) and os.listdir(TOKENIZER_CACHE):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CACHE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.save_pretrained(TOKENIZER_CACHE)

    def tokenize(batch):
        second_seq = [q + " " + r for q, r in zip(batch["question"], batch["response"])]
        return tokenizer(
            batch["context"], second_seq,
            truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )

    tokenized = splits.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized.save_to_disk(TOKENIZED_PATH)
    print("✓ Tokenized & saved")
    return tokenized


def train():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if os.path.exists(FINAL_MODEL_PATH):
        print("✓ Final model already exists. Delete models/final_model/ to retrain.")
        return

    # Data
    splits    = prepare_splits()
    tokenized = tokenize_splits(splits)

    # Model
    if os.path.exists(MODEL_CACHE) and os.listdir(MODEL_CACHE):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CACHE, num_labels=NUM_LABELS
        )
        print("✓ Model loaded from cache")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=NUM_LABELS
        )
        os.makedirs(MODEL_CACHE, exist_ok=True)
        model.save_pretrained(MODEL_CACHE)
        print("✓ Model downloaded & cached")

    # Training args
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        fp16=FP16,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC,
        greater_is_better=True,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        compute_metrics=compute_metrics
    )

    import glob
    completed = glob.glob(f"{CHECKPOINT_PATH}/checkpoint-*")
    trainer.train(
        resume_from_checkpoint=completed[-1] if completed else None
    )

    # Save label config
    model.config.id2label = {0: "FAITHFUL", 1: "HALLUCINATED"}
    model.config.label2id = {"FAITHFUL": 0, "HALLUCINATED": 1}
    trainer.save_model(FINAL_MODEL_PATH)
    print(f"\n✓ Training complete. Model saved → {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    train()
