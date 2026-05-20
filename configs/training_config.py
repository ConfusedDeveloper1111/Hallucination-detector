# ── Training Configuration ──────────────────────────────────────────────────
# All hyperparameters in one place. Edit here before training.

# Paths
DATA_PATH          = "data/halueval_clean.parquet"
SPLITS_PATH        = "data/splits_saved"
TOKENIZED_PATH     = "data/tokenized_saved"
MODEL_CACHE        = "models/roberta_base"
TOKENIZER_CACHE    = "models/tokenizer_saved"
CHECKPOINT_PATH    = "models/checkpoints"
FINAL_MODEL_PATH   = "models/final_model"

# Model
BASE_MODEL         = "roberta-base"
NUM_LABELS         = 2
MAX_LENGTH         = 512

# Training
TRAIN_BATCH_SIZE   = 8
GRAD_ACCUM_STEPS   = 4        # effective batch = 32
FP16               = True
NUM_EPOCHS         = 3
SAVE_TOTAL_LIMIT   = 2
METRIC             = "f1"
TEST_SIZE          = 0.1
RANDOM_STATE       = 42

# Inference thresholds
CONTRADICT_THRESHOLD = 0.6    # NLI contradiction threshold for sentence flagging
NLI_MODEL            = "cross-encoder/nli-deberta-v3-small"

# HuggingFace Hub
HF_ROBERTA_REPO    = "JBond07/hallucination-detector-roberta"
