#!/bin/bash
# ── One-command setup ─────────────────────────────────────────────────────────
# Usage: bash setup.sh

echo "Setting up Hallucination Detector..."

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Create required directories
mkdir -p data models/checkpoints

echo ""
echo "✓ Setup complete."
echo ""
echo "Next steps:"
echo "  Option A — Run demo instantly (loads pretrained model from HuggingFace Hub):"
echo "    python app.py"
echo ""
echo "  Option B — Train from scratch:"
echo "    python src/data_prep.py    # download & clean dataset"
echo "    python src/train.py        # fine-tune RoBERTa (~20-30 min on GPU)"
echo "    python app.py              # launch app with your trained model"
