"""
Stage 3 + 4: Inference & Prompt Remediation
─────────────────────────────────────────────
NLI-based sentence flagging + rule-based corrective prompt generation.

Architecture:
- RoBERTa classifier  → overall hallucination signal
- DeBERTa NLI         → sentence-level entailment/contradiction scoring
- Remediation engine  → 3-tier rule-based corrective prompt

Usage:
    from src.inference import load_models, flag_sentences, generate_remediation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from configs.training_config import (
    FINAL_MODEL_PATH, TOKENIZER_CACHE,
    NLI_MODEL, CONTRADICT_THRESHOLD, HF_ROBERTA_REPO
)

# ── Globals ──────────────────────────────────────────────────────────────────
_model         = None
_tokenizer     = None
_nli_model     = None
_nli_tokenizer = None
_nlp           = None
_device        = None

PRONOUNS = {"it", "they", "he", "she", "this", "that", "its", "their"}


def load_models():
    global _model, _tokenizer, _nli_model, _nli_tokenizer, _nlp, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    # RoBERTa classifier
    roberta_src = FINAL_MODEL_PATH if os.path.exists(FINAL_MODEL_PATH) else HF_ROBERTA_REPO
    print(f"Loading RoBERTa from: {roberta_src}")
    _tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_CACHE if os.path.exists(TOKENIZER_CACHE) else roberta_src
    )
    _model = AutoModelForSequenceClassification.from_pretrained(roberta_src)
    _model.to(_device).eval()
    print("✓ RoBERTa loaded")

    # NLI model
    print(f"Loading NLI model: {NLI_MODEL}")
    _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    _nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    _nli_model.to(_device).eval()
    print("✓ NLI model loaded")

    # SpaCy
    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        _nlp = spacy.load("en_core_web_sm")
    print("✓ SpaCy loaded")
    print(f"✓ All models ready on {_device}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_pronouns(sentence, subject):
    tokens = sentence.split()
    if not tokens:
        return sentence
    if tokens[0].rstrip(".,!?").lower() in PRONOUNS:
        tokens[0] = subject
    return " ".join(tokens)


def extract_subject(doc):
    for t in doc:
        if t.dep_ == "nsubj":
            return t.text
    for t in doc:
        if t.pos_ == "PROPN":
            return t.text
    for t in doc:
        if t.pos_ == "NOUN":
            return t.text
    return doc[0].text if doc else ""


# ── Core inference ─────────────────────────────────────────────────────────────
def get_overall_verdict(context, question, response):
    second_seq = question + " " + response
    inputs = _tokenizer(
        context, second_seq,
        truncation=True, max_length=512,
        padding="max_length", return_tensors="pt"
    ).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(**inputs).logits, dim=1)
    return probs[0][1].item()


def get_entailment_score(context, hypothesis):
    """Returns (entailment_prob, contradiction_prob)"""
    inputs = _nli_tokenizer(
        context, hypothesis,
        truncation=True, max_length=512,
        return_tensors="pt"
    ).to(_device)
    with torch.no_grad():
        probs = F.softmax(_nli_model(**inputs).logits, dim=1)
    return probs[0][2].item(), probs[0][0].item()


def flag_sentences(context, question, response,
                   contradict_threshold=CONTRADICT_THRESHOLD):
    if _model is None:
        load_models()

    overall_prob = get_overall_verdict(context, question, response)

    doc       = _nlp(response)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    subject   = extract_subject(_nlp(response))

    results = []
    for sent in sentences:
        resolved        = resolve_pronouns(sent, subject)
        entail, contradict = get_entailment_score(context, resolved)
        flagged         = contradict > contradict_threshold
        flag_reason     = "CONTRADICTION" if flagged else None

        results.append({
            "sentence":           sent,
            "resolved":           resolved,
            "entailment_prob":    round(entail, 4),
            "contradiction_prob": round(contradict, 4),
            "flagged":            flagged,
            "flag_reason":        flag_reason
        })

    any_flagged = any(r["flagged"] for r in results)
    verdict     = "HALLUCINATED" if any_flagged else "FAITHFUL"
    return overall_prob, verdict, results


# ── Remediation engine ─────────────────────────────────────────────────────────
def generate_remediation(context, verdict, results):
    flagged = [r for r in results if r["flagged"]]

    if not flagged:
        return "✅ No prompt — response is faithful"

    has_contradiction = any(r["flag_reason"] == "CONTRADICTION" for r in flagged)
    flagged_bullets   = "\n".join(
        f'  • "{r["sentence"]}  [{r["flag_reason"]}]"' for r in flagged
    )

    if has_contradiction:
        header = "🚨 The following sentence(s) contradict the source context:"
        instructions = (
            "- Correct the unsupported or contradictory claims listed above\n"
            "- Use only facts explicitly present in the context\n"
            "- Do not infer, guess, or add outside knowledge\n"
            "- If information is missing from the context, explicitly state it is not available\n"
            "- Be factually precise and grounded to the source"
        )
    else:
        header = "⚠️ The response may contain unsupported claims."
        instructions = (
            "- Use only facts explicitly stated in the context above\n"
            "- Do not add information from general knowledge\n"
            "- Do not infer unsupported claims\n"
            '- If the answer is not in the context, say:\n'
            '  "The provided context does not contain this information"\n'
            "- Keep your answer concise and factually grounded"
        )

    return (
        f"{header}\n\n"
        f"Flagged content:\n{flagged_bullets}\n\n"
        f"Please re-answer using ONLY the following context:\n\n"
        f'"""{context}"""\n\n'
        f"Instructions:\n{instructions}"
    )


# ── CLI quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()

    ctx  = "The Eiffel Tower is located in Paris, France. It was built in 1889."
    q    = "Where is the Eiffel Tower and when was it built?"
    resp = "The Eiffel Tower is in Paris. It was constructed in 1799. It is the tallest structure in Europe."

    overall, verdict, results = flag_sentences(ctx, q, resp)
    print(f"\nOverall: {round(overall, 4)} → {verdict}")
    for r in results:
        status = "🚨 FLAGGED" if r["flagged"] else "✓ ok"
        print(f"  {status} [C={r['contradiction_prob']}] → {r['sentence']}")

    print("\n--- Remediation ---")
    print(generate_remediation(ctx, verdict, results))
