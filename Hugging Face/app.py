"""
Stage 5: Gradio UI — Enhanced
──────────────────────────────
Hallucination Detection & Prompt Remediation System
- RoBERTa classifier  → overall hallucination signal
- DeBERTa NLI         → sentence-level contradiction scoring
- Rule-based engine   → 3-tier corrective prompt generation

Run:
    python app.py
"""

import subprocess
import sys
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
               capture_output=True)

import os
import json
import torch
import spacy
import torch.nn.functional as F
import gradio as gr
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Model loading ─────────────────────────────────────────────────────────────
ROBERTA_REPO = "JBond07/hallucination-detector-roberta"
NLI_REPO     = "cross-encoder/nli-deberta-v3-small"
LOCAL_MODEL  = "models/final_model"
LOCAL_TOK    = "models/tokenizer_saved"

print("Loading models...")
tok_src  = LOCAL_TOK   if os.path.exists(LOCAL_TOK)  else ROBERTA_REPO
mod_src  = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else ROBERTA_REPO

tokenizer     = AutoTokenizer.from_pretrained(tok_src)
model         = AutoModelForSequenceClassification.from_pretrained(mod_src)
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_REPO)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_REPO)
nlp           = spacy.load("en_core_web_sm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
nli_model.to(device).eval()
print(f"✓ All models ready on {device}")

# ── History store ─────────────────────────────────────────────────────────────
history = []

# ── Helpers ───────────────────────────────────────────────────────────────────
PRONOUNS = {"it", "they", "he", "she", "this", "that", "its", "their"}

def resolve_pronouns(sentence, subject):
    tokens = sentence.split()
    if not tokens:
        return sentence
    if tokens[0].rstrip(".,!?").lower() in PRONOUNS:
        tokens[0] = subject
    return " ".join(tokens)

def extract_subject(doc):
    for t in doc:
        if t.dep_ == "nsubj": return t.text
    for t in doc:
        if t.pos_ == "PROPN": return t.text
    for t in doc:
        if t.pos_ == "NOUN":  return t.text
    return doc[0].text if doc else ""

def get_overall_prob(context, question, response):
    inputs = tokenizer(
        context, question + " " + response,
        truncation=True, max_length=512,
        padding="max_length", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)
    return probs[0][1].item()

def get_nli_scores(context, hypothesis):
    inputs = nli_tokenizer(
        context, hypothesis,
        truncation=True, max_length=512,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(nli_model(**inputs).logits, dim=1)
    return {
        "contradiction": round(probs[0][0].item(), 4),
        "neutral":       round(probs[0][1].item(), 4),
        "entailment":    round(probs[0][2].item(), 4)
    }

def flag_sentences(context, question, response, threshold=0.6):
    overall_prob = get_overall_prob(context, question, response)
    doc          = nlp(response)
    sentences    = [s.text.strip() for s in doc.sents if s.text.strip()]
    subject      = extract_subject(nlp(response))

    results = []
    for sent in sentences:
        resolved = resolve_pronouns(sent, subject)
        scores   = get_nli_scores(context, resolved)
        flagged  = scores["contradiction"] > threshold
        results.append({
            "sentence":    sent,
            "resolved":    resolved,
            "scores":      scores,
            "flagged":     flagged,
            "flag_reason": "CONTRADICTION" if flagged else None
        })

    verdict = "HALLUCINATED" if any(r["flagged"] for r in results) else "FAITHFUL"
    return overall_prob, verdict, results

def generate_remediation(context, results):
    flagged = [r for r in results if r["flagged"]]
    if not flagged:
        return "✅ Response is faithful to the source context. No correction needed."

    bullets = "\n".join(
        f'  • "{r["sentence"]}  [{r["flag_reason"]}]"' for r in flagged
    )
    instructions = (
        "- Correct the contradictory claims listed above\n"
        "- Use only facts explicitly present in the context\n"
        "- Do not infer, guess, or add outside knowledge\n"
        "- If information is missing, explicitly state it is not available\n"
        "- Be factually precise and grounded to the source"
    )
    return (
        f"🚨 The following sentence(s) contradict the source context:\n\n"
        f"Flagged content:\n{bullets}\n\n"
        f"Please re-answer using ONLY the following context:\n\n"
        f'"""{context}"""\n\n'
        f"Instructions:\n{instructions}"
    )

# ── Output builders ───────────────────────────────────────────────────────────
def build_verdict_html(overall_prob, verdict):
    pct      = round(overall_prob * 100, 1)
    color    = "#ef4444" if verdict == "HALLUCINATED" else "#22c55e"
    icon     = "🚨" if verdict == "HALLUCINATED" else "✅"
    bar_pct  = pct if verdict == "HALLUCINATED" else 100 - pct
    bar_color = "#ef4444" if verdict == "HALLUCINATED" else "#22c55e"

    return f"""
<div style="font-family:sans-serif; padding:16px; border-radius:10px;
            border: 2px solid {color}; background:#1a1a2e;">
  <div style="font-size:1.4em; font-weight:bold; color:{color};">
    {icon} {verdict}
  </div>
  <div style="margin-top:8px; color:#aaa; font-size:0.9em;">
    Hallucination Probability: <strong style="color:{color};">{pct}%</strong>
  </div>
  <div style="margin-top:8px; background:#333; border-radius:6px; height:12px; overflow:hidden;">
    <div style="width:{bar_pct}%; background:{bar_color};
                height:100%; border-radius:6px; transition:width 0.4s;"></div>
  </div>
  <div style="margin-top:6px; color:#777; font-size:0.75em;">
    {"High hallucination risk detected" if verdict == "HALLUCINATED" else "Response appears grounded in context"}
  </div>
</div>"""

def build_breakdown_html(results):
    rows = ""
    for r in results:
        s       = r["scores"]
        icon    = "🚨" if r["flagged"] else "✓"
        bg      = "#3b0000" if r["flagged"] else "#0a2a0a"
        border  = "#ef4444" if r["flagged"] else "#22c55e"
        label   = f'<span style="color:#ef4444; font-weight:bold;">[{r["flag_reason"]}]</span>' \
                  if r["flagged"] else '<span style="color:#22c55e;">Supported</span>'

        e_color = "#22c55e" if s["entailment"] > 0.5 else "#aaa"
        c_color = "#ef4444" if s["contradiction"] > 0.3 else "#aaa"

        resolved_note = ""
        if r["sentence"] != r["resolved"]:
            resolved_note = f'<div style="color:#888; font-size:0.78em; margin-top:4px;">📝 Resolved: "{r["resolved"]}"</div>'

        rows += f"""
<div style="margin-bottom:10px; padding:12px; border-radius:8px;
            background:{bg}; border-left:4px solid {border};">
  <div style="font-size:0.95em; color:#eee;">
    {icon} {r['sentence']}
  </div>
  {resolved_note}
  <div style="margin-top:8px; display:flex; gap:12px; flex-wrap:wrap;">
    <span style="font-size:0.78em; color:{e_color};">
      Entailment: <strong>{s['entailment']}</strong>
    </span>
    <span style="font-size:0.78em; color:#aaa;">
      Neutral: <strong>{s['neutral']}</strong>
    </span>
    <span style="font-size:0.78em; color:{c_color};">
      Contradiction: <strong>{s['contradiction']}</strong>
    </span>
    <span style="font-size:0.78em;">{label}</span>
  </div>
</div>"""

    return f"""
<div style="font-family:sans-serif;">
  <div style="font-size:0.85em; color:#888; margin-bottom:10px;">
    {len(results)} sentence(s) analyzed
    — {sum(1 for r in results if r['flagged'])} flagged
  </div>
  {rows}
</div>"""

def build_history_html():
    if not history:
        return "<div style='color:#666; font-size:0.85em;'>No history yet.</div>"
    rows = ""
    for h in reversed(history[-5:]):
        icon  = "🚨" if h["verdict"] == "HALLUCINATED" else "✅"
        color = "#ef4444" if h["verdict"] == "HALLUCINATED" else "#22c55e"
        rows += f"""
<div style="padding:8px 12px; margin-bottom:6px; border-radius:6px;
            background:#1e1e2e; border-left:3px solid {color};
            font-family:sans-serif; font-size:0.82em;">
  <span style="color:{color};">{icon} {h['verdict']}</span>
  <span style="color:#888; margin-left:8px;">{h['time']}</span>
  <div style="color:#aaa; margin-top:3px; white-space:nowrap;
              overflow:hidden; text-overflow:ellipsis; max-width:400px;">
    {h['response'][:80]}...
  </div>
</div>"""
    return rows

def build_report(context, question, response, verdict, overall_prob, results, remediation):
    lines = [
        "=" * 60,
        "HALLUCINATION DETECTION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "CONTEXT:",
        context,
        "",
        "QUESTION:",
        question or "(none)",
        "",
        "RESPONSE:",
        response,
        "",
        "-" * 60,
        f"VERDICT: {verdict}",
        f"HALLUCINATION PROBABILITY: {round(overall_prob * 100, 1)}%",
        "",
        "SENTENCE ANALYSIS:",
    ]
    for r in results:
        status = "FLAGGED" if r["flagged"] else "OK"
        lines.append(
            f"  [{status}] {r['sentence']}\n"
            f"          E={r['scores']['entailment']} "
            f"N={r['scores']['neutral']} "
            f"C={r['scores']['contradiction']}"
        )
    lines += ["", "-" * 60, "CORRECTIVE PROMPT:", remediation, "=" * 60]
    return "\n".join(lines)

# ── Main analyze function ─────────────────────────────────────────────────────
def analyze(context, question, response):
    if not context.strip() or not response.strip():
        empty = "<div style='color:#888;'>Waiting for input...</div>"
        return empty, empty, "", empty, None

    if not question.strip():
        question = "What does the response claim?"

    overall_prob, verdict, results = flag_sentences(context, question, response)
    remediation = generate_remediation(context, results)

    # History
    history.append({
        "verdict":  verdict,
        "response": response,
        "time":     datetime.now().strftime("%H:%M:%S")
    })

    # Report file
    report_text = build_report(
        context, question, response,
        verdict, overall_prob, results, remediation
    )
    report_path = "/tmp/hallucination_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    return (
        build_verdict_html(overall_prob, verdict),
        build_breakdown_html(results),
        remediation,
        build_history_html(),
        report_path
    )

# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { background: #0f0f1a !important; }
.gr-button-primary { background: #6366f1 !important; border: none !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, theme=gr.themes.Soft(), title="Hallucination Detector") as demo:

    gr.Markdown("""
# 🔍 Hallucination Detection & Prompt Remediation
**Checks whether an LLM response is grounded in the source context.**
Flags contradictory sentences, explains why, and generates a corrective prompt.
""")

    with gr.Row():
        # ── Left: Inputs ──
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")
            context_input = gr.Textbox(
                label="📄 Source Context",
                placeholder="Paste the source document or reference context here...",
                lines=6
            )
            question_input = gr.Textbox(
                label="❓ Question (optional)",
                placeholder="What question was the LLM answering? Leave blank if not applicable.",
                lines=2
            )
            response_input = gr.Textbox(
                label="🤖 LLM Response to Check",
                placeholder="Paste the LLM-generated response here...",
                lines=6
            )
            analyze_btn = gr.Button("🔍 Analyze Response", variant="primary", size="lg")

        # ── Right: Outputs ──
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            verdict_output = gr.HTML(
                label="Overall Verdict",
                value="<div style='color:#555; font-family:sans-serif; padding:16px;'>Run analysis to see verdict.</div>"
            )
            breakdown_output = gr.HTML(
                label="Sentence-Level Breakdown",
                value="<div style='color:#555; font-family:sans-serif; padding:16px;'>Sentence analysis will appear here.</div>"
            )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🛠️ Corrective Prompt")
            gr.Markdown(
                "<span style='font-size:0.85em; color:#888;'>"
                "Copy this prompt and paste it back into your LLM to get a grounded response."
                "</span>"
            )
            remediation_output = gr.Textbox(
                label="",
                interactive=False,
                lines=10,
                show_copy_button=True
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📥 Export & History")
            download_btn = gr.File(label="⬇️ Download Report", interactive=False)
            gr.Markdown("### 🕒 Recent Tests")
            history_output = gr.HTML(
                value="<div style='color:#666; font-size:0.85em;'>No history yet.</div>"
            )

    # ── Examples ──
    gr.Markdown("### 💡 Try These Examples")
    gr.Examples(
        examples=[
            [
                "The Eiffel Tower is located in Paris, France. It was built in 1889.",
                "Where is the Eiffel Tower and when was it built?",
                "The Eiffel Tower is in Paris. It was constructed in 1799. It is the tallest structure in Europe."
            ],
            [
                "Python was created by Guido van Rossum and released in 1991.",
                "Who created Python and when?",
                "Python was created by Guido van Rossum. It was released in 1991."
            ],
            [
                "The Amazon River flows through Brazil and is the largest river by discharge.",
                "What is the Amazon River known for?",
                "The Amazon River is the largest river in the world. It is also the longest."
            ],
            [
                "Albert Einstein was born in Ulm, Germany in 1879. He developed the theory of relativity.",
                "Where was Einstein born and what is he known for?",
                "Einstein was born in Berlin. He is known for inventing the telephone."
            ],
        ],
        inputs=[context_input, question_input, response_input],
        label=""
    )

    # ── Wiring ──
    analyze_btn.click(
        fn=analyze,
        inputs=[context_input, question_input, response_input],
        outputs=[
            verdict_output,
            breakdown_output,
            remediation_output,
            history_output,
            download_btn
        ]
    )

demo.launch()
