# run_pipeline.py
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, ClassLabel

# 1) Φέρνουμε τη συνάρτηση transcribe από το asr_pipeline.py
from asr_pipeline import transcribe_audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Φόρτωμα Category labels (ίδια σειρά με training) ---
# Θα τα πάρουμε όπως στο evaluation σου: sorted unique από val.csv
cat_ds = load_dataset("csv", data_files={"validation": "val.csv"})
CATEGORIES = sorted(cat_ds["validation"].unique("category"))

# --- Severity labels (σταθερή λίστα όπως εκπαιδεύτηκε) ---
SEVERITIES = ["Low", "Medium", "High", "Critical"]

# 2) Φόρτωση μοντέλων
cat_tok = AutoTokenizer.from_pretrained("nautivoice-category-augm")
cat_mdl = AutoModelForSequenceClassification.from_pretrained("nautivoice-category-augm").to(DEVICE)

sev_tok = AutoTokenizer.from_pretrained("nautivoice-severity-augm-2")
sev_mdl = AutoModelForSequenceClassification.from_pretrained("nautivoice-severity-augm-2").to(DEVICE)

softmax = torch.nn.Softmax(dim=-1)

def predict_category(text: str):
    enc = cat_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = cat_mdl(**enc).logits
        probs = softmax(logits).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CATEGORIES[idx], float(probs[idx]), probs

def predict_severity(text: str):
    enc = sev_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = sev_mdl(**enc).logits
        probs = softmax(logits).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return SEVERITIES[idx], float(probs[idx]), probs

def run(audio_path: str):
    # 3) ASR -> report text
    print("Transcribing:", audio_path)
    report = transcribe_audio(audio_path)
    print("\nTranscript:\n", report)

    # 4) Classification
    cat_label, cat_conf, _ = predict_category(report)
    sev_label, sev_conf, _ = predict_severity(report)

    print("\n--- Predicted ---")
    print(f"Category: {cat_label}  (conf: {cat_conf:.3f})")
    print(f"Severity: {sev_label} (conf: {sev_conf:.3f})")

if __name__ == "__main__":
    # βάλε εδώ το αρχείο που θες να τρέξεις
    audio = Path(__file__).parent / "my_report.wav"
    run(str(audio))
