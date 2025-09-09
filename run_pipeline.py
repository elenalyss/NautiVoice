# run_pipeline.py
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, ClassLabel

# Function that transcribes audio to text
from asr_pipeline import transcribe_audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Category labels
cat_ds = load_dataset("csv", data_files={"validation": "val.csv"})
CATEGORIES = sorted(cat_ds["validation"].unique("category"))

# Severity labels
SEVERITIES = ["Low", "Medium", "High", "Critical"]

# We load models and tokenizers 
cat_tok = AutoTokenizer.from_pretrained("nautivoice-category-augm")
cat_mdl = AutoModelForSequenceClassification.from_pretrained("nautivoice-category-augm").to(DEVICE)

sev_tok = AutoTokenizer.from_pretrained("nautivoice-severity-augm-2")
sev_mdl = AutoModelForSequenceClassification.from_pretrained("nautivoice-severity-augm-2").to(DEVICE)

# Softmax layer to convert logits to probabilities
softmax = torch.nn.Softmax(dim=-1)

# Incident Category prediction
def predict_category(text: str):
    enc = cat_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = cat_mdl(**enc).logits
        probs = softmax(logits).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CATEGORIES[idx], float(probs[idx]), probs

# Incident Severity prediction
def predict_severity(text: str):
    enc = sev_tok(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = sev_mdl(**enc).logits
        probs = softmax(logits).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return SEVERITIES[idx], float(probs[idx]), probs

# Pipeline function
def run(audio_path: str):
    # 3) ASR -> report text
    print("Transcribing:", audio_path)
    report = transcribe_audio(audio_path)
    print("\nTranscript:\n", report)

    # Classification
    cat_label, cat_conf, _ = predict_category(report)
    sev_label, sev_conf, _ = predict_severity(report)
    # Predictions and confidence
    print("\n--- Predicted ---")
    print(f"Category: {cat_label}  (conf: {cat_conf:.3f})")
    print(f"Severity: {sev_label} (conf: {sev_conf:.3f})")

# Entry point
if __name__ == "__main__":
   # Running the pipeline with the input audio file
    audio = Path(__file__).parent / "my_report.wav"
    run(str(audio))
