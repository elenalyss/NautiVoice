# app.py
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ------- Optional fallback decoder for non-WAV (webm/mp3/m4a) -------
from pydub import AudioSegment  # uses system ffmpeg

# ---------------- Settings ----------------
DEVICE = 0 if torch.cuda.is_available() else -1
SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]

# labels in the correct order (like in training)
CATEGORY_LABELS = [
    "Accident to person(s)",
    "Capsizing / Listing",
    "Collision",
    "Contact",
    "Damage / Loss Of Equipment",
    "Fire / Explosion",
    "Flooding / Foundering",
    "Grounding / Stranding",
    "Loss Of Control",
]

# Paths of our fine‑tuned models
CAT_MODEL_DIR = "nautivoice-category-augm"
SEV_MODEL_DIR = "nautivoice-severity-augm-2"

# -------------- Helpers (/functions) --------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def read_wav_to_array(file_bytes: bytes):
    """Διαβάζει WAV bytes -> (float32 mono array, sr)."""
    data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    if data.ndim == 2:  # stereo -> mono
        data = data.mean(axis=1)
    return data, sr

def read_audio_any(file_bytes: bytes, filename: str):
    """
    Διάβασε *οτιδήποτε* (.wav/.webm/.mp3/.m4a…) -> (float32 mono array, sr).
    1) Δοκίμασε WAV με soundfile
    2) Fallback: pydub/ffmpeg για άλλα formats
    """
    name = (filename or "").lower()
    # we will try with WAV first
    try:
        if name.endswith(".wav"):
            return read_wav_to_array(file_bytes)
        # some times it is coming as WAV with generic name  
        # so let's try soundfile anyway:
        return read_wav_to_array(file_bytes)
    except Exception:
        pass

    # Fallback με pydub/ffmpeg για webm/mp3/m4a
    try:
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        # normalization from int16 -> float32 [-1,1]
        samples /= 32768.0
        return samples, seg.frame_rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported/corrupted audio: {repr(e)}")

# ---------------- Models (load once) ----------------
# Whisper ASR (raw array in)
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",   # for faster -> "openai/whisper-base"
    device=DEVICE,
    chunk_length_s=30,
    generate_kwargs={"task": "transcribe"},
    return_timestamps=False,
    ignore_warning=True,
)

# Category
cat_tokenizer = AutoTokenizer.from_pretrained(CAT_MODEL_DIR)
cat_model     = AutoModelForSequenceClassification.from_pretrained(CAT_MODEL_DIR)
cat_model.eval()

# Severity
sev_tokenizer = AutoTokenizer.from_pretrained(SEV_MODEL_DIR)
sev_model     = AutoModelForSequenceClassification.from_pretrained(SEV_MODEL_DIR)
sev_model.eval()

# ---------------- FastAPI ----------------
app = FastAPI(title="NautiVoice API")

# CORS to be called by Bolt/Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI if exists
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h3>NautiVoice API</h3><p>POST /predict με αρχείο audio (form-data: file)</p>"#

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Δέχεται audio από Bolt/web UI (wav/webm/mp3/m4a), κάνει:
      - ASR (Whisper) -> report text
      - Category & Severity classification
    """
    bytes_data = await file.read()
    if not bytes_data:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) We read sound as (array, sr) with robust reader
    audio, sr = read_audio_any(bytes_data, file.filename)

    # 2) ASR
    try:
        asr_out = asr({"array": audio, "sampling_rate": sr})
        report = (asr_out.get("text") or "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR failed: {repr(e)}")

    if not report:
        # For viewable feedback in UI
        return JSONResponse({
            "transcript": "",
            "category": {"label": "Unknown", "scores": {}, "confidence": 0.0},
            "severity": {"label": "Unknown", "scores": {}, "confidence": 0.0},
        })

    # 3) Category classification
    try:
        cat_inputs = cat_tokenizer(report, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = cat_model(**{k: v for k, v in cat_inputs.items()}).logits
        cat_logits = logits.detach().cpu().numpy()[0]
        cat_probs  = softmax(cat_logits)
        cat_idx    = int(np.argmax(cat_probs))
        cat_label  = CATEGORY_LABELS[cat_idx] if 0 <= cat_idx < len(CATEGORY_LABELS) else str(cat_idx)
        cat_scores = {CATEGORY_LABELS[i]: float(cat_probs[i])
                      for i in range(min(len(CATEGORY_LABELS), len(cat_probs)))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Category classification failed: {repr(e)}")

    # 4) Severity classification
    try:
        sev_inputs = sev_tokenizer(report, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = sev_model(**{k: v for k, v in sev_inputs.items()}).logits
        sev_logits = logits.detach().cpu().numpy()[0]
        sev_probs  = softmax(sev_logits)
        sev_idx    = int(np.argmax(sev_probs))
        sev_label  = SEVERITY_LABELS[sev_idx]
        sev_scores = {SEVERITY_LABELS[i]: float(sev_probs[i]) for i in range(len(SEVERITY_LABELS))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Severity classification failed: {repr(e)}")

    return JSONResponse({
        "transcript": report,
        "category": {"label": cat_label, "scores": cat_scores, "confidence": float(cat_probs[cat_idx])},
        "severity": {"label": sev_label, "scores": sev_scores, "confidence": float(sev_probs[sev_idx])},
    })
