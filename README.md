# NautiVoice – Voice-enabled Maritime Incident Reporting

## Overview
NautiVoice is a prototype application designed to support **voice-based reporting of maritime incidents**.  
It integrates:
- **Automatic Speech Recognition (ASR)** using OpenAI Whisper  
- **Incident Classification** (fine-tuned BERT models for *Category* & *Severity*)  
- **FastAPI backend** serving the models as an API  
- A simple **web frontend** for recording incidents and viewing predictions  

The system allows a user to report an incident through voice input. The audio is transcribed into text, classified into the correct **incident category** and **severity level**.

---

## Requirements
- **Python 3.10+**
- **pip** (Python package manager)
- **ffmpeg** (for audio processing – must be installed separately)
- Recommended: use a **virtual environment** (venv/conda)

---

## Installation

1. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate      # on Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ffmpeg**
   - On macOS (Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - On Ubuntu/Debian:
     ```bash
     sudo apt-get install ffmpeg
     ```
   - On Windows:  
     Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

4. **Download the fine-tuned models**
   The fine-tuned models are now available as downloadable links from Google Drive due to GitHub size limitation.
   Please check the models_link.md file in the repository for download links.
   Download and extract the models into the project root, so the structure looks like this:
   nautivoice-category-augm/
   nautivoice-severity-augm-2/

---

## Running the Application

1. **Start the FastAPI backend**
   ```bash
   python -m uvicorn app:app --reload --port 8000
   ```
   This will start the API at:
   ```
   http://127.0.0.1:8000
   ```

2. **Open the Web UI**  
   Go to `http://127.0.0.1:8000/` in your browser.  
   You can record incidents directly and see predicted **Category** & **Severity**.

---

## Project Structure
```
.
├── app.py                      # FastAPI backend (ASR + Classification API)
├── asr_pipeline.py             # Standalone ASR pipeline (Whisper)
├── train_category_augm.py      # Training script for Category model
├── evaluate_category_augm.py   # Evaluation script for Category model
├── train_severity_augm_2.py    # Training script for Severity model
├── evaluate_severity_augm2.py  # Evaluation script for Severity model
├── run_pipeline.py             # Example pipeline runner
├── requirements.txt            # Dependencies
├── models_link.md              # Download links for fine-tuned models
├── static/                     # Frontend (index.html, JS, CSS)
├── nautivoice-category-augm/   # Fine-tuned Category model (after download from google drive)
└── nautivoice-severity-augm-2/ # Fine-tuned Severity model (after download from google drive)
```

---

## Features

~ Voice input via browser  
~ Real-time ASR using Whisper  
~ Incident classification using BERT  
~ Web UI + FastAPI backend

## TODO / Future Work

- Add user authentication  
- Store incidents in a database  
- Improve UI design  
- Support multiple languages

---


## Troubleshooting
- **Audio not recognized / ASR error** → Ensure `ffmpeg` is correctly installed and accessible in your PATH.  
- **Model not found** → Verify the fine-tuned model folders are placed correctly (`nautivoice-category-augm` and `nautivoice-severity-augm-2`).  
- **Port already in use** → Run with a different port:
  ```bash
  uvicorn app:app --reload --port 8080
  ```

---

## Authors / Roles
- **Elena** – Category classification training, FastAPI backend, integration of frontend  
- **Natalia** – Severity classification training, data preprocessing/annotation, evaluation, documentation  

---

## License
This project was developed as part of the **Machine Learning mini-project** for educational purposes.
