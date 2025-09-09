# NautiVoice – Voice-enabled Maritime Incident Reporting

## Overview
NautiVoice is a prototype application designed to support **voice-based reporting of maritime incidents**.  
It integrates:
- **Automatic Speech Recognition (ASR)** using OpenAI Whisper  
- **Incident Classification** (fine-tuned BERT models for *Category* & *Severity*)  
- **FastAPI backend** serving the models as an API  
- A simple **web frontend** for recording incidents and viewing predictions  

The system allows a user to report an incident through voice input. The audio is transcribed into text, classified into the correct **incident category** and **severity level**, and finally displayed in a dashboard-style interface.

---

## Requirements
- **Python 3.10+**
- **pip** (Python package manager)
- **git** (to clone repository)
- **ffmpeg** (for audio processing – must be installed separately)
- Recommended: use a **virtual environment** (venv/conda)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nautivoice.git
   cd nautivoice
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on macOS/Linux
   venv\Scripts\activate      # on Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install ffmpeg**
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

5. **Download / Place fine-tuned models**
   - Place the following model folders inside the project root:
     - `nautivoice-category-augm`
     - `nautivoice-severity-augm-2`

---

## Running the Application

1. **Start the FastAPI backend**
   ```bash
   uvicorn app:app --reload
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
├── app.py                     # FastAPI backend (ASR + Classification API)
├── asr_pipeline.py            # Standalone ASR pipeline (Whisper)
├── train_category_augm.py     # Training script for Category model
├── evaluate_category_augm.py  # Evaluation script for Category model
├── train_severity_augm_2.py     # Training script for Severity model
├── evaluate_severity_augm2.py  # Evaluation script for Severity model
├── run_pipeline.py             # Example pipeline runner
├── requirements.txt            # Dependencies
├── static/                     # Frontend (index.html, JS, CSS)
├── nautivoice-category-augm/   # Fine-tuned Category model
└── nautivoice-severity-augm-2/ # Fine-tuned Severity model
```

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
This project was developed as part of the **Machine Learning mini-project (Spring 2025)** for educational purposes.
