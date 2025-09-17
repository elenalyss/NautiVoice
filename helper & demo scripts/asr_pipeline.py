from transformers import pipeline
import torch
from pathlib import Path

# We set the condition to use GPU if available, otherwise will run on CPU.
device = 0 if torch.cuda.is_available() else -1

# We load the speech-to-text ASR model using the hugging face pipeline
# We use openAI's whisper-small model and set chunk_length to 30s for long files
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device,
    chunk_length_s=30  
)

def transcribe_audio(path_or_bytes):
    """
    This function takes a path to a WAV/MP3 file or raw audio bytes
    and returns the transcribed text.
    """
    result = asr(path_or_bytes)
    return result["text"]

if __name__ == "__main__": # We look for the audio file in the same folder as the script
    audio_path = Path(__file__).parent / "my_report.wav" 
    transcript = transcribe_audio(str(audio_path))  # we call the function and print the transcribed text
    print("Transcript:\n", transcript)
