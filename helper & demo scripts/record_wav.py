import sounddevice as sd
from scipy.io.wavfile import write
#We set the sample rate - 16kHz is recommended for speech recognition
SR = 16000    
SECS = 15 # We set the duration of the recording in seconds     
#We  start recording from the microphone 
print("Recording... speak now (it will stop in", SECS, "seconds)")
audio = sd.rec(int(SECS * SR), samplerate=SR, channels=1, dtype="int16")
sd.wait()
write("my_report.wav", SR, audio)
print("Saved as: my_report.wav")
