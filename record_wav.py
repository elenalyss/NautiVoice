import sounddevice as sd
from scipy.io.wavfile import write
 
SR = 16000      # sample rate (καλό για ASR)
SECS = 15       # διάρκεια εγγραφής σε δευτ.
 
print("🎙️ Recording... μίλα τώρα (θα σταματήσει σε", SECS, "δευτερόλεπτα)")
audio = sd.rec(int(SECS * SR), samplerate=SR, channels=1, dtype="int16")
sd.wait()
write("my_report.wav", SR, audio)
print("✅ Αποθηκεύτηκε: my_report.wav")