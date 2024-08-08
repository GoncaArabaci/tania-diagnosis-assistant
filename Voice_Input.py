import sounddevice as sd
import numpy as np
import time
import os
import vosk
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Ses kaydını kontrol etmek için bazı parametreler
DURATION = 5  # Ses kaydı ne kadar süre sessiz kalırsa dursun (saniye)
THRESHOLD = 0.01  # Sessizlik tespiti için eşik değeri

# Modeli bir kez yüklemek
model_path = "vosk-model-small-tr-0.3"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

# Silence timer'ı global olarak tanımlayın
silence_start = None

def callback(indata, frames, c_time, status):
    global silence_start
    volume_norm = np.linalg.norm(indata) * 10
    
    if volume_norm > THRESHOLD:
        silence_start = time.time()  # Sessizlik zamanlayıcısını sıfırla
        print("Listening...")

        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                print(result.get("text", ""))
            else:
                partial_result = json.loads(recognizer.PartialResult())
                print(partial_result.get("partial", ""))

    else:
        print("Sessizlik...")

def listen_and_stop_on_silence(duration):
    global silence_start
    print("Program başlatıldı, ses dinleniyor...")
    silence_start = time.time()  # Zamanlayıcıyı başlat
    with sd.InputStream(callback=callback):
        while True:
            current_time = time.time()
            if current_time - silence_start > duration:
                print("Uzun süredir ses yok, kayıt durduruluyor.")
                break
            time.sleep(0.1)  # CPU kullanımını düşürmek için kısa bir bekleme

if __name__ == "__main__":
    listen_and_stop_on_silence(DURATION)
