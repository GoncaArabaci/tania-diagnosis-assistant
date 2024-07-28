import os
import vosk
from vosk import Model, KaldiRecognizer
import pyaudio
import json

# Set path to the new model
model_path = "vosk-model-small-tr-0.3"

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Listening...")

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if len(data) == 0:
        break

    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        print(result.get("text", ""))
