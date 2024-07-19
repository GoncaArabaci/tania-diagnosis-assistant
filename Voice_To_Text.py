from vosk import Model, KaldiRecognizer
import wave
import json

def transcribe_vosk(audio_file):
    model = Model("model")  # Model dosyasının yolu
    wf = wave.open(audio_file, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            results.append(json.loads(result))

    final_result = rec.FinalResult()
    results.append(json.loads(final_result))
    transcription = " ".join([res['text'] for res in results if 'text' in res])
    return transcription

text = transcribe_vosk("output.wav")
print(text)
