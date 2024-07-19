import pyaudio
import wave
import keyboard

def record_audio(filename, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    
    frames = []
    print("Recording... Press ESC to stop.")
    while True:
        if keyboard.is_pressed('esc'):
            print("Recording stopped.")
            break
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

record_audio("output.wav")
