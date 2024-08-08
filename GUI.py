import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPolygon, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint, QTimer

import os
import vosk
from vosk import Model, KaldiRecognizer
import pyaudio
import json

class RoundButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)  # Boyutlarını belirler
        self.setStyleSheet("border-radius: 60px; background-color: #35383e;")
        self.isRecording = False

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(Qt.white)))

        if self.isRecording:
            # Durdurma simgesi (iki dikdörtgen)
            painter.drawRect(40, 30, 20, 60)
            painter.drawRect(60, 30, 20, 60)
        else:
            # Başlatma simgesi (üçgen)
            triangle = QPolygon([
                QPoint(45, 30),  # Sol alt köşe
                QPoint(75, 60),  # Sağ köşe
                QPoint(45, 90)   # Sol üst köşe
            ])
            painter.drawPolygon(triangle)

    def toggleRecording(self):
        self.isRecording = not self.isRecording
        self.update()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.stream = None
        self.timer = QTimer(self)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("TanIA")
        self.setGeometry(100, 100, 600, 800)  # Telefon ekranı boyutları
        
        # Arka Plan Resmi
        self.background_label = QLabel(self)
        self.background_pixmap = QPixmap('background (1).png').scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setGeometry(0, 0, 600, 800)
        
        # Kayıt Başlat/Duraklat Düğmesi (Yuvarlak buton)
        self.record_button = RoundButton(self)
        self.record_button.clicked.connect(self.toggleRecording)
        self.record_button.setGeometry((600-120)//2, (800-85)//2, 120, 120)  # Ekranın ortasına yerleştir
        
        # Sonuç Etiketi
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont('Arial', 18))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #333333;")
        self.result_label.setGeometry(0, 700, 600, 100)  # Alt merkez

    def toggleRecording(self):
        self.record_button.toggleRecording()

        if self.record_button.isRecording:
            self.startListening()
        else:
            self.stopListening()

    def startListening(self):
        model_path = "vosk-model-small-tr-0.3"

        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            exit(1)

        model = Model(model_path)
        self.recognizer = KaldiRecognizer(model, 16000)

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        self.stream.start_stream()

        print("Listening...")

        self.timer.timeout.connect(self.processAudio)
        self.timer.start(100)  # 100 ms aralıklarla çağır

    def processAudio(self):
        if self.record_button.isRecording:
            data = self.stream.read(4000, exception_on_overflow=False)
            if len(data) == 0:
                return

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                self.result_label.setText(result.get("text", ""))
                print(result.get("text", ""))
            self.result_label.setText("Başlatıldı")  # Durduruldu mesajını göster

    def stopListening(self):
        if self.stream is not None:
            self.timer.stop()
            self.stream.stop_stream()
            self.stream.close()
        if self.p is not None:
            self.p.terminate()
        self.result_label.setText("Durduruldu")  # Durduruldu mesajını göster

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
