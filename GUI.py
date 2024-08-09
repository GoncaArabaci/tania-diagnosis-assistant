import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout
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
        self.background_pixmap = QPixmap('background.png').scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setGeometry(0, 0, 600, 800)
        
        # TC Kimlik Numarası Girişi
        self.input_widget = QWidget(self)
        self.input_layout = QVBoxLayout()

        self.input_label = QLabel("TC Kimlik Numarası Girin:", self.input_widget)
        self.input_label.setFont(QFont('Arial', 14))
        self.input_label.setAlignment(Qt.AlignCenter)

        self.input_field = QLineEdit(self.input_widget)
        self.input_field.setPlaceholderText("TC Kimlik Numarası")
        self.input_field.setFixedSize(350, 40)  # Boyutları ayarla

        self.confirm_button = QPushButton("Onayla", self.input_widget)
        self.confirm_button.setFixedSize(100, 40)  # Boyutları ayarla
        self.confirm_button.clicked.connect(self.onInputFinished)
        
        # Merkezi hizalama
        self.input_layout.addStretch()  # Üstte boşluk bırakır
        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.confirm_button)

        self.input_layout.addStretch()  # Altta boşluk bırakır
        self.input_layout.setAlignment(Qt.AlignCenter)

        self.input_widget.setLayout(self.input_layout)
        self.input_widget.setGeometry(0, 0, 600, 800)
        self.input_widget.show()  # Başlangıçta göster

        # Kayıt Başlat/Duraklat Düğmesi (Yuvarlak buton) ve Sonuç Etiketi
        self.record_button = RoundButton(self)
        self.record_button.clicked.connect(self.toggleRecording)
        self.record_button.setGeometry((600-120)//2, (800-120)//2+45, 120, 120)  # Ekranın ortasına yerleştir
        self.record_button.hide()  # Başlangıçta gizle
        
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont('Arial', 18))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #333333;")
        self.result_label.setGeometry(0, 700, 600, 100)  # Alt merkez
        self.result_label.hide()  # Başlangıçta gizle
        
        # Hoşgeldiniz Mesajı ve Konuşma Talimatı
        self.welcome_label = QLabel("", self)
        self.welcome_label.setFont(QFont('Arial', 14))
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setGeometry(0, 315, 600, 50)  # Üst merkez
        self.welcome_label.hide()  # Başlangıçta gizle
        
        self.speak_instruction = QLabel("Konuşmak için tıklayınız", self)
        self.speak_instruction.setFont(QFont('Arial', 14))
        self.speak_instruction.setAlignment(Qt.AlignCenter)
        self.speak_instruction.setGeometry(0, 535, 600, 20)  # Alt merkez
        self.speak_instruction.hide()  # Başlangıçta gizle

    def onInputFinished(self):
        tc_number = self.input_field.text()
        if tc_number:
            print(f"Hastanın TC Kimlik Numarası: {tc_number}")
            self.input_widget.hide()
            self.record_button.show()
            self.result_label.show()
            
            # Örnek isim ve soyisim
            name = "<name>"
            surname = "<surname>"
            
            self.welcome_label.setText(f"Hoşgeldiniz {name} {surname}, şikayetiniz nedir?")
            self.welcome_label.show()
            self.speak_instruction.show()

    def toggleRecording(self):
        self.record_button.toggleRecording()

        if self.record_button.isRecording:
            self.startListening()
            self.welcome_label.hide()  # Hide the welcome message
            self.speak_instruction.hide()  # Hide the speak instruction
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

        print("Dinleniyor...")

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
            self.result_label.setText("Durduruldu")  # Durduruldu mesajını göster
        if self.p is not None:
            self.p.terminate()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
