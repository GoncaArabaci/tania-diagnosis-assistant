import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPolygon, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint, QTimer
import os
import vosk
from vosk import Model, KaldiRecognizer
import pyaudio
import json
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# İlk olarak mevcut kodunuzu yukarıda verdiğiniz şekliyle yükleyelim.

# Ses tanıma sonucu burada tutulacak
recognized_text = ""

# Kategoriyi sınıflandırmak için verilerinizi ve modeli hazırlayalım
with open('category_data.json', 'r', encoding='utf-8') as f:
    category_data = json.load(f)

category_df = pd.DataFrame([(cat, text) for cat, texts in category_data.items() for text in texts], columns=['category', 'text'])
le_category = LabelEncoder()
category_df['category'] = le_category.fit_transform(category_df['category'])
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(category_df['text'], category_df['category'], test_size=0.2, random_state=42)
category_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
category_model.fit(X_cat_train, y_cat_train)

# Hastalık verilerini hazırlamak ve modelleri eğitmek için kod
with open('datasets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def prepare_data(data, key):
    df = pd.DataFrame(data[key])
    df['text'] = df['answers'].apply(lambda x: ' '.join(x))
    le_disease = LabelEncoder()
    df['disease'] = le_disease.fit_transform(df['disease'])
    return df, le_disease

datasets = {
    'göz': prepare_data(data, 'göz'),
    'üriner': prepare_data(data, 'üriner'),
    'boğaz': prepare_data(data, 'boğaz'),
    'omuz': prepare_data(data, 'omuz'),
    'nefes_darligi': prepare_data(data, 'nefes_darligi'),
    'ayak_bilek': prepare_data(data, 'ayak_bilek'),
    'baş_ağrısı': prepare_data(data, 'baş_ağrısı'),
    'ellerde_uyusma_karincalanma': prepare_data(data, 'ellerde_uyusma_karincalanma'),
    'mide_bulantisi_kusma': prepare_data(data, 'mide_bulantisi_kusma'),
    'yutma_zorlugu': prepare_data(data, 'yutma_zorlugu'),
    'gögüs_agrisi': prepare_data(data, 'gögüs_agrisi'),
    'karin_agrisi': prepare_data(data, 'karin_agrisi')
}

def get_features_and_labels(df):
    return df['text'], df['disease']

splits = {}
for name, (df, le) in datasets.items():
    X, y = get_features_and_labels(df)
    splits[name] = train_test_split(X, y, test_size=0.2, random_state=42)

def create_and_train_model(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)
    return model

models = {}
le_encoders = {}
for name, (X_train, X_test, y_train, y_test) in splits.items():
    models[name] = create_and_train_model(X_train, y_train)
    le_encoders[name] = datasets[name][1]

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Etkileşimli tanı fonksiyonunu güncelleyelim
def interactive_diagnosis(user_input):
    global recognized_text
    if not user_input:
        user_input = recognized_text  # Eğer kullanıcı inputu boşsa, ses tanımadan gelen sonucu kullan
    
    cat_prediction = category_model.predict([user_input])
    category = le_category.inverse_transform(cat_prediction)[0]
    
    if category in models:
        model = models[category]
        questions = data[category][0]['questions']
        le_disease = le_encoders[category]
    else:
        print("Geçersiz alan. Lütfen doğru bir seçenek giriniz.")
        return
    
    answers = []
    for q in questions:
        recognized_text = ""
        main_window.result_label.setText(q)
        main_window.speak_instruction.setText("Cevap verin...")
        while not recognized_text:  # Cevabın alınmasını bekle
            app.processEvents()

        answers.append(recognized_text)
    
    text = ' '.join(answers)
    prediction = model.predict([text])
    disease = le_disease.inverse_transform(prediction)
    main_window.result_label.setText(f"Tahmin edilen hastalık: {disease[0]}")
    print(f"Tahmin edilen hastalık: {disease[0]}")

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
            painter.drawRect(40, 30, 20, 60)
            painter.drawRect(60, 30, 20, 60)
        else:
            triangle = QPolygon([
                QPoint(45, 30),
                QPoint(75, 60),
                QPoint(45, 90)
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
        self.setGeometry(100, 100, 600, 800)
        
        self.background_label = QLabel(self)
        self.background_pixmap = QPixmap('background.png').scaled(self.width(), self.height(), Qt.KeepAspectRatioByExpanding)
        self.background_label.setPixmap(self.background_pixmap)
        self.background_label.setGeometry(0, 0, 600, 800)
        
        self.input_widget = QWidget(self)
        self.input_layout = QVBoxLayout()

        self.input_label = QLabel("TC Kimlik Numarası Girin:", self.input_widget)
        self.input_label.setFont(QFont('Arial', 14))
        self.input_label.setAlignment(Qt.AlignCenter)

        self.input_field = QLineEdit(self.input_widget)
        self.input_field.setPlaceholderText("TC Kimlik Numarası")
        self.input_field.setFixedSize(350, 40)

        self.confirm_button = QPushButton("Onayla", self.input_widget)
        self.confirm_button.setFixedSize(100, 40)
        self.confirm_button.clicked.connect(self.onInputFinished)
        
        self.input_layout.addStretch()
        self.input_layout.addWidget(self.input_label)
        self.input_layout.addWidget(self.input_field)
        self.input_layout.addWidget(self.confirm_button)
        self.input_layout.addStretch()
        self.input_layout.setAlignment(Qt.AlignCenter)

        self.input_widget.setLayout(self.input_layout)
        self.input_widget.setGeometry(0, 0, 600, 800)
        self.input_widget.show()

        self.record_button = RoundButton(self)
        self.record_button.clicked.connect(self.toggleRecording)
        self.record_button.setGeometry((600-120)//2, (800-120)//2+45, 120, 120)
        self.record_button.hide()
        
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont('Arial', min(18, int(0.04 * self.height()))))  # Dinamik font boyutu
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #333333;")
        self.result_label.setGeometry(0, 600, 600, 100)
        self.result_label.setWordWrap(True)  # Metnin sarmalanmasını sağlar
        self.result_label.hide()

        
        self.welcome_label = QLabel("", self)
        self.welcome_label.setFont(QFont('Arial', 14))
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.welcome_label.setGeometry(0, 315, 600, 50)
        self.welcome_label.hide()
        
        self.speak_instruction = QLabel("Konuşmak için tıklayınız", self)
        self.speak_instruction.setFont(QFont('Arial', 14))
        self.speak_instruction.setAlignment(Qt.AlignCenter)
        self.speak_instruction.setGeometry(0, 535, 600, 20)
        self.speak_instruction.hide()

    def onInputFinished(self):
        tc_number = self.input_field.text()
        if tc_number:
            print(f"Hastanın TC Kimlik Numarası: {tc_number}")
            self.input_widget.hide()
            self.record_button.show()
            self.result_label.show()
            
            name = "<name>"
            surname = "<surname>"
            
            self.welcome_label.setText(f"Hoşgeldiniz {name} {surname}, şikayetiniz nedir?")
            self.welcome_label.show()
            self.speak_instruction.show()

    def toggleRecording(self):
        self.record_button.toggleRecording()

        if self.record_button.isRecording:
            self.startListening()
            self.welcome_label.hide()
            self.speak_instruction.hide()
        else:
            self.stopListening()
            interactive_diagnosis("")  # Sesli girdiyi kullanarak tanı yap

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
        self.timer.start(100)

    def processAudio(self):
        global recognized_text
        if self.record_button.isRecording:
            data = self.stream.read(4000, exception_on_overflow=False)
            if len(data) == 0:
                return

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                recognized_text = result.get("text", "")
                self.result_label.setText(recognized_text)
                print(recognized_text)

    def stopListening(self):
        if self.stream is not None:
            self.timer.stop()
            self.stream.stop_stream()
            self.stream.close()
            self.result_label.setText("Durduruldu")
        if self.p is not None:
            self.p.terminate()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
