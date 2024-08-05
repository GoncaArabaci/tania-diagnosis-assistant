import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK veri setlerini indir
nltk.download('punkt')
nltk.download('stopwords')

# Türkçe stopwords kullanımı
stop_words = set(stopwords.words('turkish'))

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    # Tokenizasyon (kelime parçalama)
    words = word_tokenize(text)
    # Stopwords kaldırma
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Load the CSV file
file_path = 'extended_symptoms_data.csv'
df = pd.read_csv(file_path)

# Load texts and labels from the dataframe
texts = df['text'].tolist()
labels = df['label'].tolist()

# Metinleri ön işleme tabi tut
processed_texts = [preprocess_text(text) for text in texts]

# Feature extraction (bag of words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Patient history data
patients = {
    "koray": {"age": 20, "past_conditions": ["sinüzit"]},
    "ayşe": {"age": 45, "past_conditions": ["migren"]},
    "mehmet": {"age": 30, "past_conditions": ["gerilme tipi baş ağrısı"]},
    "ali": {"age": 50, "past_conditions": ["küme tipi baş ağrısı"]},
}

# Voice recognition and prediction
model_path = "vosk-model-small-tr-0.3"

if not os.path.exists(model_path):
    print(f"model bulunamadı:{model_path}")
    exit(1)

model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Dinleniyor...")

patient_name = None

while True:
    data = stream.read(4000, exception_on_overflow=False)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        speech_text = result.get("text", "")
        if speech_text:
            print(f"İsminizi Söyleyin: {speech_text}")
            if not patient_name:
                # İlk alınan metni hasta ismi olarak kabul et
                patient_name = speech_text.lower()
                patient_info = patients.get(patient_name)
                if patient_info:
                    print(f"Hasta Adı: {patient_name.capitalize()}")
                    print(f"Yaş: {patient_info['age']}")
                    print(f"Geçmiş Tanılar: {', '.join(patient_info['past_conditions'])}")
                    print(f"Şikayetiniz Nedir:")
                else:
                    print(f"Hastayla ilgili bilgi bulunamadı lütfen tekrar adınızı söyleyin: {patient_name}")
                    patient_name = None  # Hastayı bulamazsa tekrar isim bekle
            else:
                # Hasta ismi belirlendikten sonra alınan metin üzerinden tahmin yap
                processed_speech_text = preprocess_text(speech_text)
                new_X = vectorizer.transform([processed_speech_text])
                pred = clf.predict(new_X)[0]
                print(f"Tahmin: {pred}")
