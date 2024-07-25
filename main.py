import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# NLTK veri indirme
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Noktalama işaretlerini kaldırma
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    # Küçük harfe çevirme
    text = text.lower()
    # Kelime tokenizasyonu
    words = word_tokenize(text)
    # Durak kelimeleri kaldırma
    stop_words = set(stopwords.words('turkish'))
    words = [word for word in words if word not in stop_words]
    return words

def normalize_word(word):
    # Belirtilerle eşleşme oranını artırmak için ekleri temizleme
    return re.sub(r'(ım|im|um|üm|ın|in|un|ün|ı|i|u|ü|m|n|r|ğüm)$', '', word)

def recognize_symptoms(text):
    symptoms = ["baş ağrısı", "öksürük", "ateş", "bulantı", "kusma"]  # Bu liste genişletilebilir
    words = preprocess_text(text)
    normalized_words = [normalize_word(word) for word in words]
    recognized_symptoms = [symptom for symptom in symptoms if any(word in symptom for word in normalized_words)]
    return recognized_symptoms

def map_symptoms_to_departments(symptoms):
    symptom_department_map = {
        "baş ağrısı": "Nöroloji",
        "öksürük": "Göğüs Hastalıkları",
        "ateş": "Enfeksiyon Hastalıkları",
        "bulantı": "Gastroenteroloji",
        "kusma": "Gastroenteroloji"
    }
    departments = list(set([symptom_department_map[symptom] for symptom in symptoms if symptom in symptom_department_map]))
    return departments

def summarize_symptoms(symptoms):
    summary = "Hasta şu belirtileri gösteriyor: " + ", ".join(symptoms) + "."
    return summary

# Örnek kullanım
text = "Benim baş ağrım var ve bununla öksürüğüm de var.ayrıca bulantım da var"

# 3. Symptom Recognition
recognized_symptoms = recognize_symptoms(text)

# 4. Symptom-Department Mapping
mapped_departments = map_symptoms_to_departments(recognized_symptoms)

# 5. Summarization
summary = summarize_symptoms(recognized_symptoms)

print("Belirlenen Belirtiler:", recognized_symptoms)
print("İlgili Bölümler:", mapped_departments)
print("Özet:", summary)
