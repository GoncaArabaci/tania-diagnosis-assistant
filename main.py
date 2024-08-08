import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# JSON dosyasını okuma
with open('datasets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Verileri DataFrame'e dönüştürme ve hazırlanma
def prepare_data(data, key):
    df = pd.DataFrame(data[key])
    df['text'] = df['answers'].apply(lambda x: ' '.join(x))
    le_disease = LabelEncoder()
    df['disease'] = le_disease.fit_transform(df['disease'])
    return df, le_disease

# Tüm verileri ve LabelEncoder'ları bir sözlükte saklayalım
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

# Özellikler ve hedef değişken
def get_features_and_labels(df):
    return df['text'], df['disease']

splits = {}
for name, (df, le) in datasets.items():
    X, y = get_features_and_labels(df)
    splits[name] = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF ve lojistik regresyon modeli oluşturma ve eğitme
def create_and_train_model(X_train, y_train):
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)
    return model

models = {}
le_encoders = {}
for name, (X_train, X_test, y_train, y_test) in splits.items():
    models[name] = create_and_train_model(X_train, y_train)
    le_encoders[name] = datasets[name][1]

# Modellerin doğruluğunu test etme
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

#for name, (X_train, X_test, y_train, y_test) in splits.items():
#    print(f"{name.capitalize()} modeli doğruluğu: {evaluate_model(models[name], X_test, y_test)}")

# İnteraktif soru-cevap sistemi
def interactive_diagnosis():
    area = input("Belirtiniz gözle mi, üriner sistemle mi, boğazla mı, omuzla mı, nefes darlığıyla mı, ayak/bilek ile mi, baş ağrısı ile mi, ellerde uyuşma/karıncalanma ile mi, mide bulantısı/kusma ile mi, yutma zorluğu ile mi, göğüs ağrısı ile mi, karın ağrısı ile mi ilgili? (göz/üriner/boğaz/omuz/nefes/ayak_bilek/baş_ağrısı/ellerde_uyusma_karincalanma/mide_bulantisi_kusma/yutma_zorlugu/gögüs_agrisi/karin_agrisi) ")
    
    if area.lower() in models:
        model = models[area.lower()]
        questions = data[area.lower()][0]['questions']
        le_disease = le_encoders[area.lower()]
    else:
        print("Geçersiz alan. Lütfen doğru bir seçenek giriniz.")
        return
    
    answers = []
    for q in questions:
        ans = input(q + " ")
        answers.append(ans)
    
    text = ' '.join(answers)
    prediction = model.predict([text])
    disease = le_disease.inverse_transform(prediction)
    print(f"Tahmin edilen hastalık: {disease[0]}")

interactive_diagnosis()
