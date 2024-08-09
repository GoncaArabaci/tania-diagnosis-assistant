import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# category_data.json dosyasını okuma
with open('category_data.json', 'r', encoding='utf-8') as f:
    category_data = json.load(f)

# Kategori sınıflandırma verilerini DataFrame'e dönüştürme
category_df = pd.DataFrame([(cat, text) for cat, texts in category_data.items() for text in texts], columns=['category', 'text'])

# Kategoriler için LabelEncoder
le_category = LabelEncoder()
category_df['category'] = le_category.fit_transform(category_df['category'])

# Kategori sınıflandırma için model oluşturma
X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(category_df['text'], category_df['category'], test_size=0.2, random_state=42)
category_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
category_model.fit(X_cat_train, y_cat_train)

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

# İnteraktif soru-cevap sistemi
def interactive_diagnosis():
    user_input = input("Lütfen şikayetinizi belirtin: ")
    
    # Kategoriyi tahmin et
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
        ans = input(q + " ")
        answers.append(ans)
    
    text = ' '.join(answers)
    prediction = model.predict([text])
    disease = le_disease.inverse_transform(prediction)
    print(f"Tahmin edilen hastalık: {disease[0]}")

interactive_diagnosis()
