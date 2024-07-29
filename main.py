import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Verileri çekme
urls = [
    "https://www.nhsinform.scot/illnesses-and-conditions/brain-nerves-and-spinal-cord/headaches/",
    "https://www.nhs.uk/conditions/headaches/"
]

# Metinleri toplama ve temizleme fonksiyonu
def fetch_and_clean_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

# Tüm metinleri birleştirme
all_texts = ' '.join([fetch_and_clean_text(url) for url in urls])

# Küçük harfe çevirme, noktalama işaretlerini kaldırma ve tokenizasyon
all_texts = all_texts.lower()
translator = str.maketrans('', '', string.punctuation)
all_texts = all_texts.translate(translator)
tokens = word_tokenize(all_texts)

# Stop kelimeleri kaldırma
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]

print(tokens[:100])  # İlk 100 tokeni yazdır

# Veriyi baş ağrısı türlerine göre etiketleme
texts = [
    "I have a severe headache and nausea",  # Migren
    "My head feels like it is being squeezed",  # Gerilim tipi baş ağrısı
    "I feel a throbbing pain on one side of my head",  # Migren
    "I have a dull, aching pain around my forehead",  # Gerilim tipi baş ağrısı
    "I experience sensitivity to light and sound",  # Migren
    "I feel pressure around my temples",  # Gerilim tipi baş ağrısı
    "I have a headache accompanied by a stiff neck",  # Gerilim tipi baş ağrısı
    "I have a headache and feel nauseous",  # Migren
    "I feel like my head is in a vice",  # Gerilim tipi baş ağrısı
    "I experience intense pain behind one eye",  # Küme baş ağrısı
    "I have a burning sensation in my head",  # Sinüzit baş ağrısı
    "My headache worsens with physical activity",  # Migren
    "I have a headache that comes with a fever",  # Sinüzit baş ağrısı
    "I feel pain around my eyes and forehead",  # Sinüzit baş ağrısı
    "I have a headache that starts suddenly",  # Küme baş ağrısı
    "I experience a stabbing pain in my head",  # Küme baş ağrısı
    "I have a constant, dull ache in my head",  # Gerilim tipi baş ağrısı
    "I feel a pulsating pain in my head",  # Migren
    "I have a headache with a stuffy nose",  # Sinüzit baş ağrısı
    "I feel like my head is being compressed",  # Gerilim tipi baş ağrısı
    "I have a headache with blurred vision",  # Migren
    "I experience a sharp pain on one side of my head",  # Küme baş ağrısı
    "I have a headache with pain around my sinuses",  # Sinüzit baş ağrısı
    "I feel a tight band around my head",  # Gerilim tipi baş ağrısı
    "I have a headache with eye redness and tearing",  # Küme baş ağrısı
    "I feel like my head is about to explode",  # Migren
    "I have a pounding headache that makes it hard to concentrate",  # Migren
    "I have a headache with dizziness",  # Migren
    "I experience a headache that lasts for days",  # Migren
    "I have a headache with pressure behind my eyes",  # Sinüzit baş ağrısı
    "My headache gets worse when I lie down",  # Sinüzit baş ağrısı
    "I feel a tightness around my forehead",  # Gerilim tipi baş ağrısı
    "I have a headache with muscle tension in my neck and shoulders",  # Gerilim tipi baş ağrısı
    "I experience a sudden, severe headache",  # Küme baş ağrısı
    "I have a headache with nausea and vomiting",  # Migren
    "I feel a band-like pain around my head",  # Gerilim tipi baş ağrısı
    "I have a headache with fatigue",  # Migren
    "I experience a headache that is worse in the morning",  # Sinüzit baş ağrısı
    "I have a headache with a runny nose",  # Sinüzit baş ağrısı
    "I feel like my head is under constant pressure",  # Gerilim tipi baş ağrısı
    "I have a headache with a metallic taste in my mouth",  # Sinüzit baş ağrısı
    "I experience a headache that wakes me up at night",  # Küme baş ağrısı
    "I have a headache with a feeling of tightness in my scalp",  # Gerilim tipi baş ağrısı
    "I feel a throbbing pain in my temples",  # Migren
    "I have a headache that is triggered by stress",  # Gerilim tipi baş ağrısı
    "I experience a headache with double vision",  # Migren
    "I have a headache with ear pressure",  # Sinüzit baş ağrısı
    "I feel a sharp pain that comes and goes",  # Küme baş ağrısı
    "I have a headache that is triggered by changes in weather",  # Migren
    "I experience a headache with a sensation of fullness in my head",  # Sinüzit baş ağrısı
    "I have a headache that improves with relaxation",  # Gerilim tipi baş ağrısı
    "I feel a headache that starts in the back of my head and moves forward",  # Gerilim tipi baş ağrısı
    "I have a headache with pulsating pain",  # Migren
    "I experience a headache with nasal congestion",  # Sinüzit baş ağrısı
]

labels = [
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "cluster",  # Küme baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "cluster",  # Küme baş ağrısı
    "cluster",  # Küme baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "cluster",  # Küme baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "cluster",  # Küme baş ağrısı
    "migraine",  # Migren
    "migraine",  # Migren
    "migraine",  # Migren
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "cluster",  # Küme baş ağrısı
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "sinus",  # Sinüzit baş ağrısı
    "cluster",  # Küme baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "cluster",  # Küme baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "tension",  # Gerilim tipi baş ağrısı
    "migraine",  # Migren
    "sinus",  # Sinüzit baş ağrısı
]

# URL'den çekilen veriyi de listeye eklemek için uygun hale getirin
# Örneğin, çekilen veriden bazı cümleler seçip etiketleyebilirsiniz
# Bu kısmı elle eklemek gerekiyor çünkü otomatik etiketleme yapmadık

# Veriyi genişletmek için daha fazla metin ve etiket ekleyin

# Özellik çıkarma (bag of words)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Veri setini ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model eğitimi
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Model değerlendirme
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Yeni cümleler üzerinde tahmin yapma
new_texts = [
    "I have a headache with nausea and sensitivity to light",  # Beklenen: Migren
    "I feel a constant pressure around my head",  # Beklenen: Gerilim tipi baş ağrısı
    "I have a sharp pain behind my eyes and a stuffy nose",  # Beklenen: Sinüzit baş ağrısı
    "My headache comes suddenly and is very severe",  # Beklenen: Küme baş ağrısı
]

new_X = vectorizer.transform(new_texts)
new_preds = clf.predict(new_X)

for text, pred in zip(new_texts, new_preds):
    print(f"Text: {text}\nPredicted Category: {pred}\n")
