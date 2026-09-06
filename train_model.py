import pandas as pd
import numpy as np
import re
import pickle
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

# 1. Data Load
print("📂 Data load ho raha hai...")
fake = pd.read_csv('dataset/Fake.csv')
real = pd.read_csv('dataset/True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real]).reset_index(drop=True)
print(f"✅ Total articles: {len(df)}")

# 2. Clean Text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("🧹 Text clean ho raha hai...")
df['clean_text'] = df['text'].apply(clean_text)

# 3. TF-IDF
print("🔢 Features bana rahe hain...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# 4. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Train 3 Models
print("🤖 Models train ho rahe hain...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

results = {}
best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"✅ {name}: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

# 6. Save Model
os.makedirs('model', exist_ok=True)
pickle.dump(best_model, open('model/model.pkl', 'wb'))
pickle.dump(tfidf, open('model/tfidf.pkl', 'wb'))
print("💾 Model save ho gaya!")

# 7. Accuracy Chart
plt.figure(figsize=(8,5))
plt.bar(results.keys(), [v*100 for v in results.values()], 
        color=['#2ecc71','#3498db','#e74c3c'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy %')
plt.ylim(80, 100)
plt.savefig('model/accuracy_chart.png')
print("📊 Chart save ho gaya!")
print("\n✅ Training complete!") 