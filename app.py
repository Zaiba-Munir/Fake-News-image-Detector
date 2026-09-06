from dotenv import load_dotenv
load_dotenv()
from flask import Flask, render_template, request
from groq import Groq
from serpapi import GoogleSearch
import json
import re
import requests
import os
from datetime import date
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np

app = Flask(__name__)

# API Keys
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

# AI-Generated Image Detection model load karna (v2 — behtar accuracy wala)
image_model = load_model('model/ai_image_detector_v2.h5')


def is_url(text):
    return re.match(r'^https?://', text.strip()) is not None


def extract_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text.strip()
    except Exception:
        return ""


def search_news(text):
    try:
        search = GoogleSearch({
            "q": text,
            "api_key": SERPAPI_KEY,
            "num": 5
        })
        results = search.get_dict()
        print("DEBUG SerpAPI raw response:", results)
        organic = results.get("organic_results", [])

        sources = []
        for r in organic[:3]:
            sources.append(r.get("title", ""))

        return sources
    except Exception as e:
        print("DEBUG SerpAPI Exception:", e)
        return []


def _ask_groq_for_json(prompt):
    """Groq ko call karke JSON nikalne ki koshish karta hai. Success pe dict, fail pe None return karta hai."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You always reply with a single valid JSON object and nothing else — no markdown fences, no extra commentary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=600
    )

    raw_content = response.choices[0].message.content
    print(f"DEBUG: Raw Groq response: {raw_content}")

    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    # Sirf pehla { se lekar aakhri } tak nikaalein, taake extra text ya galat quotes na aayein
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def check_news_with_groq(text, sources):
    sources_text = "\n".join(sources) if sources else "No sources found"
    today = date.today().strftime("%B %d, %Y")

    prompt = f"""You are a fake news detector. Today's real date is {today}. Use this to judge whether events mentioned are past, present, or genuinely future/impossible.

Analyze the following news or message.

Text to analyze: "{text}"

Google Search Results:
{sources_text}

Consider these factors:
- Does Google search confirm this news?
- Is it factually accurate?
- Does it contain sensational or misleading language?
- Is it a known fake/viral hoax?
- IMPORTANT: Only flag an event as "impossible/future" if it is dated AFTER {today}. Do not assume your own training knowledge is up to date — trust the given date and search results over your memory.

Reply ONLY with a JSON object in this exact format, nothing else, no markdown, no extra words. Keep the "reason" to one short sentence so the whole reply stays short:
{{"label": "REAL" or "FAKE", "confidence": number between 50 and 99, "reason": "one line explanation in same language as input"}}"""

    # Pehli koshish
    result = _ask_groq_for_json(prompt)

    # Agar pehli dafa JSON parse na ho saka, to ek dafa dobara koshish karein
    if result is None:
        print("DEBUG: Pehli koshish fail ho gayi, dobara try kar rahe hain")
        result = _ask_groq_for_json(prompt)

    # Dono koshishon ke baad bhi fail ho to fallback use karein
    if result is None:
        print("DEBUG: Dono koshishein fail ho gayin, fallback use kar rahe hain")
        result = {
            "label": "FAKE",
            "confidence": 60,
            "reason": "AI response ko sahi se parse nahi kiya ja saka. Barah-e-karam dobara try karein."
        }

    return result


def predict_image(img_path):
    # V2 model 128x128 size pe train hua tha
    img = keras_image.load_img(img_path, target_size=(128, 128))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = image_model.predict(img_array)[0][0]
    print(f"DEBUG: Raw prediction score = {prediction}")

    # V2 dataset mein label 0 = REAL (human), label 1 = AI-GENERATED
    # Isliye high score (>0.5) = AI-GENERATED, low score = REAL
    if prediction > 0.5:
        label = "AI-GENERATED"
        confidence = prediction * 100
    else:
        label = "REAL"
        confidence = (1 - prediction) * 100

    return label, round(confidence, 2)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_input = request.form['news']

    # Agar user ne link diya hai to pehle article nikaalein
    if is_url(news_input):
        news = extract_article_text(news_input)
        if not news:
            return render_template('index.html',
                                 result="⚠️ ERROR",
                                 confidence="0%",
                                 color="red",
                                 news=news_input,
                                 reason="Is link se article text nahi nikal saka. Barah-e-karam news text seedha paste karein.")
    else:
        news = news_input

    # Pehle Google search — sirf pehle 200 characters se search karein (behtar results ke liye)
    search_query = news[:200]
    sources = search_news(search_query)

    # Phir Groq analyze
    result = check_news_with_groq(news[:3000], sources)

    label_text = result.get('label', 'UNKNOWN')
    confidence = result.get('confidence', 0)
    reason = result.get('reason', '')

    if label_text == "REAL":
        label = "✅ REAL NEWS"
        color = "green"
    else:
        label = "❌ FAKE NEWS"
        color = "red"

    return render_template('index.html',
                         result=label,
                         confidence=f"{confidence}%",
                         color=color,
                         news=news,
                         reason=reason)


@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return render_template('index.html', img_error="Koi image select nahi hui.")

    img_file = request.files['image']
    img_path = 'uploaded_temp.jpg'
    img_file.save(img_path)

    label, confidence = predict_image(img_path)

    os.remove(img_path)

    return render_template('index.html', img_result=label, img_confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)