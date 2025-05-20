# app.py
from flask import Flask, render_template, request, jsonify
import os, torch, json, numpy as np
from datetime import datetime
import joblib
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --- Paths ---
base_path = os.path.abspath(os.path.dirname(__file__))
processed_dir = os.path.join(base_path, "processed")
models_dir = os.path.join(base_path, "models")
statistics_file = os.path.join(processed_dir, "statistics.json")
recent_file = os.path.join(processed_dir, "recent_detections.json")

# --- Load Models ---
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
bert_model.load_state_dict(torch.load(os.path.join(models_dir, 'bert_model.pt'), map_location=torch.device('cpu')))
bert_model.eval()

lstm_model = load_model(os.path.join(processed_dir, "lstm_model.h5"))
rf_model = joblib.load(os.path.join(processed_dir, "RandomForest_model.pkl"))
xgb_model = joblib.load(os.path.join(processed_dir, "XGBoost_model.pkl"))
nb_model = joblib.load(os.path.join(processed_dir, "NaiveBayes_model.pkl"))
lr_model = joblib.load(os.path.join(processed_dir, "LogisticRegression_model.pkl"))
tfidf_vectorizer = joblib.load(os.path.join(processed_dir, "tfidf_vectorizer.pkl"))
lstm_tokenizer = joblib.load(os.path.join(processed_dir, "tokenizer.pkl"))

# --- Utilities ---
def save_statistics(human, bot):
    data = {"human": human, "bot": bot, "total": human + bot}
    with open(statistics_file, "w") as f:
        json.dump(data, f)

def load_statistics():
    if not os.path.exists(statistics_file):
        save_statistics(0, 0)
    with open(statistics_file, "r") as f:
        return json.load(f)

def save_recent_detection(prediction, confidence):
    detection = {
        "id": datetime.now().strftime('%H%M%S'),
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "time": datetime.now().strftime('%I:%M %p')
    }
    if os.path.exists(recent_file):
        with open(recent_file, "r") as f:
            recent = json.load(f)
    else:
        recent = []
    recent.insert(0, detection)
    recent = recent[:10]
    with open(recent_file, "w") as f:
        json.dump(recent, f)

def load_recent_detections():
    if os.path.exists(recent_file):
        with open(recent_file, "r") as f:
            return json.load(f)
    return []

# --- Prediction Functions ---
def bert_predict(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()
    label = "😎 Human" if pred_class == 0 else "🤖 Bot"
    return label, confidence * 100

def lstm_predict(text):
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = lstm_model.predict(padded, verbose=0)
    pred_class = int(pred[0][0] > 0.5)
    confidence = pred[0][0] if pred_class == 1 else (1 - pred[0][0])
    label = "🤖 Bot" if pred_class == 1 else "😎 Human"
    return label, confidence * 100

def traditional_predict(model, text):
    tfidf = tfidf_vectorizer.transform([text])
    pred = model.predict(tfidf)
    prob = model.predict_proba(tfidf)
    confidence = np.max(prob)
    label = "🤖 Bot" if pred[0] == 1 else "😎 Human"
    return label, confidence * 100

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/dashboard')
def dashboard():
    stats = load_statistics()
    recent = load_recent_detections()
    return render_template(
        'dashboard.html',
        humans_detected=stats['human'],
        bots_detected=stats['bot'],
        total_tweets=stats['total'],
        recent_detections=recent
    )

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    prediction_results = {}
    if request.method == 'POST':
        tweet = request.form['tweet']

        prediction_results['BERT'] = bert_predict(tweet)
        prediction_results['LSTM'] = lstm_predict(tweet)
        prediction_results['Random Forest'] = traditional_predict(rf_model, tweet)
        prediction_results['XGBoost'] = traditional_predict(xgb_model, tweet)
        prediction_results['Naive Bayes'] = traditional_predict(nb_model, tweet)
        prediction_results['Logistic Regression'] = traditional_predict(lr_model, tweet)

        # Final Decision Logic
        votes = []
        confidences = {"human": [], "bot": []}
        for _, (label, conf) in prediction_results.items():
            if "Human" in label:
                votes.append("human")
                confidences["human"].append(conf)
            else:
                votes.append("bot")
                confidences["bot"].append(conf)

        final_label = max(set(votes), key=votes.count)
        if votes.count('human') == votes.count('bot'):
            avg_human = np.mean(confidences['human']) if confidences['human'] else 0
            avg_bot = np.mean(confidences['bot']) if confidences['bot'] else 0
            final_label = "human" if avg_human > avg_bot else "bot"

        final_conf = np.mean(confidences[final_label]) if confidences[final_label] else 0
        final_emoji = "😎 Human" if final_label == "human" else "🤖 Bot"
        prediction_results['🚀 Final Decision'] = (final_emoji, final_conf)

        stats = load_statistics()
        stats['human' if final_label == 'human' else 'bot'] += 1
        stats['total'] = stats['human'] + stats['bot']
        save_statistics(stats['human'], stats['bot'])

        save_recent_detection(final_emoji, final_conf)

    return render_template('detector.html', prediction_results=prediction_results)

if __name__ == "__main__":
    app.run(debug=True)
