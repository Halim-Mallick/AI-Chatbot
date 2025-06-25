from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import nltk
import numpy as np
import pickle
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# NLTK ডাউনলোড চেক
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()

# Model and utils load
model = load_model('chatbot_model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
data = pd.read_csv("ecommerce.csv")

def preprocess_text(text):
    # Lowercase, lemmatize, remove non-letters
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return ' '.join(tokens)

def predict_class(sentence):
    # Text preprocessing
    clean_sentence = preprocess_text(sentence)
    # Vectorization
    X = vectorizer.transform([clean_sentence]).toarray()
    # Predict
    prediction = model.predict(X)[0]
    intent_index = np.argmax(prediction)
    intent = label_encoder.inverse_transform([intent_index])[0]
    probability = prediction[intent_index]
    return {"intent": intent, "probability": str(probability)}

def get_response(intents_dict):
    tag = intents_dict['intent']
    # সাধারণত ডেটাতে 'intent' কলাম থাকে, আপনার ডেটাতে যদি 'tag' থাকে তাহলে নিচের লাইনে tag->intent বদলান
    responses = data[data['intent'] == tag]['response'].tolist()
    if responses:
        return random.choice(responses)
    return "Sorry, I couldn't understand that."

from flask import Flask, render_template
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

import os
import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))


@app.route('/chat', methods=['POST'])
def chat():
    data_json = request.get_json()
    message = data_json.get('message')
    if not message:
        return jsonify({"response": "No message received."})
    intent_dict = predict_class(message)
    res = get_response(intent_dict)
    return jsonify({"response": res})

import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
