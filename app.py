from flask import Flask, render_template, request
import pickle
import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('notebooks/model.pkl', 'rb'))
vectorizer = pickle.load(open('notebooks/vectorizer.pkl', 'rb'))
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    lemmatizer = WordNetLemmatizer()
    #lowercase, remove punctuation, remove stopwords, 
    text = message.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    #tokenization
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words_without_stopwords = [i for i in words if i not in stop_words]

    #lemmatize
    lemmatizer_words =[lemmatizer.lemmatize(i) for i in words_without_stopwords]
    documents= " ".join(lemmatizer_words)
    vect_msg = vectorizer.transform([documents])

    #predict
    prediction = model.predict(vect_msg)[0]
    return render_template('result.html', prediction=prediction, message=message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=True)
