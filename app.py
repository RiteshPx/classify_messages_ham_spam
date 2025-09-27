# from flask import Flask, render_template, request
# import pickle
# import os
# import pandas as pd
# import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# app = Flask(__name__)

# # Load model and vectorizer
# model = pickle.load(open('notebooks/model.pkl', 'rb'))
# vectorizer = pickle.load(open('notebooks/vectorizer.pkl', 'rb'))
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     message = request.form['message']
#     lemmatizer = WordNetLemmatizer()
#     #lowercase, remove punctuation, remove stopwords, 
#     text = message.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

#     #tokenization
#     words = text.split()
#     stop_words = set(stopwords.words('english'))
#     words_without_stopwords = [i for i in words if i not in stop_words]

#     #lemmatize
#     lemmatizer_words =[lemmatizer.lemmatize(i) for i in words_without_stopwords]
#     documents= " ".join(lemmatizer_words)
#     vect_msg = vectorizer.transform([documents])

#     #predict
#     prediction = model.predict(vect_msg)[0]
#     return render_template('result.html', prediction=prediction, message=message)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000)) 
#     app.run(host="0.0.0.0", port=port, debug=True)

from flask import Flask, render_template, request
import pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# --- GLOBAL SETUP (Runs once on startup for better performance) ---

# 1. NLTK Resource Check and Download
# This ensures that 'stopwords' and 'wordnet' are available before the app runs.
# On deployment systems like Render, this may need to be handled by a build script,
# but keeping it here as a robust initial step.
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    print("NLTK resources not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    print(f"Error checking/downloading NLTK resources: {e}")

# 2. Initialize Model and Vectorizer
try:
    # Ensure this path is correct relative to your app.py
    MODEL_PATH = 'notebooks/model.pkl' 
    VECTORIZER_PATH = 'notebooks/vectorizer.pkl'
    
    # Check if files exist before loading
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer file not found at the specified path.")
        
    model = pickle.load(open(MODEL_PATH, 'rb'))
    vectorizer = pickle.load(open(VECTORIZER_PATH, 'rb'))
    print("Model and Vectorizer loaded successfully.")

except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    # In a real deployed app, you might want to exit or use a placeholder model
    # For now, we set them to None to prevent immediate crash if paths are wrong
    model = None
    vectorizer = None


# 3. Preprocessing Tool Initialization
# These objects are expensive to create, so we initialize them only once.
lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# ------------------------------------------------------------------

app = Flask(__name__)


def preprocess_message(message):
    """
    Applies the same preprocessing pipeline used during training.
    """
    if not message or not isinstance(message, str):
        return ""

    # Lowercase and remove non-alphanumeric characters
    text = message.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization and Stop-word removal
    words = text.split()
    words_without_stopwords = [
        word for word in words if word not in STOP_WORDS and word
    ]

    # Lemmatize
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words_without_stopwords]
    
    return " ".join(lemmatized_words)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return render_template('result.html', 
                               prediction=-1, # Use a special code for error
                               message="Model not loaded. Check logs for details.")

    # Get the raw message from the form
    raw_message = request.form.get('message', '')
    
    # Apply the full preprocessing pipeline
    processed_document = preprocess_message(raw_message)
    
    if not processed_document:
        # Handle case where message is empty or only stop-words
        prediction = 0 # Default to Ham for an empty/clean message
        prediction_text = "HAM"
    else:
        # Vectorize and Predict
        vect_msg = vectorizer.transform([processed_document])
        prediction_int = model.predict(vect_msg)[0]
        
        # Convert integer prediction to text for display
        prediction_text = "SPAM" if prediction_int == 1 else "HAM"

    return render_template('result.html', 
                           prediction=prediction_text, 
                           message=raw_message)

if __name__ == "__main__":
    # Use environment variable PORT for deployment
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=True)

