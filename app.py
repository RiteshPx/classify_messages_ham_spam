from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('notebooks/model.pkl', 'rb'))
vectorizer = pickle.load(open('notebooks/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    vect_msg = vectorizer.transform([message])
    prediction = model.predict(vect_msg)[0]
    return render_template('result.html', prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
