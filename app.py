from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
from flask_cors import CORS
from flask import render_template

model = tf.keras.models.load_model('emotion_model.h5')

app = Flask(__name__)
CORS(app) 

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

def extract_mfcc(audio_file):
    audio, sr = librosa.load(audio_file, duration=3) 
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file and file.filename.endswith('.wav'):
        audio_file_path = f"temp_{file.filename}"
        file.save(audio_file_path)

        features = extract_mfcc(audio_file_path)
        features = features.reshape(1, 13, 1, 1) 

        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        return jsonify({'emotion': emotion_labels[predicted_class]})

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
