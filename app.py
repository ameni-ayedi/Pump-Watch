import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import pickle 

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load scaler and top features indices
with open('scaler_subset.pkl', 'rb') as f:
    scaler_subset = pickle.load(f)

top_features_indices = np.load('top_features_indices.npy')
N_MFCC = 13 

# Load the trained model
loaded_model = load_model('pump_watch_model.keras')


def extract_mfcc(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs.mean(axis=1)

def extract_spectral(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0]
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast)

def extract_temporal(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def extract_additional_features(audio, sample_rate):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    return np.mean(chroma_stft), np.mean(spec_bw), np.mean(spec_flatness), np.mean(rolloff), np.mean(rms)

def extract_features(audio_data, sample_rate):
    features = []
    for audio in audio_data:
        mfccs = extract_mfcc(audio, sample_rate, N_MFCC)
        spectral_features = extract_spectral(audio, sample_rate)
        temporal_features = extract_temporal(audio)
        additional_features = extract_additional_features(audio, sample_rate)
        all_features = np.concatenate([mfccs, spectral_features, temporal_features, additional_features])
        features.append(all_features)
    return np.array(features)


#  EXTRACTING FEATURES FUNCTION

def preprocess_audio(file_path, scaler_subset, top_features_indices):
    audio, sample_rate = librosa.load(file_path, sr=None)
    audio_data = [audio]  
    features = extract_features(audio_data, sample_rate)
    features_subset = features[:, top_features_indices]
    scaled_features = scaler_subset.transform(features_subset)
    return scaled_features


#  UPLOAD & PREDICTION ROUTE

@app.route('/upload', methods=['POST'])
def upload_file():
    if not request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file_key = next(iter(request.files))
    file = request.files[file_key]

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File extension not allowed'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        preprocessed_audio = preprocess_audio(file_path, scaler_subset, top_features_indices)
        reconstructed_audio = loaded_model.predict(preprocessed_audio)
        mse = np.mean(np.power(preprocessed_audio - reconstructed_audio, 2), axis=1)
        optimal_threshold = 0.5
        classification = "Abnormal" if mse[0] > optimal_threshold else "Normal"

        return jsonify({
            'Prediction': classification,
            'Message': f'{file.filename} uploaded successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
