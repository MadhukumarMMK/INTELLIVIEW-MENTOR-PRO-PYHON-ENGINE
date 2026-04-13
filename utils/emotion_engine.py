import librosa
import numpy as np
import tensorflow as tf # For CNN-LSTM

# Load pre-trained CNN-LSTM model (Trained on RAVDESS/SAVEE datasets)
model = tf.keras.models.load_model('models/speech_emotion_model.h5')

def analyze_speech_confidence(audio_path):
    # Extract MFCCs using Librosa
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    # Reshape for CNN input
    feature = mfccs.reshape(1, 40, 1)
    
    # Predict Emotion/Confidence
    prediction = model.predict(feature)
    # Mapping prediction to a confidence score 0-100
    confidence_score = np.max(prediction) * 100
    return confidence_score