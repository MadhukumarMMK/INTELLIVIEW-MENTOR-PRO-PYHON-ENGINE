"""
IntelliView Audio Analyzer — Real Speech Emotion Recognition
============================================================
Supports TWO model types (auto-detected via model_meta.json):

1. Basic 1D-CNN (train_model.py — local, 44% accuracy)
   - Input: 180 averaged features (40 MFCC + 12 Chroma + 128 Mel)

2. Production 2D-CNN (train_colab.ipynb — GPU, 70-80% accuracy)
   - Input: 128x130 mel-spectrogram
   - 6x augmented training data

Falls back to a file-size heuristic if no model is found.
"""

import os
# Force Keras 3 to use TensorFlow backend — must be set BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info logs

import json
import shutil
import glob
import subprocess
import numpy as np

# ── RESOLVE ffmpeg PATH AT IMPORT TIME ──────────────────────────────
def _find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget_glob = os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_*\ffmpeg-*\bin\ffmpeg.exe"
    )
    matches = glob.glob(winget_glob)
    if matches:
        return matches[0]
    for candidate in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


FFMPEG_BIN = _find_ffmpeg()
if FFMPEG_BIN:
    print(f"🎬 ffmpeg located at: {FFMPEG_BIN}")
else:
    print("⚠️  ffmpeg not found — audio conversion will be skipped")

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "speech_emotion_model.h5")
META_PATH   = os.path.join(MODEL_DIR, "model_meta.json")

# Basic 1D model files
SCALER_MEAN = os.path.join(MODEL_DIR, "scaler_mean.npy")
SCALER_STD  = os.path.join(MODEL_DIR, "scaler_scale.npy")

# Production 2D model files
GLOBAL_MEAN = os.path.join(MODEL_DIR, "global_mean.npy")
GLOBAL_STD  = os.path.join(MODEL_DIR, "global_std.npy")

LABEL_PATH  = os.path.join(MODEL_DIR, "label_classes.npy")

SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40

# Emotion → confidence score mapping
EMOTION_TO_CONFIDENCE = {
    "neutral":   75,
    "calm":      85,
    "happy":     80,
    "sad":       30,
    "angry":     35,
    "fearful":   20,
    "disgust":   25,
    "surprised": 55
}

# ── LAZY-LOADED STATE ───────────────────────────────────────────────
_model = None
_label_classes = None
_model_type = None          # "1d_basic" or "2d_cnn_mel_spectrogram"
_norm_params = {}           # holds scaler_mean/scale OR global_mean/std
_model_meta = None
_model_load_attempted = False


def _load_model_once():
    """Load model on first call — auto-detect 1D-basic vs 2D-production."""
    global _model, _label_classes, _model_type, _norm_params
    global _model_meta, _model_load_attempted

    if _model_load_attempted:
        return _model is not None

    _model_load_attempted = True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        print("⚠️  SER model not found — falling back to size heuristic")
        print(f"    Expected: {MODEL_PATH}")
        print("    Train with: python train_model.py  OR  run train_colab.ipynb")
        return False

    # Detect model type via meta file (production model) or fallback (basic)
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            _model_meta = json.load(f)
        _model_type = _model_meta.get('model_type', '1d_basic')
    else:
        _model_type = '1d_basic'
        _model_meta = {}

    try:
        # Support both old (tf.keras) and new (standalone keras 3) APIs
        try:
            import keras
            load_model = keras.models.load_model
        except ImportError:
            import tensorflow as tf
            load_model = tf.keras.models.load_model
        print(f"🧠 Loading SER model ({_model_type}) from {MODEL_PATH}...")
        _model = load_model(MODEL_PATH, compile=False)
        _label_classes = np.load(LABEL_PATH, allow_pickle=True)

        if _model_type == '2d_cnn_mel_spectrogram':
            _norm_params['mean'] = np.load(GLOBAL_MEAN)[0]
            _norm_params['std'] = np.load(GLOBAL_STD)[0]
            acc = _model_meta.get('test_accuracy', 0) * 100
            print(f"✅ Production 2D-CNN loaded | Accuracy: {acc:.1f}% | Classes: {list(_label_classes)}")
        else:
            _norm_params['mean'] = np.load(SCALER_MEAN)
            _norm_params['scale'] = np.load(SCALER_STD)
            print(f"✅ Basic 1D-CNN loaded | Classes: {list(_label_classes)}")

        return True
    except Exception as e:
        print(f"❌ Failed to load SER model: {e}")
        _model = None
        return False


# ── AUDIO CONVERSION ────────────────────────────────────────────────
def convert_webm_to_wav(webm_path):
    if not FFMPEG_BIN:
        return None
    wav_path = webm_path.replace('.webm', '.wav').replace('.ogg', '.wav')
    if wav_path == webm_path:
        wav_path = webm_path + '.wav'
    try:
        result = subprocess.run(
            [FFMPEG_BIN, '-i', webm_path, '-ar', str(SAMPLE_RATE),
             '-ac', '1', '-f', 'wav', wav_path, '-y'],
            capture_output=True, timeout=10
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
        print(f"⚠️  ffmpeg exit {result.returncode}: {result.stderr[:200].decode('utf-8', errors='ignore')}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠️  ffmpeg error: {e}")
    return None


# ── FEATURE EXTRACTION (two variants) ───────────────────────────────
def _extract_features_1d(wav_path):
    """Basic model: 40 MFCC + 12 Chroma + 128 Mel, averaged → 180-dim."""
    import librosa
    y, sr = librosa.load(wav_path, duration=DURATION, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel])


def _extract_features_2d(wav_path):
    """Production model: full mel-spectrogram, 128 bands x N_FRAMES time."""
    import librosa
    n_mels = _model_meta.get('n_mels', 128)
    n_frames = _model_meta.get('n_frames', 130)

    y, sr = librosa.load(wav_path, duration=DURATION, sr=SAMPLE_RATE)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < n_frames:
        pad = n_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant', constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :n_frames]
    return mel_db


# ── PUBLIC API ──────────────────────────────────────────────────────
def analyze_audio(audio_file_path):
    """Run SER. Returns {emotion, confidence, emotion_probability, ...} or None."""
    if not _load_model_once():
        return None

    try:
        wav_path = convert_webm_to_wav(audio_file_path)
        if wav_path is None:
            if audio_file_path.endswith('.wav'):
                wav_path = audio_file_path
            else:
                return None

        # Extract features per model type
        if _model_type == '2d_cnn_mel_spectrogram':
            mel = _extract_features_2d(wav_path)
            # Per-sample z-score normalization (same as training)
            mel_norm = (mel - mel.mean()) / (mel.std() + 1e-8)
            features_cnn = mel_norm[np.newaxis, ..., np.newaxis]  # (1, 128, 130, 1)
        else:
            features = _extract_features_1d(wav_path)
            features_scaled = (features - _norm_params['mean']) / _norm_params['scale']
            features_cnn = features_scaled.reshape(1, features_scaled.shape[0], 1)

        if wav_path != audio_file_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        probs = _model.predict(features_cnn, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        emotion = str(_label_classes[pred_idx])
        emotion_prob = float(probs[pred_idx])

        # Blend toward neutral when model is uncertain
        base_confidence = EMOTION_TO_CONFIDENCE.get(emotion, 50)
        final_confidence = int(base_confidence * emotion_prob + 50 * (1 - emotion_prob))

        return {
            "emotion": emotion,
            "confidence": final_confidence,
            "emotion_probability": round(emotion_prob, 3),
            "model_type": _model_type,
            "all_probabilities": {
                str(_label_classes[i]): round(float(probs[i]), 3)
                for i in range(len(probs))
            }
        }

    except Exception as e:
        print(f"❌ SER analysis error: {e}")
        return None


def extract_audio_features(audio_file_path):
    """Legacy entry point — returns single confidence 0-100 with fallback."""
    result = analyze_audio(audio_file_path)
    if result is not None:
        print(f"🎙️  SER: emotion={result['emotion']} "
              f"(p={result['emotion_probability']}) → confidence={result['confidence']}")
        return result["confidence"]
    try:
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:    return 30
        elif file_size < 5000:  return 45
        elif file_size < 20000: return 60
        elif file_size < 50000: return 72
        else:                   return 80
    except Exception as e:
        print(f"Audio Analysis Error: {e}")
        return 50


def analyze_audio_size(audio_size):
    """Size-only heuristic for Render deployment (no binary sent)."""
    if not audio_size or audio_size <= 0:
        return None
    if audio_size < 1000:    return 30
    elif audio_size < 5000:  return 45
    elif audio_size < 20000: return 60
    elif audio_size < 50000: return 72
    else:                    return 80
