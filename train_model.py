"""
IntelliView SER (Speech Emotion Recognition) — Local Training Pipeline
======================================================================
Dataset : RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Features: 40 MFCCs + Chroma (12) + Mel-Spectrogram (128) = 180 features
Model   : 1D-CNN → Dense  (lightweight, <5 MB saved model)
Output  : models/speech_emotion_model.h5

Emotion labels in RAVDESS filenames:
  01=neutral, 02=calm, 03=happy, 04=sad,
  05=angry, 06=fearful, 07=disgust, 08=surprised

RAM usage: ~2-3 GB during training (safe for 16 GB system)
Time     : ~5-15 minutes on CPU
"""

import os
import sys
import glob
import zipfile
import urllib.request
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "dataset", "ravdess")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "speech_emotion_model.h5")
SAMPLE_RATE  = 22050
DURATION     = 3          # seconds to load per clip
N_MFCC       = 40
EPOCHS       = 50
BATCH_SIZE   = 32
TEST_SPLIT   = 0.2

# RAVDESS emotion codes → human labels
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# For the interview context, map each emotion to a confidence bucket
# High confidence: calm, happy, neutral  → the candidate sounds assured
# Medium confidence: surprised            → uncertain but engaged
# Low confidence: sad, angry, fearful, disgust → stressed/negative
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


# ── STEP 1: DOWNLOAD RAVDESS ────────────────────────────────────────
def download_ravdess():
    """Download RAVDESS speech-only dataset (24 actors, ~1440 files, ~250 MB)."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Check if already extracted
    existing = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    if len(existing) >= 1400:
        print(f"✅ RAVDESS already downloaded ({len(existing)} files found)")
        return

    print("📦 Downloading RAVDESS dataset...")
    print("   Source: Zenodo (official RAVDESS mirror)")
    print("   Size: ~250 MB — this will take a few minutes\n")

    # RAVDESS is hosted on Zenodo — 24 actor zip files
    base_url = "https://zenodo.org/record/1188976/files"
    for actor_id in range(1, 25):
        zip_name = f"Audio_Speech_Actors_01-24.zip"
        zip_path = os.path.join(DATASET_DIR, zip_name)

        if not os.path.exists(zip_path):
            url = f"{base_url}/{zip_name}?download=1"
            print(f"   Downloading {zip_name}...")
            try:
                urllib.request.urlretrieve(url, zip_path, _progress_hook)
                print()  # newline after progress
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
                print(f"\n📌 MANUAL DOWNLOAD INSTRUCTIONS:")
                print(f"   1. Go to: https://zenodo.org/record/1188976")
                print(f"   2. Download 'Audio_Speech_Actors_01-24.zip'")
                print(f"   3. Place it in: {DATASET_DIR}")
                print(f"   4. Re-run this script")
                sys.exit(1)
        break  # Only one zip file for all actors

    # Extract
    print("📂 Extracting audio files...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATASET_DIR)
    print(f"✅ Extracted to {DATASET_DIR}")

    # Clean up zip
    os.remove(zip_path)
    print("🗑️  Removed zip file to save space")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    bar_len = 40
    filled = int(bar_len * percent / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r   [{bar}] {percent:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
    sys.stdout.flush()


# ── STEP 2: EXTRACT FEATURES ────────────────────────────────────────
def extract_features(file_path):
    """Extract MFCC + Chroma + Mel features from a single audio file."""
    try:
        y, sr = librosa.load(file_path, duration=DURATION, sr=SAMPLE_RATE)

        # Pad short clips to ensure consistent feature length
        if len(y) < SAMPLE_RATE * DURATION:
            y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')

        # 40 MFCCs — captures vocal tract shape (how words are spoken)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T, axis=0)

        # 12 Chroma — captures pitch/tonal content
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        # 128 Mel-spectrogram — captures overall frequency distribution
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        return np.hstack([mfccs, chroma, mel])  # 180-dim feature vector

    except Exception as e:
        print(f"   ⚠️ Skipping {os.path.basename(file_path)}: {e}")
        return None


def load_dataset():
    """Load all RAVDESS audio files, extract features, and create labels."""
    print("\n🔬 Extracting audio features (MFCC + Chroma + Mel)...")

    features = []
    labels = []
    emotion_counts = {}

    audio_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True)
    if not audio_files:
        print(f"❌ No .wav files found in {DATASET_DIR}")
        print("   Make sure RAVDESS was downloaded and extracted correctly")
        sys.exit(1)

    total = len(audio_files)
    print(f"   Found {total} audio files\n")

    for i, file_path in enumerate(audio_files):
        # RAVDESS filename format: 03-01-05-01-01-02-12.wav
        # 3rd position (index 2) = emotion code
        filename = os.path.basename(file_path)
        parts = filename.split("-")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion_label = EMOTION_MAP.get(emotion_code)

        if emotion_label is None:
            continue

        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(emotion_label)
            emotion_counts[emotion_label] = emotion_counts.get(emotion_label, 0) + 1

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"   Processed {i+1}/{total} files...")

    print(f"\n📊 Dataset Summary:")
    for emo, count in sorted(emotion_counts.items()):
        print(f"   {emo:12s}: {count} samples")
    print(f"   {'TOTAL':12s}: {len(features)} samples")

    return np.array(features), np.array(labels)


# ── STEP 3: BUILD & TRAIN CNN MODEL ─────────────────────────────────
def build_and_train(X, y):
    """Build a 1D-CNN model and train on the extracted features."""
    # Lazy import — only needed during training
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Dense, Dropout, Conv1D, MaxPooling1D,
        Flatten, BatchNormalization
    )
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print("\n🏗️  Building CNN model...")

    # Ensure model output directory exists before saving scaler/labels
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Encode labels: "happy" → 0, "sad" → 1, etc.
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)
    num_classes = y_cat.shape[1]

    # Save label mapping for inference
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print(f"   Classes: {label_mapping}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler parameters for inference
    np.save(os.path.join(MODEL_DIR, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "label_classes.npy"), le.classes_)

    # Reshape for 1D-CNN: (samples, features, 1)
    X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_cat, test_size=TEST_SPLIT, random_state=42, stratify=y_encoded
    )

    print(f"   Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"   Feature dim: {X_train.shape[1]} | Classes: {num_classes}\n")

    # Build 1D-CNN architecture
    model = Sequential([
        # Block 1
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 2
        Conv1D(128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Block 3
        Conv1D(256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=10,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    )

    # Train
    print("\n🚀 Training started...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Final Results:")
    print(f"   Test Accuracy: {accuracy * 100:.2f}%")
    print(f"   Test Loss:     {loss:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n💾 Model saved: {MODEL_PATH} ({model_size:.1f} MB)")

    return model, history, accuracy


# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  IntelliView — Speech Emotion Recognition Training")
    print("  Dataset: RAVDESS | Model: 1D-CNN | Features: MFCC+Chroma+Mel")
    print("=" * 60)

    # Step 1: Download dataset
    download_ravdess()

    # Step 2: Extract features
    X, y = load_dataset()

    # Step 3: Train model
    model, history, accuracy = build_and_train(X, y)

    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete! Accuracy: {accuracy * 100:.2f}%")
    print(f"  📁 Model file: {MODEL_PATH}")
    print(f"  📁 Scaler:     {MODEL_DIR}/scaler_mean.npy, scaler_scale.npy")
    print(f"  📁 Labels:     {MODEL_DIR}/label_classes.npy")
    print("=" * 60)
    print("\n  Next step: Run your Python engine locally (python app.py)")
    print("  The audio analyzer will automatically use the trained model.")
    print()
