# Local SER (Speech Emotion Recognition) Setup

Train a real audio emotion model locally, then use it in the adaptive interview pipeline.

## Prerequisites

- **16 GB RAM** (for training)
- **~2 GB free disk** (dataset + model)
- **Python 3.10+** (already in your `venv`)
- **ffmpeg** on PATH (for webm → wav conversion)
  - Windows: `winget install ffmpeg` or download from https://www.gyan.dev/ffmpeg/builds/
  - Verify: `ffmpeg -version`

## One-Time Training (~10-20 minutes)

```bash
cd intelliview-python-engine
.\venv\Scripts\activate                     # Windows
pip install -r requirements-training.txt
python train_model.py
```

The script will:
1. Download RAVDESS dataset (~250 MB) from Zenodo
2. Extract MFCC + Chroma + Mel features from 1,440 audio clips
3. Train a 1D-CNN (auto-stops when accuracy plateaus)
4. Save `models/speech_emotion_model.h5` (~5 MB)

Expected test accuracy: **60-75%** on 8 emotion classes.

## Running the App Locally (Real SER Mode)

After training, just start your services normally:

```bash
# Terminal 1 — Python engine
cd intelliview-python-engine
python app.py

# Terminal 2 — Node backend
cd backend
npm start

# Terminal 3 — React frontend
cd frontend
npm start
```

The Python engine detects `models/speech_emotion_model.h5` and switches to real SER automatically. You'll see in the logs:

```
🧠 Loading SER model from .../speech_emotion_model.h5...
✅ SER model loaded | Classes: ['angry' 'calm' 'disgust' 'fearful' 'happy' 'neutral' 'sad' 'surprised']
```

And per question:

```
🎙️ SER | Emotion: calm | Confidence: 82
Adaptive Step | Score: 75 | Face: 68 | Audio: 82 | Fused: 73.6 -> Next: Hard
```

## Environment Toggle

| Variable | Values | Used When |
|----------|--------|-----------|
| `AUDIO_MODE` in backend `.env` | `real` (default) \| `size` | `size` = Render (no audio binary sent) |
| `AUDIO_MODE` in python `.env`  | `real` (default) \| `size` | `size` = skip model loading |

For Render deployment, set `AUDIO_MODE=size` on both services. Locally, leave as `real`.

## Architecture Summary

```
Frontend (MediaRecorder)
    │ audio_chunk (webm)
    ▼
Backend (Socket.io)
    │ base64-encode when AUDIO_MODE=real
    ▼
Python Engine
    │ ffmpeg: webm → wav
    │ librosa: extract 180 features (40 MFCC + 12 Chroma + 128 Mel)
    │ CNN model: predict emotion + probability
    │ Map: emotion → confidence score (20-85)
    ▼
RL Brain: (0.6 × face) + (0.4 × audio) → next difficulty
```

## Confidence Mapping

| Emotion | Base Confidence | Reasoning |
|---------|----------------|-----------|
| calm | 85 | Best interview state |
| happy | 80 | Engaged, positive |
| neutral | 75 | Composed |
| surprised | 55 | Uncertain but alert |
| angry | 35 | Frustrated |
| sad | 30 | Discouraged |
| disgust | 25 | Negative affect |
| fearful | 20 | Nervous, lowest confidence |

Final confidence = `base × prediction_probability + 50 × (1 - prediction_probability)`
— uncertain predictions blend toward neutral 50, confident ones use the mapping directly.
