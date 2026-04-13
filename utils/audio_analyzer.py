import numpy as np
import subprocess
import os

def convert_webm_to_wav(webm_path):
    """
    Convert webm/ogg audio to WAV using ffmpeg if available,
    otherwise return None so the caller can use a fallback.
    """
    wav_path = webm_path.replace('.webm', '.wav')
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', webm_path, '-ar', '16000', '-ac', '1', '-f', 'wav', wav_path, '-y'],
            capture_output=True, timeout=10
        )
        if result.returncode == 0 and os.path.exists(wav_path):
            return wav_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def extract_audio_features(audio_file_path):
    """
    Goal #20: Extract audio confidence score for the Adaptive Brain.
    Tries Librosa with ffmpeg first, falls back to raw audio analysis.
    Returns a confidence score 0-100.
    """
    try:
        # Step 1: Try converting webm to wav
        wav_path = convert_webm_to_wav(audio_file_path)

        if wav_path:
            try:
                import librosa
                y, sr = librosa.load(wav_path, duration=10, sr=16000)
                os.remove(wav_path)

                if len(y) < 1600:  # Less than 0.1s of audio
                    return 50

                # Extract features for confidence estimation
                # RMS energy — louder = more confident
                rms = np.mean(librosa.feature.rms(y=y))

                # Zero crossing rate — lower = clearer speech
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))

                # Spectral contrast — higher = more articulated
                contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

                # Combine into confidence score
                # High energy + low ZCR + high contrast = confident speaker
                energy_score = min(rms * 500, 40)       # 0-40 points
                clarity_score = max(20 - zcr * 100, 0)  # 0-20 points
                contrast_score = min(contrast * 2, 40)   # 0-40 points

                confidence = int(energy_score + clarity_score + contrast_score)
                return max(10, min(confidence, 100))

            except Exception as e:
                print(f"Librosa analysis error: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        # Step 2: Fallback — analyze raw file size and duration as proxy
        # Larger audio data with consistent chunks = active speaking = more confident
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1000:
            return 30  # Very short/silent — low confidence
        elif file_size < 5000:
            return 45  # Short response
        elif file_size < 20000:
            return 60  # Normal response
        elif file_size < 50000:
            return 72  # Detailed response
        else:
            return 80  # Long, detailed response — high confidence

    except Exception as e:
        print(f"Audio Analysis Error: {e}")
        return 50  # Neutral default
