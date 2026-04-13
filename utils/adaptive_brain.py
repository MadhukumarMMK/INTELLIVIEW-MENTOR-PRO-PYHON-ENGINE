import numpy as np

class AdaptiveInterviewer:
    def __init__(self):
        # Levels: 0: Easy, 1: Medium, 2: Hard
        self.levels = ["Easy", "Medium", "Hard"]
        # Weights for multimodal confidence fusion (Goal #20)
        self.face_weight = 0.6
        self.audio_weight = 0.4

    def fuse_confidence(self, face_confidence, audio_confidence):
        """
        Goal #20: Combine Face-API visual confidence and Librosa/CNN-LSTM
        audio confidence into a single multimodal score.
        If audio is unavailable (0 or None), fall back to face-only.
        """
        if audio_confidence is None or audio_confidence <= 0:
            return face_confidence
        return (self.face_weight * face_confidence) + (self.audio_weight * audio_confidence)

    def decide_next_level(self, current_level_str, accuracy_score, face_confidence, was_skipped=False, audio_confidence=None):
        """
        Policy Logic (Goal #9 & #10 & #20):
        - If skipped: Stay at current level (do not count as attempt).
        - Fuses face + audio confidence into a single multimodal score.
        - High Accuracy (>=80) + High Confidence (>=70): Level Up.
        - Low Accuracy (<=40) or Low Confidence (<=30): Level Down.
        - Otherwise: Stay.
        """
        if was_skipped:
            return current_level_str, self.fuse_confidence(face_confidence, audio_confidence)

        # Multimodal confidence fusion
        confidence_score = self.fuse_confidence(face_confidence, audio_confidence)

        current_idx = self.levels.index(current_level_str)

        # Logic for "Strong Candidate" -> Level Up
        if accuracy_score >= 80 and confidence_score >= 70:
            next_idx = min(current_idx + 1, 2)

        # Logic for "Struggling Candidate" -> Level Down
        elif accuracy_score <= 40 or confidence_score <= 30:
            next_idx = max(current_idx - 1, 0)

        # Logic for "Consistent Candidate" -> Stay
        else:
            next_idx = current_idx

        return self.levels[next_idx], round(confidence_score, 2)

# Global instance
brain = AdaptiveInterviewer()