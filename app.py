from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
import binascii
import tempfile

# Import your custom logic
from utils.resume_parser import extract_data_from_pdf
from utils.question_generator import (
    generate_interview_questions,
    generate_adaptive_question
)
from utils.evaluator import evaluate_answer
from utils.adaptive_brain import brain # RL Policy Logic (Goal #20)
from utils.audio_analyzer import analyze_audio_size

# AUDIO_MODE: "real" uses trained SER model (local 16GB RAM)
#             "size" uses file-size heuristic (Render free tier)
AUDIO_MODE = os.getenv("AUDIO_MODE", "real").lower()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "engine": "IntelliView Multimodal RL"})

# -------------------------------------------------------------------
# 1. Resume Parsing (Goal #1)
# -------------------------------------------------------------------
@app.route('/api/extract-resume', methods=['POST'])
def extract_resume():
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ['.pdf', '.docx', '.doc']:
        return jsonify({"message": f"Unsupported format: {ext}. Use .pdf or .docx"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        extracted_data = extract_data_from_pdf(filepath)
        return jsonify({"message": "Resume parsed successfully", "data": extracted_data}), 200
    except Exception as e:
        print(f"Resume parse error: {e}")
        return jsonify({"message": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# -------------------------------------------------------------------
# 2. Initial Question Generation (Updated for Mode Logic)
# -------------------------------------------------------------------
@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json
        # Extract parameters with safety defaults
        mode = data.get('mode', 'custom') 
        tech = data.get('tech', 'General')
        module = data.get('module', 'General')
        topic = data.get('topic', 'General')
        difficulty = data.get('difficulty', 'Medium')
        skills = data.get('skills', []) # Used for Resume Mode

        print(f"Generating initial questions | Mode: {mode} | Tech: {tech}")

        # Pass all parameters to the updated generator
        result = generate_interview_questions(
            tech=tech, 
            module=module, 
            topic=topic, 
            difficulty=difficulty, 
            mode=mode, 
            skills=skills
        )
        
        return jsonify({"data": result}), 200
    except Exception as e:
        print(f"🔥 Generation Error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------------------
# 3. Multimodal Adaptive Step (Goal #10 & #20)
# -------------------------------------------------------------------
@app.route('/api/generate-adaptive-step', methods=['POST'])
def adaptive_step():
    try:
        data = request.json

        # Extract data from the Multimodal Payload
        current_diff = data.get('difficulty', 'Medium')
        answer_text = data.get('answer', '')
        question_text = data.get('question', '')
        face_confidence = data.get('avg_confidence', 50)
        was_skipped = data.get('was_skipped', False)
        history = data.get('history', [])
        tech = data.get('tech', 'Developer')
        mode = data.get('mode', 'custom')
        skills = data.get('skills', [])

        # 1. Evaluate Textual Accuracy
        # Skip the LLM call entirely on skipped questions — no answer to evaluate,
        # saves an API call + ~1s latency.
        if was_skipped:
            eval_result = {"score": 0, "feedback": "Question skipped — no answer given."}
            last_score = 0
        else:
            eval_result = evaluate_answer(question_text, answer_text)
            last_score = eval_result.get('score', 0)

        # 2. Audio Confidence — dual-mode
        #    AUDIO_MODE=real → decode base64 audio, run SER model (local)
        #    AUDIO_MODE=size → size-based heuristic only (Render)
        # Skip audio analysis entirely on skipped questions — silence has no signal.
        audio_confidence = None
        audio_emotion = None
        audio_size = data.get('audio_size', 0)
        audio_base64 = data.get('audio_base64')

        if was_skipped:
            print("⏭️  Skipped — no audio analysis, no evaluator call")
        else:
            if AUDIO_MODE == "real" and audio_base64:
                tmp_path = None
                try:
                    # Validate base64 before writing to disk
                    try:
                        audio_bytes = base64.b64decode(audio_base64, validate=True)
                    except (binascii.Error, ValueError) as decode_err:
                        raise ValueError(f"Invalid base64 audio payload: {decode_err}")

                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    from utils.audio_analyzer import analyze_audio
                    ser_result = analyze_audio(tmp_path)
                    if ser_result is not None:
                        audio_confidence = ser_result['confidence']
                        audio_emotion = ser_result['emotion']
                        print(f"🎙️ SER | Emotion: {audio_emotion} | Confidence: {audio_confidence}")
                    else:
                        audio_confidence = analyze_audio_size(audio_size)
                        print(f"🎙️ SER unavailable, using size heuristic: {audio_confidence}")
                except Exception as audio_err:
                    print(f"⚠️ Audio SER fallback: {audio_err}")
                    audio_confidence = analyze_audio_size(audio_size)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            elif audio_size and audio_size > 0:
                audio_confidence = analyze_audio_size(audio_size)
                print(f"Audio Confidence (size-based): {audio_confidence} ({audio_size} bytes)")

            # Safety default for ANSWERED questions — fall back to neutral if SER returned nothing
            if audio_confidence is None:
                audio_confidence = 50

        # Note: when was_skipped, audio_confidence stays None — brain.fuse_confidence
        # handles this by falling back to face-only score.

        # 3. RL Brain Logic — fuses face + audio confidence (Goal #20)
        next_diff, fused_confidence = brain.decide_next_level(
            current_diff, last_score, face_confidence,
            was_skipped, audio_confidence
        )

        # 4. Generate Next Question — relatable to candidate's answer (Goal #15)
        print(f"Adaptive Step | Score: {last_score} | Face: {face_confidence} | Audio: {audio_confidence} | Fused: {fused_confidence} -> Next: {next_diff}")

        result = generate_adaptive_question(
            tech=tech,
            skills=skills,
            difficulty=next_diff,
            previous_questions=history,
            mode=mode,
            last_answer=answer_text,
            last_feedback=eval_result.get('feedback', ''),
            last_score=last_score,
            fused_confidence=fused_confidence
        )

        return jsonify({
            "question": result.get('question'),
            "new_difficulty": next_diff,
            "last_score": last_score,
            "feedback": eval_result.get('feedback'),
            "fused_confidence": fused_confidence,
            "audio_confidence": audio_confidence,
            "audio_emotion": audio_emotion
        })
    except Exception as e:
        print(f"🔥 Adaptive Step Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)