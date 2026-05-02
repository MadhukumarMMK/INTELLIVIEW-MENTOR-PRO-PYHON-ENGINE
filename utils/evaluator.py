import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Accept either ANTHROPIC_API_KEY (convention) or Anthropic_API_KEY (the name
# this project's .env historically used) so an old .env keeps working.
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Anthropic_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None

# Frozen system prompt — identical across every call so the prefix is cacheable.
# Question + answer go in the user turn so they don't invalidate the cached prefix.
EVALUATOR_SYSTEM_PROMPT = """You are a technical interviewer evaluating a candidate's response.

Evaluate the answer for technical accuracy and completeness. Return a JSON object with exactly two fields:
1. 'score': integer in [0, 100]
2. 'feedback': a single concise sentence explaining the score

Respond with ONLY the JSON object — no preamble, no markdown fences, no commentary.
Example: {"score": 85, "feedback": "Excellent explanation, but could mention X."}"""


def _extract_json(text: str) -> dict:
    """Tolerantly extract a JSON object from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` or ``` ... ``` fences if the model adds them
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    return json.loads(text)


def evaluate_answer(question, user_answer):
    # Safety check: Anthropic client failed to initialize
    if not client:
        return {"score": 50, "feedback": "AI Engine configuration missing. Check .env file."}

    # Goal #9 & #10: Handle skipped or very short answers gracefully
    if not user_answer or len(user_answer.strip()) < 5:
        return {"score": 0, "feedback": "The answer was too short or skipped to be evaluated technically."}

    user_message = f"Question: {question}\n\nCandidate's Answer: {user_answer}"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=512,
            temperature=0.1,  # objective technical scoring
            system=[
                {
                    "type": "text",
                    "text": EVALUATOR_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        )

        text = next((b.text for b in response.content if b.type == "text"), "")
        result = _extract_json(text)
        return {
            "score": result.get("score", 0),
            "feedback": result.get("feedback", "No specific feedback provided."),
        }
    except Exception as e:
        print(f"❌ Anthropic Evaluation Error: {e}")
        # Goal #5: Fallback score so the session doesn't hang
        return {"score": 40, "feedback": "System encountered an error during evaluation."}
