import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Groq client with a fallback to prevent 500 errors if ENV is missing
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

def evaluate_answer(question, user_answer):
    # Safety Check: If Groq client failed to initialize
    if not client:
        return {"score": 50, "feedback": "AI Engine configuration missing. Check .env file."}

    # Goal #9 & #10: Handle skipped or very short answers gracefully
    if not user_answer or len(user_answer.strip()) < 5:
        return {"score": 0, "feedback": "The answer was too short or skipped to be evaluated technically."}

    system_prompt = f"""
    You are a technical interviewer evaluating a candidate's response.
    
    Question: {question}
    Candidate's Answer: {user_answer}
    
    Evaluate the answer for technical accuracy and completeness. 
    Return a JSON object with:
    1. 'score': (0-100 integer)
    2. 'feedback': (A single concise sentence explaining the score)
    
    Format: {{"score": 85, "feedback": "Excellent explanation, but could mention X."}}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.3-70b-versatile", # Highly recommended for more accurate technical evaluation
            temperature=0.1, # Extremely low temperature for objective technical scoring
            response_format={"type": "json_object"}
        )
        
        # Parse result and ensure keys exist before returning to Flask
        result = json.loads(chat_completion.choices[0].message.content)
        return {
            "score": result.get("score", 0),
            "feedback": result.get("feedback", "No specific feedback provided.")
        }
    except Exception as e:
        print(f"❌ Groq Evaluation Error: {e}")
        # Goal #5: Ensure a fallback score is returned so the session doesn't hang
        return {"score": 40, "feedback": "System encountered an error during evaluation."}