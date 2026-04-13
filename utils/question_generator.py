import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize the Groq client
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

def generate_interview_questions(tech=None, module=None, topic=None, difficulty="Medium", mode="custom", skills=None):
    """
    Goal #1, #2, & #3: Theory-focused Multi-mode generation.
    Updated to prioritize conceptual communication over coding syntax.
    """
    # Safety Check: If Groq client failed to initialize, provide a mode-aware fallback.
    if not client:
        print("❌ Groq client not initialized. Check GROQ_API_KEY in .env file.")
        fallback_q = "Describe a challenging situation you faced at work and how you handled it." if mode == "hr" else "Explain the core architectural principles of your primary stack."
        return {"questions": [fallback_q]}

    # Conciseness rule applied to all modes
    CONCISE_RULE = """
    CRITICAL: Keep the question SHORT and CRISPY — maximum 1-2 sentences (under 30 words).
    Ask ONE clear thing. Do NOT combine multiple questions into one.
    The candidate must understand instantly without re-reading.
    """

    if mode == "hr":
        system_prompt = f"""
        You are an HR interviewer. Generate 1 opening behavioral question.
        Difficulty: {difficulty}. Focus: conflict resolution, leadership, adaptability.
        No code or technical concepts.
        {CONCISE_RULE}
        Return ONLY JSON: {{"questions": ["your question"]}}
        """

    elif mode == "resume":
        skills_list = skills if isinstance(skills, list) else []
        skills_str = ", ".join(skills_list) if skills_list else "Full Stack Development"
        system_prompt = f"""
        You are a technical interviewer. Candidate skills: {skills_str}.
        Generate 1 theoretical question about their strongest skill.
        Difficulty: {difficulty}. No code syntax — focus on "why" and "how".
        {CONCISE_RULE}
        Return ONLY JSON: {{"questions": ["your question"]}}
        """

    else:
        system_prompt = f"""
        You are a technical interviewer.
        Tech: {tech or 'General'} | Module: {module} | Topic: {topic}
        Generate 1 conceptual question. Difficulty: {difficulty}.
        No code — focus on architecture, trade-offs, and design.
        {CONCISE_RULE}
        Return ONLY JSON: {{"questions": ["your question"]}}
        """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    
    except Exception as e:
        print(f"❌ Groq API Error in generate_interview_questions: {e}")
        # Provide a mode-aware fallback question
        fallback_q = "Describe a challenging situation you faced at work and how you handled it." if mode == "hr" else "Explain the core architectural principles of your primary stack."
        return {"questions": [fallback_q]}

# --- Adaptive Pacing Logic (Goal #10 & #20) ---

def generate_adaptive_question(tech, skills, difficulty, previous_questions, mode="custom",
                                last_answer="", last_feedback="", last_score=0, fused_confidence=50):
    """
    Goal #10, #15 & #20: Generates the next RELATABLE, ADAPTIVE, INTERACTIVE question.
    - Relatable: References what the candidate just said in their answer.
    - Adaptive: Adjusts depth/tone based on score + multimodal confidence.
    - Interactive: Maintains conversational interview flow across all modes.
    """
    # Safety Check: If Groq client failed to initialize
    if not client:
        print("❌ Groq client not initialized. Check GROQ_API_KEY in .env file.")
        return {"question": "Explain the data flow in your recent project architecture.", "difficulty": difficulty}

    # Build context based on interview mode
    if mode == "hr":
        context = "HR Behavioral Interview — focus on leadership, conflict resolution, and adaptability"
    elif mode == "resume":
        context = f"Resume-Based Interview — Candidate Skills: {', '.join(skills) if skills else 'Full Stack Development'}"
    else:
        context = f"Technical Interview — Technology: {tech}"

    # Build candidate performance snapshot for the LLM
    performance_hint = ""
    if last_answer and last_answer != "SKIPPED":
        performance_hint = f"""
    --- CANDIDATE'S LAST RESPONSE ---
    Their Answer: "{last_answer}"
    Evaluator Feedback: {last_feedback}
    Accuracy Score: {last_score}/100
    Multimodal Confidence (face + voice): {fused_confidence}/100
    ---
    USE this answer to make your next question a DIRECT FOLLOW-UP.
    If they mentioned a concept, dig deeper into it.
    If they were vague, ask them to clarify that specific point.
    If they were wrong, gently redirect by asking about the correct approach.
    """
    elif last_answer == "SKIPPED":
        last_q_ref = previous_questions[-1] if previous_questions else "the previous question"
        performance_hint = f"""
    --- CANDIDATE SKIPPED THE LAST QUESTION ---
    Skipped Question: "{last_q_ref}"
    They may be uncomfortable with that specific topic.
    Pivot to a DIFFERENT but related angle at an easier depth.
    Do NOT re-ask the same question in different words.
    """

    prev_q_str = ", ".join(previous_questions) if previous_questions else "None yet"

    CONCISE_RULE = """
    CRITICAL: Keep the question SHORT — max 1-2 sentences, under 30 words.
    Ask ONE clear thing. No multi-part questions. The candidate must get it instantly.
    """

    if mode == "hr":
        system_prompt = f"""
    HR interviewer. Difficulty: {difficulty}.
    Previous (DON'T REPEAT): {prev_q_str}
    {performance_hint}
    Generate ONE short follow-up behavioral question connected to the candidate's answer.
    {CONCISE_RULE}
    Return ONLY a valid JSON object: {{"question": "...", "difficulty": "{difficulty}"}}
    """
    else:
        system_prompt = f"""
    Technical interviewer. {context} | Difficulty: {difficulty}
    Previous (DON'T REPEAT): {prev_q_str}
    {performance_hint}
    Generate ONE short follow-up question building on the candidate's last answer.
    No code — focus on "why" and "how". If confidence is low ({fused_confidence}%), be encouraging.
    {CONCISE_RULE}
    Return ONLY a valid JSON object: {{"question": "...", "difficulty": "{difficulty}"}}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq API Error in generate_adaptive_question: {e}")
        return {"question": "Explain the data flow in your recent project architecture.", "difficulty": difficulty}