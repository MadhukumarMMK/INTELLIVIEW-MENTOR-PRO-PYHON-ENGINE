import os
import random
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

# Variety pools — LLM picks from a randomized subset each turn to avoid ruts
HR_THEMES = [
    "teamwork and collaboration", "conflict resolution", "leadership under pressure",
    "failure and learning from it", "handling disagreement with a manager",
    "prioritization when overwhelmed", "giving difficult feedback", "adapting to change",
    "making a hard decision with limited info", "motivating a struggling teammate",
    "dealing with ambiguous requirements", "owning a mistake publicly",
    "time management across competing deadlines", "professional growth moments",
    "resolving cross-team dependencies", "managing up / influencing without authority",
    "balancing quality and speed", "cultural differences at work"
]

RESUME_ANGLES = [
    "why they chose that technology over alternatives",
    "the trickiest bug they fixed in that stack",
    "a trade-off they made in their architecture",
    "how they would improve their previous project today",
    "performance optimization they did",
    "a concept from that tech they find confusing",
    "how they'd onboard a junior dev to that stack",
    "a design decision that didn't age well",
    "scaling challenges they anticipate",
    "security considerations they think about"
]

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
        chosen_theme = random.choice(HR_THEMES)
        system_prompt = f"""
        You are an HR interviewer. Generate 1 opening behavioral question.
        Difficulty: {difficulty}.
        THIS QUESTION'S THEME (use this specific angle): {chosen_theme}
        No code or technical concepts. Use a STAR-style opener (Tell me about a time...).
        {CONCISE_RULE}
        Return ONLY a valid JSON object: {{"questions": ["your question"]}}
        """

    elif mode == "resume":
        skills_list = skills if isinstance(skills, list) else []
        skills_str = ", ".join(skills_list) if skills_list else "Full Stack Development"
        chosen_skill = random.choice(skills_list) if skills_list else "their strongest skill"
        chosen_angle = random.choice(RESUME_ANGLES)
        system_prompt = f"""
        You are a technical interviewer. Candidate skills: {skills_str}.
        FOCUS THIS QUESTION ON: {chosen_skill}
        ANGLE (use this framing): {chosen_angle}
        Difficulty: {difficulty}. No code syntax — focus on "why" and "how".
        {CONCISE_RULE}
        Return ONLY a valid JSON object: {{"questions": ["your question"]}}
        """

    else:
        system_prompt = f"""
        You are a technical interviewer.
        Tech: {tech or 'General'} | Module: {module} | Topic: {topic}
        Generate 1 conceptual question strictly within {tech or 'this'} and {topic or 'this topic'}.
        Difficulty: {difficulty}. No code — focus on architecture, trade-offs, and design.
        {CONCISE_RULE}
        Return ONLY a valid JSON object: {{"questions": ["your question"]}}
        """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.95,
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
        # Avoid themes whose first two words already appear in prior questions.
        # Using two words tightens matching so unrelated themes aren't over-filtered.
        def _theme_used(theme, prev):
            head = " ".join(theme.split()[:2]).lower()
            return any(head in q.lower() for q in prev)
        unused_themes = [t for t in HR_THEMES if not _theme_used(t, previous_questions)]
        pool = unused_themes if unused_themes else HR_THEMES
        new_theme = random.choice(pool)
        system_prompt = f"""
    HR interviewer. Difficulty: {difficulty}.
    Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}
    {performance_hint}
    NEW THEME FOR THIS QUESTION (pivot here — don't stay on the last theme): {new_theme}
    Generate ONE short behavioral question on the new theme, loosely connected to their last answer.
    {CONCISE_RULE}
    Return ONLY a valid JSON object: {{"question": "...", "difficulty": "{difficulty}"}}
    """
    elif mode == "resume":
        skills_list = [s for s in (skills or []) if isinstance(s, str) and s.strip()]
        if skills_list:
            unused_skills = [s for s in skills_list
                             if not any(s.lower() in q.lower() for q in previous_questions)]
            next_skill = random.choice(unused_skills if unused_skills else skills_list)
        else:
            # No skills extracted from resume → fall back to generic full-stack topics
            next_skill = random.choice([
                "their primary web framework", "their most-used database",
                "their preferred language", "the deployment stack they know best"
            ])
        next_angle = random.choice(RESUME_ANGLES)
        system_prompt = f"""
    Technical interviewer. {context} | Difficulty: {difficulty}
    Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}
    {performance_hint}
    FOCUS THIS QUESTION ON SKILL: {next_skill}
    ANGLE: {next_angle}
    Generate ONE short follow-up question grounded in that skill.
    No code — focus on "why" and "how". If confidence is low ({fused_confidence}%), be encouraging.
    {CONCISE_RULE}
    Return ONLY a valid JSON object: {{"question": "...", "difficulty": "{difficulty}"}}
    """
    else:
        system_prompt = f"""
    Technical interviewer. {context} | Difficulty: {difficulty}
    Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}
    {performance_hint}
    Stay strictly within {tech}. Generate ONE short follow-up question that extends their last answer.
    No code — focus on "why" and "how". If confidence is low ({fused_confidence}%), be encouraging.
    {CONCISE_RULE}
    Return ONLY a valid JSON object: {{"question": "...", "difficulty": "{difficulty}"}}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.9,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ Groq API Error in generate_adaptive_question: {e}")
        return {"question": "Explain the data flow in your recent project architecture.", "difficulty": difficulty}