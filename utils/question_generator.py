import os
import json
import random
from anthropic import Anthropic
from dotenv import load_dotenv

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

# Accept either ANTHROPIC_API_KEY (convention) or Anthropic_API_KEY (the name
# this project's .env historically used) so an old .env keeps working.
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("Anthropic_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None

CONCISE_RULE = """CRITICAL: Keep the question SHORT and CRISPY — maximum 1-2 sentences (under 30 words).
Ask ONE clear thing. Do NOT combine multiple questions into one. The candidate must understand instantly without re-reading."""

# --- Frozen per-mode system prompts (cacheable). All dynamic context goes in the user turn. ---

HR_INITIAL_SYSTEM = f"""You are an HR interviewer generating the OPENING behavioral question of an interview.
No code or technical concepts. Use a STAR-style opener (e.g. "Tell me about a time...").
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"questions": ["your question"]}}"""

RESUME_INITIAL_SYSTEM = f"""You are a technical interviewer generating the OPENING question for a resume-based interview.
No code syntax — focus on "why" and "how".
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"questions": ["your question"]}}"""

CUSTOM_INITIAL_SYSTEM = f"""You are a technical interviewer generating the OPENING question for a topic-based interview.
No code — focus on architecture, trade-offs, and design.
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"questions": ["your question"]}}"""

HR_ADAPTIVE_SYSTEM = f"""You are an HR interviewer running a multi-turn behavioral interview.
DO NOT repeat or rephrase any previous question.
Pivot to the new theme provided — don't stay on the previous theme.
Loosely connect the new question to the candidate's last answer when one is given.
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"question": "...", "difficulty": "..."}}"""

RESUME_ADAPTIVE_SYSTEM = f"""You are a technical interviewer running a multi-turn resume-based interview.
DO NOT repeat or rephrase any previous question.
Generate a follow-up grounded in the focus skill provided.
No code — focus on "why" and "how". If candidate confidence is low, be encouraging.
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"question": "...", "difficulty": "..."}}"""

CUSTOM_ADAPTIVE_SYSTEM = f"""You are a technical interviewer running a multi-turn topic-based interview.
DO NOT repeat or rephrase any previous question.
Stay strictly within the technology provided. Extend the candidate's last answer.
No code — focus on "why" and "how". If candidate confidence is low, be encouraging.
{CONCISE_RULE}
Return ONLY a valid JSON object in this exact shape: {{"question": "...", "difficulty": "..."}}"""


def _extract_json(text: str) -> dict:
    """Tolerantly extract a JSON object from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.strip()
    return json.loads(text)


def _normalize_adaptive_payload(parsed: dict, fallback_difficulty: str) -> dict:
    """
    Adaptive endpoint expects {"question": str, "difficulty": str}.
    But Claude occasionally returns the initial-question shape
    {"questions": [str]} — accept that too. Returns a guaranteed-valid
    payload, or raises ValueError if neither shape has a question.
    """
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    # Try the canonical shape first
    q = parsed.get("question")

    # Fall back to the initial-question shape
    if not q and isinstance(parsed.get("questions"), list) and parsed["questions"]:
        q = parsed["questions"][0]

    # If question is itself a dict (rare), pull text out of it
    if isinstance(q, dict):
        q = q.get("question") or q.get("text")

    if not q or not isinstance(q, str) or not q.strip():
        raise ValueError(f"No usable 'question' in payload: {parsed}")

    return {
        "question": q.strip(),
        "difficulty": parsed.get("difficulty") or fallback_difficulty,
    }


def _call_claude(system_prompt: str, user_message: str, temperature: float) -> str:
    """Single Anthropic call with cached system prompt. Returns the text content."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=temperature,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    return next((b.text for b in response.content if b.type == "text"), "")


def generate_interview_questions(tech=None, module=None, topic=None, difficulty="Medium", mode="custom", skills=None):
    """
    Goal #1, #2, & #3: Theory-focused multi-mode generation.
    Returns the OPENING question only — adaptive turns come from generate_adaptive_question.
    """
    if not client:
        print("❌ Anthropic client not initialized. Check ANTHROPIC_API_KEY in .env file.")
        fallback_q = "Describe a challenging situation you faced at work and how you handled it." if mode == "hr" else "Explain the core architectural principles of your primary stack."
        return {"questions": [fallback_q]}

    if mode == "hr":
        chosen_theme = random.choice(HR_THEMES)
        system_prompt = HR_INITIAL_SYSTEM
        user_msg = (
            f"Difficulty: {difficulty}\n"
            f"Theme for this question: {chosen_theme}\n\n"
            f"Generate the question now."
        )

    elif mode == "resume":
        skills_list = skills if isinstance(skills, list) else []
        skills_str = ", ".join(skills_list) if skills_list else "Full Stack Development"
        chosen_skill = random.choice(skills_list) if skills_list else "their strongest skill"
        chosen_angle = random.choice(RESUME_ANGLES)
        system_prompt = RESUME_INITIAL_SYSTEM
        user_msg = (
            f"Candidate skills: {skills_str}\n"
            f"Focus this question on: {chosen_skill}\n"
            f"Angle (use this framing): {chosen_angle}\n"
            f"Difficulty: {difficulty}\n\n"
            f"Generate the question now."
        )

    else:
        system_prompt = CUSTOM_INITIAL_SYSTEM
        user_msg = (
            f"Tech: {tech or 'General'} | Module: {module} | Topic: {topic}\n"
            f"Stay strictly within {tech or 'this'} and {topic or 'this topic'}.\n"
            f"Difficulty: {difficulty}\n\n"
            f"Generate the question now."
        )

    try:
        text = _call_claude(system_prompt, user_msg, temperature=0.95)
        return _extract_json(text)
    except Exception as e:
        print(f"❌ Anthropic API Error in generate_interview_questions: {e}")
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
    if not client:
        print("❌ Anthropic client not initialized. Check ANTHROPIC_API_KEY in .env file.")
        return {"question": "Explain the data flow in your recent project architecture.", "difficulty": difficulty}

    # Build candidate performance hint
    performance_hint = ""
    if last_answer and last_answer != "SKIPPED":
        performance_hint = (
            "--- CANDIDATE'S LAST RESPONSE ---\n"
            f"Their Answer: \"{last_answer}\"\n"
            f"Evaluator Feedback: {last_feedback}\n"
            f"Accuracy Score: {last_score}/100\n"
            f"Multimodal Confidence (face + voice): {fused_confidence}/100\n"
            "---\n"
            "USE this answer to make your next question a DIRECT FOLLOW-UP.\n"
            "If they mentioned a concept, dig deeper into it.\n"
            "If they were vague, ask them to clarify that specific point.\n"
            "If they were wrong, gently redirect by asking about the correct approach."
        )
    elif last_answer == "SKIPPED":
        last_q_ref = previous_questions[-1] if previous_questions else "the previous question"
        performance_hint = (
            "--- CANDIDATE SKIPPED THE LAST QUESTION ---\n"
            f"Skipped Question: \"{last_q_ref}\"\n"
            "They may be uncomfortable with that specific topic.\n"
            "Pivot to a DIFFERENT but related angle at an easier depth.\n"
            "Do NOT re-ask the same question in different words."
        )

    prev_q_str = ", ".join(previous_questions) if previous_questions else "None yet"

    if mode == "hr":
        # Avoid themes whose first two words already appear in prior questions.
        def _theme_used(theme, prev):
            head = " ".join(theme.split()[:2]).lower()
            return any(head in q.lower() for q in prev)
        unused_themes = [t for t in HR_THEMES if not _theme_used(t, previous_questions)]
        pool = unused_themes if unused_themes else HR_THEMES
        new_theme = random.choice(pool)

        system_prompt = HR_ADAPTIVE_SYSTEM
        user_msg = (
            f"Difficulty: {difficulty}\n"
            f"Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}\n"
            f"{performance_hint}\n\n"
            f"NEW THEME for this question (pivot here): {new_theme}\n\n"
            f"Generate the question now."
        )

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
        skills_str = ", ".join(skills_list) if skills_list else "Full Stack Development"

        system_prompt = RESUME_ADAPTIVE_SYSTEM
        user_msg = (
            f"Context: Resume-Based Interview — Candidate Skills: {skills_str}\n"
            f"Difficulty: {difficulty}\n"
            f"Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}\n"
            f"{performance_hint}\n\n"
            f"FOCUS THIS QUESTION ON SKILL: {next_skill}\n"
            f"ANGLE: {next_angle}\n"
            f"Confidence: {fused_confidence}/100\n\n"
            f"Generate the question now."
        )

    else:
        system_prompt = CUSTOM_ADAPTIVE_SYSTEM
        user_msg = (
            f"Context: Technical Interview — Technology: {tech}\n"
            f"Difficulty: {difficulty}\n"
            f"Previous questions (DO NOT REPEAT OR REPHRASE): {prev_q_str}\n"
            f"{performance_hint}\n\n"
            f"Stay strictly within {tech}. Confidence: {fused_confidence}/100\n\n"
            f"Generate the question now."
        )

    # Mode-specific fallback questions — never let the user get stuck.
    if mode == "hr":
        fallback = "Tell me about a time you had to adapt to an unexpected change at work."
    elif mode == "resume":
        fallback = "Walk me through a technical decision you made on a recent project and the trade-offs involved."
    else:
        fallback = f"Explain a core architectural concept in {tech or 'your stack'} and why it matters."

    # Try once, retry once on bad shape (Claude sometimes returns the wrong JSON
    # schema — accepting either form via _normalize_adaptive_payload). If both
    # attempts fail, return a safe fallback so the interview never deadlocks.
    for attempt in range(2):
        try:
            text = _call_claude(system_prompt, user_msg, temperature=0.9)
            parsed = _extract_json(text)
            return _normalize_adaptive_payload(parsed, difficulty)
        except Exception as e:
            print(f"⚠️  Adaptive question attempt {attempt + 1}/2 failed: {e}")

    print("❌ Both attempts failed — returning fallback question to keep interview moving.")
    return {"question": fallback, "difficulty": difficulty}
