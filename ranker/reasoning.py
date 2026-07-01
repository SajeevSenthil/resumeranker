"""
Deterministic, template-driven reasoning for each ranked candidate.
Every sentence is derived from observable profile fields and computed scores
— no hallucination possible, no LLM dependency.
"""

from datetime import date

_TODAY = date.today()


def _days_since(date_str: str) -> int:
    try:
        return (_TODAY - date.fromisoformat(date_str)).days
    except (ValueError, TypeError):
        return 999


# ---------------------------------------------------------------------------
# Sentence builders
# ---------------------------------------------------------------------------

def _who_sentence(meta: dict, scores: dict) -> str:
    title = meta["title"]
    yoe = meta["yoe"]
    role_score = scores["role"]
    company_score = scores["company"]
    skill_score = scores["skills"]
    top_skills = meta["top_skills"][:3]
    skill_str = ", ".join(top_skills) if top_skills else "no listed skills"

    # Role framing
    if role_score >= 0.85:
        role_phrase = f"{title} with {yoe:.0f} years of production ML/AI experience"
    elif role_score >= 0.55:
        role_phrase = f"{title} with {yoe:.0f} years of technical experience"
    else:
        role_phrase = f"{title} with {yoe:.0f} years of experience"

    # Company context
    if company_score >= 0.72:
        company_phrase = "at product-focused companies"
    elif company_score >= 0.45:
        company_phrase = "across product and services companies"
    elif company_score >= 0.25:
        company_phrase = "primarily in non-tech or services roles"
    else:
        company_phrase = "entirely in IT-services roles"

    # Skill alignment
    if skill_score >= 0.55:
        skill_phrase = f"strong match on retrieval/ranking skills ({skill_str})"
    elif skill_score >= 0.28:
        skill_phrase = f"partial skill alignment ({skill_str})"
    else:
        skill_phrase = f"limited overlap with required skills ({skill_str})"

    return f"{role_phrase} {company_phrase}; {skill_phrase}."


def _availability_sentence(meta: dict, scores: dict) -> str:
    behavior_score = scores["behavior"]
    open_to_work = meta["open_to_work"]
    notice = meta["notice_period"]
    response_rate = meta["response_rate"]
    days_inactive = _days_since(meta["last_active"])

    parts: list[str] = []

    # Availability intent
    if open_to_work:
        if notice <= 15:
            parts.append(f"immediately available ({notice}-day notice)")
        elif notice <= 30:
            parts.append("available within 30 days")
        elif notice <= 60:
            parts.append(f"{notice}-day notice period")
        else:
            parts.append(f"long notice period ({notice} days)")
    else:
        if days_inactive > 180:
            parts.append(f"not actively seeking ({days_inactive} days since last activity)")
        elif days_inactive > 60:
            parts.append("passively open, low recent activity")
        else:
            parts.append("not marked open-to-work but recently active")

    # Engagement signal
    if response_rate >= 0.70:
        parts.append(f"high recruiter responsiveness ({response_rate:.0%})")
    elif response_rate <= 0.20:
        parts.append(f"low recruiter responsiveness ({response_rate:.0%})")

    if not parts:
        return "Availability unclear."

    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reasoning(meta: dict, scores: dict) -> str:
    who = _who_sentence(meta, scores)
    availability = _availability_sentence(meta, scores)
    return f"{who} {availability}"
