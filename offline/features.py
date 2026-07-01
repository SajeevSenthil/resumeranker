"""
Structured feature extraction for each candidate.
Produces a float32 vector: [role_score, company_score, skill_score, behavior_score]

All scores are in [0, 1]. Higher = better fit for the JD.
"""

import math
import re
from datetime import date
from typing import Optional

import numpy as np

from ranker.ontology import (
    EDUCATION_TIER_SCORES,
    IT_SERVICES_INDUSTRIES,
    NON_TECH_INDUSTRY_SCORE,
    PROFICIENCY_MULTIPLIERS,
    PRODUCT_INDUSTRY_SCORES,
    SIZE_SCORES,
    SKILL_GROUP_WEIGHTS,
    SKILL_GROUPS,
    SKILL_TO_GROUP,
    TITLE_PATTERNS,
    FALLBACK_TITLE_SCORE,
)

FEATURE_NAMES = ["role", "company", "skills", "behavior"]
N_FEATURES = len(FEATURE_NAMES)

_TODAY = date.today()

# ML/IR keywords expected in actual role descriptions (not skills list).
# Presence here means the candidate used these things at work — harder to fake.
_ML_CAREER_KEYWORDS: frozenset[str] = frozenset({
    "retrieval", "ranking", "recommendation", "embedding", "embeddings",
    "vector", "semantic search", "similarity", "dense retrieval",
    "machine learning", "deep learning", "neural", "transformer",
    "fine-tuning", "fine-tune", "llm", "language model",
    "nlp", "natural language", "text classification",
    "faiss", "qdrant", "milvus", "pinecone", "weaviate", "elasticsearch",
    "opensearch", "chromadb", "pgvector",
    "ndcg", "mrr", "a/b test", "evaluation framework", "offline evaluation",
    "feature store", "model serving", "inference", "model deployment",
    "pytorch", "tensorflow", "hugging face", "scikit-learn", "xgboost",
    "rag", "retrieval augmented", "reranking", "re-ranking", "two-tower",
    "candidate generation", "bert", "gpt", "lora", "qlora",
    "learning to rank", "learning-to-rank",
})


def _career_ml_signal(candidate: dict) -> float:
    """
    Fraction of ML/IR keywords found across all role description text.
    Scores evidence of actual ML work, independent of the skills list.
    Capped at 15 unique hits for a full score.
    """
    descriptions = " ".join(
        r.get("description", "").lower() for r in candidate["career_history"]
    )
    hits = sum(1 for kw in _ML_CAREER_KEYWORDS if kw in descriptions)
    return min(hits / 15.0, 1.0)


# ---------------------------------------------------------------------------
# Role score
# Blends title-pattern matching with career-description ML evidence.
# Title alone is gameable (people add "AI" to any title); descriptions less so.
# ---------------------------------------------------------------------------

def _match_title(title: str) -> float:
    t = title.strip()
    for pattern, score in TITLE_PATTERNS:
        if pattern.search(t):
            return score
    return FALLBACK_TITLE_SCORE


def _role_score(candidate: dict) -> float:
    profile = candidate["profile"]
    career = candidate["career_history"]

    total_months = sum(r["duration_months"] for r in career) or 1
    career_weighted = sum(_match_title(r["title"]) * r["duration_months"] for r in career)
    career_avg = career_weighted / total_months

    current_score = _match_title(profile["current_title"])
    title_score = career_avg * 0.45 + current_score * 0.55

    # Career description evidence: harder to fake than title or skills list
    desc_signal = _career_ml_signal(candidate)

    return title_score * 0.65 + desc_signal * 0.35


# ---------------------------------------------------------------------------
# Company score
# Rewards product-company and startup experience; penalizes pure IT-services
# backgrounds. Computed as a tenure-weighted average across all roles.
# ---------------------------------------------------------------------------

def _company_role_score(role: dict) -> float:
    industry = role.get("industry", "")
    size = role.get("company_size", "")

    if industry in IT_SERVICES_INDUSTRIES:
        # JD explicitly flags pure IT-services careers; still nonzero so
        # blending with product-company roles produces a fair weighted average
        return 0.08

    industry_score = PRODUCT_INDUSTRY_SCORES.get(industry, NON_TECH_INDUSTRY_SCORE)
    size_score = SIZE_SCORES.get(size, 0.50)

    return industry_score * 0.60 + size_score * 0.40


def _company_score(candidate: dict) -> float:
    career = candidate["career_history"]
    total_months = sum(r["duration_months"] for r in career) or 1
    weighted = sum(_company_role_score(r) * r["duration_months"] for r in career)
    return weighted / total_months


# ---------------------------------------------------------------------------
# Skill score
# Depth-weighted matching against JD skill groups.
# Rewards proficiency + sustained use + peer endorsement; not just presence.
# ---------------------------------------------------------------------------

def _skill_depth(skill: dict) -> float:
    prof = PROFICIENCY_MULTIPLIERS.get(skill.get("proficiency", "beginner"), 0.20)
    # log scale so 48 months (4yr) doesn't overshadow 12 months by 4x
    dur = min(math.log(skill.get("duration_months", 1) + 1) / math.log(49), 1.0)
    # endorsements are social proof; use log to dampen outliers
    end = min(math.log(skill.get("endorsements", 0) + 2) / math.log(52), 1.0)
    return prof * 0.55 + dur * 0.28 + end * 0.17


def _resolve_group(skill_name: str) -> Optional[str]:
    """Return the ontology group for a skill name, or None if unrecognised."""
    lower = skill_name.lower().strip()
    if lower in SKILL_TO_GROUP:
        return SKILL_TO_GROUP[lower]
    # Substring check: handles "Sentence Transformers (SBERT)" etc.
    for member, group in SKILL_TO_GROUP.items():
        if member in lower or lower in member:
            return group
    return None


def _skill_score(candidate: dict) -> float:
    skills = candidate.get("skills", [])

    # Accumulate the best depth score per group
    group_best: dict[str, float] = {g: 0.0 for g in SKILL_GROUPS}
    for skill in skills:
        group = _resolve_group(skill["name"])
        if group is not None:
            depth = _skill_depth(skill)
            if depth > group_best[group]:
                group_best[group] = depth

    total_weight = sum(SKILL_GROUP_WEIGHTS.values())
    weighted_sum = sum(
        SKILL_GROUP_WEIGHTS[g] * group_best[g] for g in SKILL_GROUPS
    )
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Behavior score
# Hiring readiness: availability, engagement, and responsiveness signals.
# Directly reflects the JD's note that "a perfect-on-paper candidate who
# hasn't logged in for 6 months is not actually available."
# ---------------------------------------------------------------------------

def _days_since(date_str: str) -> int:
    try:
        return (_TODAY - date.fromisoformat(date_str)).days
    except (ValueError, TypeError):
        return 365


def _behavior_score(signals: dict) -> float:
    score = 0.0

    # Open-to-work is the strongest single availability signal
    if signals.get("open_to_work_flag"):
        score += 0.25

    # Notice period: JD explicitly prefers sub-30-day; offers buyout up to 30d
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:
        score += 0.20
    elif notice <= 30:
        score += 0.17
    elif notice <= 60:
        score += 0.10
    elif notice <= 90:
        score += 0.04
    # > 90 days contributes nothing

    # Recency of platform activity
    days_inactive = _days_since(signals.get("last_active_date", "2000-01-01"))
    if days_inactive <= 7:
        score += 0.20
    elif days_inactive <= 30:
        score += 0.16
    elif days_inactive <= 90:
        score += 0.10
    elif days_inactive <= 180:
        score += 0.04
    # > 180 days contributes nothing

    score += signals.get("recruiter_response_rate", 0.0) * 0.15
    score += signals.get("interview_completion_rate", 0.0) * 0.10

    # Active job seeker signal; cap at 10 applications to avoid gaming
    apps = min(signals.get("applications_submitted_30d", 0), 10) / 10.0
    score += apps * 0.10

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Top-level extractors
# ---------------------------------------------------------------------------

def extract_features(candidate: dict) -> np.ndarray:
    signals = candidate["redrob_signals"]
    return np.array(
        [
            _role_score(candidate),
            _company_score(candidate),
            _skill_score(candidate),
            _behavior_score(signals),
        ],
        dtype=np.float32,
    )


def extract_metadata(candidate: dict) -> dict:
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])

    # JD-relevant skills: those that matched the ontology, ranked by depth.
    # Used in reasoning so the explanation cites actual matching skills,
    # not globally popular ones like TTS or Speech Recognition.
    jd_skills = sorted(
        [s for s in skills if _resolve_group(s["name"]) is not None],
        key=_skill_depth,
        reverse=True,
    )[:4]

    # Fallback: if no JD-relevant skills found, use globally top skills
    if not jd_skills:
        jd_skills = sorted(
            skills,
            key=lambda s: (
                {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(
                    s["proficiency"], 0
                ),
                s.get("endorsements", 0),
            ),
            reverse=True,
        )[:3]

    best_edu_score = max(
        (EDUCATION_TIER_SCORES.get(e.get("tier", "unknown"), 0.35) for e in education),
        default=0.35,
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "title": profile["current_title"],
        "company": profile["current_company"],
        "industry": profile["current_industry"],
        "company_size": profile["current_company_size"],
        "yoe": profile["years_of_experience"],
        "top_skills": [s["name"] for s in jd_skills],
        "open_to_work": signals["open_to_work_flag"],
        "notice_period": signals["notice_period_days"],
        "response_rate": signals["recruiter_response_rate"],
        "last_active": signals["last_active_date"],
        "github_score": signals.get("github_activity_score", -1),
        "interview_completion": signals.get("interview_completion_rate", 0.0),
        "education_score": best_edu_score,
    }


def build_feature_matrix(candidates: list[dict]) -> np.ndarray:
    matrix = np.zeros((len(candidates), N_FEATURES), dtype=np.float32)
    for i, c in enumerate(candidates):
        matrix[i] = extract_features(c)
    return matrix
