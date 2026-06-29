from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


def build_candidate_text(candidate: dict) -> str:
    profile = candidate["profile"]
    career = candidate["career_history"]
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])

    parts = [profile["headline"]]

    if profile.get("summary"):
        parts.append(profile["summary"][:400])

    # Most recent 3 roles, trimmed — older roles dilute the signal
    for role in sorted(career, key=lambda r: r["start_date"], reverse=True)[:3]:
        desc = role.get("description", "")[:220]
        if desc:
            parts.append(f"{role['title']} at {role['company']}: {desc}")

    top_skills = sorted(
        skills,
        key=lambda s: (
            {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}.get(
                s["proficiency"], 0
            ),
            s.get("endorsements", 0),
        ),
        reverse=True,
    )[:12]
    if top_skills:
        parts.append("Skills: " + ", ".join(s["name"] for s in top_skills))

    if certs:
        parts.append("Certifications: " + ", ".join(c["name"] for c in certs[:4]))

    return " ".join(parts)


def load_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
