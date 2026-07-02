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

    # Career descriptions only — what they actually did at work.
    # Skills list deliberately excluded: it is already captured in the
    # structured skill_score and including it here lets keyword stuffers
    # (e.g. a Frontend Engineer who lists FAISS) boost their semantic score.
    for role in sorted(career, key=lambda r: r["start_date"], reverse=True)[:4]:
        desc = role.get("description", "")[:250]
        if desc:
            parts.append(f"{role['title']} at {role['company']}: {desc}")

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
