import numpy as np

from config import SCORE_WEIGHTS

# Feature matrix column indices — must match offline/features.py FEATURE_NAMES
_COL_ROLE = 0
_COL_COMPANY = 1
_COL_SKILLS = 2
_COL_BEHAVIOR = 3


def combine_scores(
    feature_matrix: np.ndarray,
    semantic_scores: np.ndarray,
) -> np.ndarray:
    """
    Weighted sum of the five scoring components.
    All inputs are in [0, 1]; output is in [0, 1].
    Weights are read from config so they can be tuned in one place.
    """
    w = SCORE_WEIGHTS
    return (
        feature_matrix[:, _COL_ROLE] * w["role"]
        + feature_matrix[:, _COL_COMPANY] * w["company"]
        + feature_matrix[:, _COL_SKILLS] * w["skills"]
        + feature_matrix[:, _COL_BEHAVIOR] * w["behavior"]
        + semantic_scores * w["semantic"]
    )


def component_scores(feature_row: np.ndarray) -> dict[str, float]:
    """Return named component scores for a single candidate (for reasoning)."""
    return {
        "role": float(feature_row[_COL_ROLE]),
        "company": float(feature_row[_COL_COMPANY]),
        "skills": float(feature_row[_COL_SKILLS]),
        "behavior": float(feature_row[_COL_BEHAVIOR]),
    }
