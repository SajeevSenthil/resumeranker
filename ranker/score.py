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
    Role score acts as a multiplicative gate over a quality score.
    A Civil Engineer (role=0.02) can never exceed 0.02 regardless of how
    good their company/behavior/semantic looks — additive rescuing is gone.
    """
    w = SCORE_WEIGHTS
    quality = (
        feature_matrix[:, _COL_COMPANY] * w["company"]
        + feature_matrix[:, _COL_SKILLS] * w["skills"]
        + feature_matrix[:, _COL_BEHAVIOR] * w["behavior"]
        + semantic_scores * w["semantic"]
    )
    return feature_matrix[:, _COL_ROLE] * quality


def component_scores(feature_row: np.ndarray) -> dict[str, float]:
    """Return named component scores for a single candidate (for reasoning)."""
    return {
        "role": float(feature_row[_COL_ROLE]),
        "company": float(feature_row[_COL_COMPANY]),
        "skills": float(feature_row[_COL_SKILLS]),
        "behavior": float(feature_row[_COL_BEHAVIOR]),
    }
