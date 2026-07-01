from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256

# Role is a multiplicative gate; these quality weights sum to 1.0.
# final_score = role_score * (company*w + skills*w + behavior*w + semantic*w)
SCORE_WEIGHTS = {
    "company": 0.30,
    "skills": 0.25,
    "behavior": 0.25,
    "semantic": 0.20,
}

TOP_N = 100

ARTIFACT_PATHS = {
    "embeddings": DATA_DIR / "embeddings.npy",
    "features": DATA_DIR / "feature_matrix.npy",
    "candidate_ids": DATA_DIR / "candidate_ids.pkl",
    "metadata": DATA_DIR / "metadata.pkl",
}
