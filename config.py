from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 256

# Weights must sum to 1.0
SCORE_WEIGHTS = {
    "role": 0.25,
    "company": 0.25,
    "skills": 0.20,
    "behavior": 0.20,
    "semantic": 0.10,
}

TOP_N = 100

ARTIFACT_PATHS = {
    "embeddings": DATA_DIR / "embeddings.npy",
    "features": DATA_DIR / "feature_matrix.npy",
    "candidate_ids": DATA_DIR / "candidate_ids.pkl",
    "metadata": DATA_DIR / "metadata.pkl",
}
