"""
Online ranking pipeline.

Expected runtime on 100K candidates (CPU-only):
  - artifact load   ~8 s
  - JD embedding    ~1 s
  - numpy matmul    ~5 s   (100K x 384 dot product)
  - scoring + sort  ~3 s
  - reasoning + csv ~3 s
  Total             < 25 s — well within the 5-minute constraint
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import ARTIFACT_PATHS, EMBEDDING_MODEL, SCORE_WEIGHTS, TOP_N
from ranker.score import combine_scores, component_scores
from ranker.reasoning import generate_reasoning


def _load_artifacts() -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    embeddings = np.load(ARTIFACT_PATHS["embeddings"], mmap_mode="r")
    features = np.load(ARTIFACT_PATHS["features"])
    with open(ARTIFACT_PATHS["candidate_ids"], "rb") as f:
        candidate_ids: list[str] = pickle.load(f)
    with open(ARTIFACT_PATHS["metadata"], "rb") as f:
        metadata: dict = pickle.load(f)
    return embeddings, features, candidate_ids, metadata


def _embed_jd(jd_text: str) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    vec = model.encode(
        [jd_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0].astype(np.float32)


def _semantic_scores(embeddings: np.ndarray, jd_vec: np.ndarray) -> np.ndarray:
    # Embeddings are L2-normalised, so dot product == cosine similarity in [-1, 1].
    # Shift to [0, 1] so it blends cleanly with the structured scores.
    raw = (embeddings @ jd_vec).astype(np.float32)
    return (raw + 1.0) / 2.0


def rank(jd_text: str, output_path: Path) -> None:
    t0 = time.perf_counter()

    print("Loading precomputed artifacts ...")
    embeddings, features, candidate_ids, metadata = _load_artifacts()
    n = len(candidate_ids)
    print(f"  {n:,} candidates loaded  ({time.perf_counter() - t0:.1f}s)")

    print("Embedding job description ...")
    jd_vec = _embed_jd(jd_text)

    print("Computing semantic similarity ...")
    sem_scores = _semantic_scores(embeddings, jd_vec)

    print("Computing final scores ...")
    final_scores = combine_scores(features, sem_scores)

    k = min(TOP_N, n)
    print(f"Selecting top {k} (of {n:,} candidates) ...")
    # argpartition is O(n); avoids sorting all 100K
    top_idx = np.argpartition(final_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(final_scores[top_idx])[::-1]]

    print("Generating reasoning ...")
    rows = []
    for rank_pos, idx in enumerate(top_idx, start=1):
        cid = candidate_ids[idx]
        score = float(final_scores[idx])
        meta = metadata[cid]
        comp = component_scores(features[idx])
        reasoning = generate_reasoning(meta, comp)
        rows.append(
            {
                "candidate_id": cid,
                "rank": rank_pos,
                "score": round(score, 4),
                "reasoning": reasoning,
            }
        )

    df = pd.DataFrame(rows)

    # Enforce submission rule: non-increasing score; tie-break by candidate_id asc
    df = df.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    df.to_csv(output_path, index=False, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"Done. Written {len(df)} rows to {output_path}  ({elapsed:.1f}s total)")
