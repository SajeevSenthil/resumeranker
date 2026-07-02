"""
Offline preprocessing pipeline.
Run once on the full candidates.jsonl; outputs four artifact files to data/.

Artifacts:
  embeddings.npy       float32 (N, 384) — L2-normalised candidate embeddings
  feature_matrix.npy   float32 (N, 4)  — [role, company, skills, behavior]
  candidate_ids.pkl    list[str]        — ordered candidate IDs (index -> id)
  metadata.pkl         dict[str, dict]  — id -> profile snapshot for reasoning
"""

import pickle
import time
from pathlib import Path

import os

import numpy as np
import torch

from config import ARTIFACT_PATHS, DATA_DIR
from offline.parse import load_candidates
from offline.features import build_feature_matrix, extract_metadata
from offline.embed import build_candidate_text, encode, load_model


def build(candidates_path: Path) -> None:
    # Use all available CPU cores for PyTorch inference.
    # Default is often 1-4; on a 16-core machine this can cut embedding time by 4-8x.
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(n_threads)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"Loading candidates from {candidates_path} ...")
    candidates = load_candidates(candidates_path)
    n = len(candidates)
    print(f"  {n:,} candidates loaded  ({time.perf_counter() - t0:.1f}s)")

    print("Extracting candidate IDs ...")
    candidate_ids = [c["candidate_id"] for c in candidates]

    print("Building feature matrix ...")
    feature_matrix = build_feature_matrix(candidates)
    print(f"  shape: {feature_matrix.shape}  ({time.perf_counter() - t0:.1f}s)")

    print("Extracting metadata ...")
    metadata = {c["candidate_id"]: extract_metadata(c) for c in candidates}

    print("Building candidate texts ...")
    texts = [build_candidate_text(c) for c in candidates]

    print("Generating embeddings (may take several minutes on CPU) ...")
    model = load_model()
    embeddings = encode(texts, model)
    print(f"  shape: {embeddings.shape}  ({time.perf_counter() - t0:.1f}s)")

    print("Saving artifacts ...")
    np.save(ARTIFACT_PATHS["embeddings"], embeddings)
    np.save(ARTIFACT_PATHS["features"], feature_matrix)
    with open(ARTIFACT_PATHS["candidate_ids"], "wb") as f:
        pickle.dump(candidate_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ARTIFACT_PATHS["metadata"], "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    emb_mb = embeddings.nbytes / 1e6
    feat_mb = feature_matrix.nbytes / 1e6
    elapsed = time.perf_counter() - t0
    print(
        f"\nArtifacts saved to {DATA_DIR}\n"
        f"  embeddings.npy    {emb_mb:.1f} MB\n"
        f"  feature_matrix.npy {feat_mb:.1f} MB\n"
        f"  candidate_ids.pkl\n"
        f"  metadata.pkl\n"
        f"Total time: {elapsed:.1f}s"
    )
