"""
HuggingFace Space entry point for the Redrob candidate ranker.

Upload a small candidates file (JSON or JSONL, up to 200 candidates) and
paste the job description. The full pipeline runs in-memory — no precomputed
artifacts needed for the demo.
"""

import os
import tempfile
from pathlib import Path

# gradio 5.0.0 imports HfFolder from huggingface_hub, but it was removed in
# huggingface_hub >= 0.37. Restore a minimal shim before gradio loads so the
# import doesn't crash. Only needed when the attribute is missing.
import huggingface_hub as _hfh
if not hasattr(_hfh, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token():
            return _hfh.get_token()
        @staticmethod
        def save_token(token: str) -> None:
            _hfh.login(token=token)
    _hfh.HfFolder = _HfFolder

import gradio as gr
import numpy as np
import pandas as pd
import torch

# Use all available CPU cores for PyTorch inference
_n_threads = os.cpu_count() or 4
torch.set_num_threads(_n_threads)
torch.set_num_interop_threads(_n_threads)

from config import TOP_N
from offline.embed import build_candidate_text, encode, load_model
from offline.features import build_feature_matrix, extract_metadata
from offline.parse import load_candidates
from ranker.reasoning import generate_reasoning
from ranker.score import combine_scores, component_scores

# ---------------------------------------------------------------------------
# Load the embedding model once at Space startup.
# all-MiniLM-L6-v2 (~80 MB) is pulled from HuggingFace Hub on first cold start
# and cached automatically by sentence-transformers.
# ---------------------------------------------------------------------------
print("Loading embedding model (all-MiniLM-L6-v2) ...")
_MODEL = load_model()
print("Model ready.")

# ---------------------------------------------------------------------------
# The actual job description for this challenge — pre-filled in the UI
# so judges can test immediately without copying text.
# ---------------------------------------------------------------------------
_DEFAULT_JD = """\
Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities
Experience Required: 5–9 years

What you'd actually be doing:
Own the intelligence layer — ranking, retrieval, and matching systems that decide what
recruiters see when they search for candidates. Ship a v2 ranking system with embeddings,
hybrid retrieval, and LLM-based re-ranking. Set up evaluation infrastructure (offline
benchmarks, A/B testing, recruiter-feedback loops).

Things you absolutely need:
- Production experience with embeddings-based retrieval (sentence-transformers, BGE, E5, or similar)
- Production experience with vector databases or hybrid search (Pinecone, Weaviate, Qdrant, Milvus,
  OpenSearch, Elasticsearch, FAISS)
- Strong Python
- Hands-on experience designing evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing)

Nice to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost or neural)
- Prior HR-tech / recruiting-tech exposure

We explicitly do NOT want:
- Candidates whose entire career has been at IT-services firms (TCS, Infosys, Wipro, Accenture, etc.)
- Computer vision / speech specialists without NLP/IR background
- Pure research backgrounds with no production deployment

Notice period: sub-30-day preferred; buyout up to 30 days. 30+ days considered but bar is higher.
"""


# ---------------------------------------------------------------------------
# Core ranking logic (in-memory, no disk artifacts)
# ---------------------------------------------------------------------------

def _rank_in_memory(candidates: list[dict], jd_text: str) -> pd.DataFrame:
    feature_matrix = build_feature_matrix(candidates)
    metadata = {c["candidate_id"]: extract_metadata(c) for c in candidates}
    candidate_ids = [c["candidate_id"] for c in candidates]

    texts = [build_candidate_text(c) for c in candidates]
    embeddings = encode(texts, _MODEL, show_progress=False)

    jd_vec = encode([jd_text], _MODEL, show_progress=False)[0]

    raw = (embeddings @ jd_vec).astype(np.float32)
    sem_scores = (raw + 1.0) / 2.0

    final_scores = combine_scores(feature_matrix, sem_scores)

    k = min(TOP_N, len(candidates))
    top_idx = np.argpartition(final_scores, -k)[-k:]
    top_idx = top_idx[np.argsort(final_scores[top_idx])[::-1]]

    rows = []
    for rank_pos, idx in enumerate(top_idx, start=1):
        cid = candidate_ids[idx]
        meta = metadata[cid]
        comp = component_scores(feature_matrix[idx])
        rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(float(final_scores[idx]), 4),
            "reasoning": generate_reasoning(meta, comp),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def run_ranker(candidates_file, jd_text, progress=gr.Progress(track_tqdm=True)):
    if candidates_file is None:
        return None, None, "Upload a candidates JSON or JSONL file to get started."
    if not jd_text or not jd_text.strip():
        return None, None, "Paste a job description in the text box."

    try:
        progress(0.05, desc="Parsing candidates ...")
        candidates = load_candidates(Path(candidates_file))

        if not candidates:
            return None, None, "No candidates found in the uploaded file."
        if len(candidates) > 200:
            return None, None, (
                f"This demo accepts up to 200 candidates; your file has {len(candidates)}. "
                "Trim to a smaller sample and re-upload."
            )

        progress(0.20, desc=f"Extracting features for {len(candidates)} candidates ...")
        progress(0.50, desc="Computing embeddings ...")
        df = _rank_in_memory(candidates, jd_text.strip())

        progress(0.90, desc="Writing output CSV ...")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", encoding="utf-8"
        )
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        tmp.close()

        progress(1.0, desc="Done.")
        top = df.iloc[0]
        status = (
            f"Ranked {len(df)} candidates. "
            f"Top result: {top['candidate_id']} — score {top['score']:.4f}"
        )
        return df, tmp.name, status

    except Exception as exc:
        return None, None, f"Error: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Redrob Candidate Ranker — Team REALM",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
# Redrob Candidate Ranker
**Team REALM** · Redrob Intelligent Candidate Discovery & Ranking Challenge

Two-phase CPU-only pipeline. Upload a candidate file (JSON or JSONL, ≤200 candidates)
and paste a job description — the ranker scores and ranks them in seconds.

**Scoring formula:** `final = role_score × (company×0.30 + skills×0.25 + behavior×0.25 + semantic×0.20)`

| Signal | What it measures |
|--------|-----------------|
| Role (gate) | Title pattern + ML keyword density in career descriptions |
| Company | Product-company vs IT-services background, tenure-weighted |
| Skills | Proficiency × log(duration) × log(endorsements) across 9 JD-relevant groups |
| Behavior | Open-to-work flag, notice period, platform activity, recruiter responsiveness |
| Semantic | Cosine similarity of `all-MiniLM-L6-v2` embedding vs JD |

Honeypot detection penalises impossible profiles (expert skills at 0 duration, tenure > date span).
Reasoning is template-generated from observable fields — no LLM called during ranking.
""")

    with gr.Row():
        with gr.Column(scale=1):
            candidates_input = gr.File(
                label="Candidates file (.json or .jsonl, up to 200 candidates)",
                file_types=[".json", ".jsonl"],
            )
            jd_input = gr.Textbox(
                label="Job description",
                lines=14,
                placeholder="Paste the full job description here ...",
                value=_DEFAULT_JD,
            )
            run_btn = gr.Button("Rank Candidates", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, lines=2)

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                label="Ranked candidates",
                headers=["candidate_id", "rank", "score", "reasoning"],
                wrap=True,
                interactive=False,
            )
            download_file = gr.File(label="Download ranked CSV")

    run_btn.click(
        fn=run_ranker,
        inputs=[candidates_input, jd_input],
        outputs=[output_table, download_file, status_box],
    )

    gr.Markdown("""
---
**Runtime**: < 10 seconds for 100 candidates on CPU &nbsp;·&nbsp;
**Model**: `all-MiniLM-L6-v2` (384-dim, ~80 MB, cached on first load) &nbsp;·&nbsp;
**No GPU required** &nbsp;·&nbsp;
**No API calls**
""")

if __name__ == "__main__":
    demo.launch()
