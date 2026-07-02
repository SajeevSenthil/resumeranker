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

# Fix: gradio_client 0.x (bundled with gradio 5.0.0) doesn't handle
# `additionalProperties: true` (a JSON boolean, not a schema dict) which
# gradio generates for Dataframe / File component schemas.  Patch get_type
# so it returns "Any" instead of crashing with
#   TypeError: argument of type 'bool' is not iterable
import gradio_client.utils as _gcu
_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

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

def run_ranker(candidates_file, jd_text):
    if candidates_file is None:
        return None, None, "Upload a candidates JSON or JSONL file to get started."
    if not jd_text or not jd_text.strip():
        return None, None, "Paste a job description in the text box."

    try:
        candidates = load_candidates(Path(candidates_file))

        if not candidates:
            return None, None, "No candidates found in the uploaded file."
        if len(candidates) > 200:
            return None, None, (
                f"This demo accepts up to 200 candidates; your file has {len(candidates)}. "
                "Trim to a smaller sample and re-upload."
            )

        df = _rank_in_memory(candidates, jd_text.strip())

        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", encoding="utf-8"
        )
        df.to_csv(tmp.name, index=False, encoding="utf-8")
        tmp.close()

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

_CSS = """
body, .gradio-container { background: #0d1117 !important; }

.hdr {
    background: linear-gradient(135deg, #0d1117 0%, #1e3a5f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 22px 28px 18px;
    margin-bottom: 18px;
}
.hdr h1  { color: #f1f5f9; font-size: 1.5rem; font-weight: 700; margin: 0 0 6px; }
.hdr p   { color: #94a3b8; font-size: 0.8rem; margin: 0; line-height: 1.6; }
.badge   {
    display: inline-block;
    background: rgba(37,99,235,0.25);
    border: 1px solid #2563eb;
    color: #93c5fd;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    padding: 2px 10px; border-radius: 999px; margin-bottom: 10px;
}

.run-btn button {
    background: #2563eb !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    height: 46px !important;
    width: 100% !important;
}
.run-btn button:hover { background: #1d4ed8 !important; }
"""

with gr.Blocks(
    title="Redrob Candidate Ranker — Team REALM",
    theme=gr.themes.Monochrome(),
    css=_CSS,
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hdr">
      <div class="badge">TEAM REALM</div>
      <h1>Redrob Candidate Ranker</h1>
      <p>
        Upload a candidates file (JSON / JSONL, up to 200) and enter the job description.
        &nbsp;·&nbsp; CPU-only &nbsp;·&nbsp; No LLM calls &nbsp;·&nbsp; Results in seconds.
      </p>
    </div>
    """)

    # ── Main layout ─────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        with gr.Column(scale=1, min_width=300):
            candidates_input = gr.File(
                label="Candidates file (.json or .jsonl)",
                file_types=[".json", ".jsonl"],
            )
            jd_input = gr.Textbox(
                label="Job description",
                lines=18,
                placeholder="Type your JD here …",
                show_label=True,
            )
            run_btn = gr.Button(
                "Rank Candidates",
                variant="primary",
                size="lg",
                elem_classes="run-btn",
            )
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                label="Ranked candidates",
                headers=["candidate_id", "rank", "score", "reasoning"],
                wrap=True,
                interactive=False,
            )
            download_file = gr.File(label="Download CSV")

    run_btn.click(
        fn=run_ranker,
        inputs=[candidates_input, jd_input],
        outputs=[output_table, download_file, status_box],
    )

if __name__ == "__main__":
    demo.launch()
