# Redrob Candidate Ranker

Submission for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**.
Ranks the top 100 candidates from a 100K-resume pool against a job description.

- CPU-only, no internet or API calls during ranking
- Runs in under 30 seconds on a 16 GB machine
- No LLM inference — deterministic reasoning from profile fields

---

## How it works

Two-phase pipeline:

**Offline (run once)** — Reads every candidate from `candidates.jsonl`, extracts four structured
feature scores and a sentence-transformer embedding, saves everything to `data/`.
Takes a few minutes on CPU for 100K candidates.

**Online (per JD, <30s)** — Loads precomputed artifacts, encodes the JD, scores all candidates
with a single numpy matmul and weighted combine, selects the top 100, and writes the CSV.

### Scoring

Role score acts as a **multiplicative gate**. A candidate irrelevant to the JD (e.g. Civil Engineer,
role\_score ≈ 0.02) cannot be rescued by strong company or skill signals.

```
final_score = role_score × (company×0.30 + skills×0.25 + behavior×0.25 + semantic×0.20)
```

| Component | Role in formula | What it measures |
|---|---|---|
| Role trajectory | Multiplicative gate | Title pattern + career description ML keyword density |
| Company type | 0.30 | Product-company vs IT-services background, tenure-weighted |
| Skill depth | 0.25 | Proficiency × duration × endorsements on 9 JD-relevant skill groups |
| Behavioral readiness | 0.25 | Open-to-work flag, notice period, platform activity, responsiveness |
| Semantic similarity | 0.20 | Cosine similarity of candidate embedding vs JD embedding |

**Honeypot detection** is applied before scoring. Candidates with ≥5 expert/advanced skills
at `duration_months=0`, or with role tenures that exceed the actual date span, receive a
`0.05` multiplier on role\_score — effectively removing them from contention.

Reasoning for each candidate is template-generated from observable profile fields — no LLM, no hallucination.

---

## Project structure

```
resumeranker/
├── config.py                   score weights, artifact paths, embedding model
├── build_offline.py            CLI: precompute artifacts from candidates.jsonl
├── rank.py                     CLI: online ranking against a JD
├── requirements.txt
├── submission_metadata.yaml    required for submission portal
├── data/                       generated artifacts (gitignored; .gitkeep tracks the folder)
│   ├── embeddings.npy          float32 (N, 384) — L2-normalised candidate embeddings
│   ├── feature_matrix.npy      float32 (N, 4)  — [role, company, skills, behavior]
│   ├── candidate_ids.pkl       ordered list of candidate IDs
│   └── metadata.pkl            profile snapshots used for reasoning
├── offline/
│   ├── parse.py                JSONL and JSON array reader
│   ├── embed.py                candidate text builder + SentenceTransformer encoder
│   ├── features.py             structured scorers + honeypot detection
│   └── build_artifacts.py      orchestrates the offline phase
└── ranker/
    ├── ontology.py             title patterns, skill groups, industry classifications
    ├── score.py                multiplicative score combination
    ├── reasoning.py            template-based reasoning strings
    └── pipeline.py             online ranking pipeline
```

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

---

## Reproducing the submission

### Step 1 — Extract the job description

```bash
pip install python-docx

python -c "
from docx import Document
doc = Document('path/to/job_description.docx')
text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
open('jd.txt', 'w', encoding='utf-8').write(text)
"
```

### Step 2 — Build artifacts (offline precomputation)

```bash
python build_offline.py path/to/candidates.jsonl
```

Accepts both `.jsonl` (one JSON object per line) and `.json` (array) formats.
Runtime: approximately 8–12 minutes on a 16 GB CPU machine for 100K candidates.
Outputs four files to `data/`.

### Step 3 — Rank (produces the submission CSV)

```bash
python rank.py jd.txt team_YOUR_ID.csv
```

Runtime: under 30 seconds. Output matches the submission spec exactly:
`candidate_id, rank, score, reasoning` — 100 data rows, UTF-8, scores non-increasing.

### Step 4 — Validate before uploading

```bash
python path/to/validate_submission.py team_YOUR_ID.csv
```

---

## Compute environment

| Property | Value |
|---|---|
| CPU | Any x86-64 or ARM |
| RAM | 16 GB minimum |
| GPU | Not used |
| Network during ranking | None |
| Python | 3.10+ |
| Offline precomputation | Required (Step 2 above) |
| Ranking runtime | < 30 seconds |

---

## Dependencies

```
sentence-transformers>=2.7.0
numpy>=1.26.0
pandas>=2.2.0
tqdm>=4.66.0
```
