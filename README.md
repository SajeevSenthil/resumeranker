# resumeranker

Candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.
Ranks top 100 candidates from a pool of 100,000 resumes against a job description.
Runs entirely on CPU, no internet or API calls during ranking.

## What this does

Two-phase pipeline:

**Offline** — run once on the candidate dataset. Extracts structured feature scores and
generates sentence-transformer embeddings for every candidate. Saves everything to disk.

**Online** — given a job description, loads the precomputed artifacts, scores all candidates
in a single numpy pass, and writes a ranked CSV in under 30 seconds.

### Scoring components

| Component | Weight | What it measures |
|---|---|---|
| Role trajectory | 25% | Career title alignment with ML/AI engineering |
| Company type | 25% | Product company vs IT-services background |
| Skill depth | 20% | Proficiency + duration + endorsements on JD-relevant skills |
| Behavioral readiness | 20% | Open-to-work, notice period, recruiter responsiveness, recency |
| Semantic similarity | 10% | Embedding cosine similarity of candidate profile vs JD |

Reasoning for each candidate is generated from templates using actual profile fields — no LLM involved.

## Project structure

```
resumeranker/
├── config.py                   score weights, artifact paths, embedding model
├── build_offline.py            CLI: precompute artifacts from candidates.jsonl
├── rank.py                     CLI: run online ranking against a JD
├── requirements.txt
├── data/                       generated artifacts (gitignored)
│   ├── embeddings.npy          float32 (N, 384) — L2-normalised embeddings
│   ├── feature_matrix.npy      float32 (N, 4)  — [role, company, skills, behavior]
│   ├── candidate_ids.pkl       ordered list of candidate IDs
│   └── metadata.pkl            profile snapshots used for reasoning
├── offline/
│   ├── parse.py                JSONL and JSON array reader
│   ├── embed.py                candidate text builder + encoder
│   ├── features.py             four structured scorers
│   └── build_artifacts.py      orchestrates the offline phase
└── ranker/
    ├── ontology.py             title patterns, skill groups, industry classifications
    ├── score.py                weighted combination of all components
    ├── reasoning.py            template-based reasoning strings
    └── pipeline.py             online ranking pipeline
```

## Setup

```bash
cd resumeranker
pip install -r requirements.txt
```

## Usage

**Step 1 — build artifacts** (run once, or whenever the candidate dataset changes):

```bash
python build_offline.py path/to/candidates.jsonl
```

Accepts both `.jsonl` (one object per line) and `.json` (array) formats.
Takes a few minutes on CPU for 100K candidates; outputs to `data/`.

**Step 2 — extract the job description** (requires `python-docx`):

```bash
pip install python-docx

python -c "
from docx import Document
doc = Document('path/to/job_description.docx')
text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
open('jd.txt', 'w', encoding='utf-8').write(text)
"
```

**Step 3 — rank**:

```bash
python rank.py jd.txt submission.csv
```

Output is a CSV with columns `candidate_id, rank, score, reasoning` — ready for submission validation.

**Validate before submitting:**

```bash
python "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" submission.csv
```
