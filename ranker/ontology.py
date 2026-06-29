"""
Domain knowledge for scoring candidates against this JD.
All sets use lowercase to match against lowercased input.
"""

import re

# ---------------------------------------------------------------------------
# Title scoring
# Each entry is (compiled_regex, score). Checked in order; first match wins.
# Score reflects alignment with "Senior AI Engineer - Founding Team" JD.
# ---------------------------------------------------------------------------

_TITLE_RAW: list[tuple[str, float]] = [
    # Senior/Lead ML-AI roles -- best fit
    (r"\b(senior|staff|principal|lead|head of|director of)\b.{0,25}\b(machine learning|ml|ai|nlp|search|ranking|retrieval|recommendation)\b", 1.00),
    # Mid-level ML-AI engineers
    (r"\b(machine learning|ml|ai|nlp|information retrieval|search ranking)\b.{0,15}\b(engineer|scientist|specialist)\b", 0.95),
    # Applied / research scientists (production-oriented)
    (r"\b(applied|research)\b.{0,10}\b(scientist|ml engineer|ai engineer)\b", 0.85),
    # General research scientist (JD warns: research-only is a disqualifier)
    (r"\bresearch scientist\b", 0.60),
    # NLP-specific
    (r"\bnlp\b.{0,10}\b(engineer|researcher|scientist)\b", 0.90),
    # Junior ML-AI (less desirable; JD targets senior judgment)
    (r"\bjunior\b.{0,20}\b(machine learning|ml|ai|nlp)\b", 0.55),
    # Data scientist
    (r"\bdata scientist\b", 0.75),
    # Data engineer (adjacent; infra experience is useful but not core)
    (r"\bdata engineer\b", 0.45),
    # Backend / software / platform engineer
    (r"\b(backend|software|platform|systems|infrastructure)\b.{0,10}\b(engineer|developer)\b", 0.30),
    # Computer vision / speech / robotics -- JD explicitly does not want
    (r"\b(computer vision|speech recognition|robotics|embedded systems)\b", 0.12),
    # Product / program / project management
    (r"\b(product manager|program manager|project manager|scrum master|delivery manager)\b", 0.08),
    # HR / talent / recruiting
    (r"\b(hr manager|human resources|talent acquisition|recruiter|people operations)\b", 0.02),
    # Marketing / content / sales / operations
    (r"\b(marketing|content writer|content manager|sales executive|operations manager|business development)\b", 0.02),
    # Finance / accounting
    (r"\b(accountant|finance manager|financial analyst|controller|auditor)\b", 0.02),
    # Mechanical / civil / electrical engineering
    (r"\b(mechanical|civil|electrical|chemical)\b.{0,5}\bengineer\b", 0.02),
    # Design
    (r"\b(graphic designer|ux designer|ui designer|product designer)\b", 0.03),
    # Customer support
    (r"\b(customer support|customer success|customer care|support engineer)\b", 0.03),
    # Business analyst
    (r"\bbusiness analyst\b", 0.07),
]

TITLE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(pattern, re.IGNORECASE), score)
    for pattern, score in _TITLE_RAW
]

FALLBACK_TITLE_SCORE = 0.05


# ---------------------------------------------------------------------------
# Industry scoring
# ---------------------------------------------------------------------------

# IT services model = consulting for clients; JD explicitly disqualifies
# candidates whose ENTIRE career is here.
IT_SERVICES_INDUSTRIES: frozenset[str] = frozenset({
    "IT Services",
    "IT Consulting",
    "Outsourcing",
    "Business Process Outsourcing",
    "BPO",
    "Professional Services",
    "Consulting",
})

# Product/tech industries -- high alignment with JD
PRODUCT_INDUSTRY_SCORES: dict[str, float] = {
    "Technology": 0.95,
    "Artificial Intelligence": 1.00,
    "Machine Learning": 1.00,
    "Data Analytics": 0.90,
    "E-Commerce": 0.90,
    "FinTech": 0.88,
    "Financial Technology": 0.88,
    "SaaS": 0.95,
    "B2B Software": 0.92,
    "Marketplace": 0.88,
    "EdTech": 0.80,
    "HealthTech": 0.80,
    "Gaming": 0.75,
    "Consumer Technology": 0.85,
    "Enterprise Software": 0.90,
    "Cybersecurity": 0.78,
    "Cloud Computing": 0.88,
    "Internet": 0.85,
    "Media Technology": 0.72,
    "AdTech": 0.72,
    "PropTech": 0.70,
    "Logistics Technology": 0.75,
}

# Non-tech industries -- partial credit; not ideal but possible career path
NON_TECH_INDUSTRY_SCORE = 0.30

# Company size as a product-company proxy (small = less likely to be services)
SIZE_SCORES: dict[str, float] = {
    "1-10": 0.90,
    "11-50": 0.92,
    "51-200": 0.88,
    "201-500": 0.80,
    "501-1000": 0.68,
    "1001-5000": 0.58,
    "5001-10000": 0.44,
    "10001+": 0.28,
}


# ---------------------------------------------------------------------------
# Skill groups and weights
# Grouped by JD priority. All skill names lowercase.
# ---------------------------------------------------------------------------

SKILL_GROUPS: dict[str, frozenset[str]] = {
    "vector_search": frozenset({
        "faiss", "qdrant", "milvus", "pinecone", "weaviate", "opensearch",
        "elasticsearch", "chroma", "chromadb", "pgvector", "vector search",
        "vector database", "vector db", "ann", "hnsw", "ivf",
        "approximate nearest neighbor", "semantic search index",
    }),
    "dense_retrieval": frozenset({
        "sentence transformers", "sentence-transformers", "bge", "e5",
        "dense retrieval", "bi-encoder", "cross-encoder", "embeddings",
        "semantic search", "dense embeddings", "openai embeddings",
        "text embeddings", "retrieval", "information retrieval",
        "neural retrieval", "dense passage retrieval", "dpr",
    }),
    "ranking": frozenset({
        "ranking", "learning-to-rank", "ltr", "lambdamart", "lambdarank",
        "listwise ranking", "pairwise ranking", "pointwise ranking",
        "learning to rank", "candidate ranking", "re-ranking", "reranking",
        "neural ranking", "cross-encoder reranking",
    }),
    "evaluation": frozenset({
        "ndcg", "mrr", "map", "precision@k", "recall@k", "hit rate",
        "a/b testing", "ab testing", "offline evaluation", "online evaluation",
        "ranking evaluation", "retrieval evaluation", "benchmark",
        "mean reciprocal rank", "normalized discounted cumulative gain",
        "evaluation framework", "ranking metrics",
    }),
    "recommendation": frozenset({
        "recommendation", "recommender", "recommender system",
        "collaborative filtering", "matrix factorization", "two-tower",
        "item2vec", "user2vec", "content-based filtering",
        "hybrid recommendation", "candidate generation",
    }),
    "hybrid_search": frozenset({
        "bm25", "tfidf", "tf-idf", "sparse retrieval", "hybrid search",
        "colbert", "splade", "rag", "retrieval augmented generation",
        "retrieval augmented", "hybrid retrieval", "sparse dense",
    }),
    "llm_fine_tuning": frozenset({
        "fine-tuning", "fine-tune", "lora", "qlora", "peft", "rlhf",
        "instruction tuning", "sft", "supervised fine-tuning",
        "llm fine-tuning", "parameter efficient fine-tuning",
        "llm", "large language model",
    }),
    "ml_infrastructure": frozenset({
        "mlflow", "weights & biases", "wandb", "kubeflow", "airflow",
        "feature store", "model serving", "triton", "torchserve", "bentoml",
        "ray", "ray serve", "mlops", "model registry", "experiment tracking",
    }),
    "python_ml": frozenset({
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "transformers", "hugging face", "huggingface", "jax", "numpy", "pandas",
        "scipy", "xgboost", "lightgbm",
    }),
}

# Weight per group based on JD priority
SKILL_GROUP_WEIGHTS: dict[str, float] = {
    "vector_search": 1.00,
    "dense_retrieval": 1.00,
    "ranking": 0.95,
    "evaluation": 0.90,
    "hybrid_search": 0.88,
    "recommendation": 0.80,
    "python_ml": 0.75,
    "llm_fine_tuning": 0.65,
    "ml_infrastructure": 0.60,
}

# Build reverse lookup: normalized skill name -> group name
# Used for O(1) exact matching
SKILL_TO_GROUP: dict[str, str] = {
    skill: group
    for group, members in SKILL_GROUPS.items()
    for skill in members
}

PROFICIENCY_MULTIPLIERS: dict[str, float] = {
    "beginner": 0.20,
    "intermediate": 0.50,
    "advanced": 0.85,
    "expert": 1.00,
}

EDUCATION_TIER_SCORES: dict[str, float] = {
    "tier_1": 1.00,
    "tier_2": 0.75,
    "tier_3": 0.45,
    "tier_4": 0.20,
    "unknown": 0.35,
}
