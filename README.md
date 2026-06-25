# HireMind

**Intelligent candidate ranking system with semantic search, LLM-powered evaluation, and recruiter feedback learning.**

HireMind is a local-first hiring copilot that ranks resumes against job descriptions using a combination of dense vector similarity, structured skill matching, and optional LLM reasoning. Everything runs offline — no API keys, no cloud dependencies, no candidate data leaving your machine.

---

## What it does

You drop in a stack of PDF resumes and a job description. HireMind parses each resume, extracts skills and experience using NLP, generates embeddings, and ranks candidates using a weighted composite score across five dimensions:

| Factor | Weight | How it works |
|--------|--------|-------------|
| Semantic similarity | 35% | Cosine similarity between resume and JD embeddings (BGE + FAISS) |
| Skill match | 25% | Taxonomy-aware matching with ~500 skill aliases and category enrichment |
| Experience relevance | 20% | Sigmoid curve centered on JD requirements, penalizes under-qualification |
| Recency | 10% | Exponential decay on year mentions — recent work scores higher |
| Keyword precision | 10% | Coverage of top-30 meaningful JD keywords in the resume |

On top of that, you can optionally enable LLM reranking, which sends the top candidates through a local LLaMA 3 8B model for evidence-based evaluation — structured reasoning, strengths/weaknesses, red flags, and a `Strong Yes / Yes / Maybe / No` verdict per candidate.

When the LLM is unavailable (not enough RAM, no GPU, model not downloaded), the system falls back to an algorithmic evaluator that still produces candidate-specific analysis from the parsed data. Nothing breaks — LLM features are entirely opt-in.

---

## Key capabilities

**Ranking pipeline** — Parses PDFs with pdfplumber, extracts entities with spaCy, generates 768-dim embeddings with BGE-base-en-v1.5, indexes with FAISS, and applies composite scoring with configurable hard filters (must-have skills that auto-reject non-matching candidates).

**LLM reasoning** — Runs LLaMA 3 8B Instruct (Q4 quantized, ~4.7GB) in-process via llama-cpp-python. Evaluates candidates on skill match quality, experience depth, project impact, and red flags. Generates targeted interview questions per candidate.

**Conversational search** — A chat interface that translates natural language queries (e.g., "Find Python developers with 3+ years" or "Who has both React and Docker?") into structured filters. Uses LLM for query parsing when available, falls back to regex-based extraction.

**Feedback learning** — Records recruiter shortlist/reject decisions and trains a lightweight LightGBM classifier after 20+ samples. The model adjusts ranking scores by up to ±10% based on learned preferences. Persists across sessions.

**Multi-tenant backend** — The FastAPI backend (v2) supports JWT auth, per-user data isolation, and a memory engine that tracks individual recruiter preferences over time. Each user gets their own resume store, FAISS index, feedback history, and personalization profile.

**Export** — Results export to CSV and Excel with candidate IDs, scores, ranks, and reasoning.

---

## Getting started

### Requirements

- Python 3.10+ (3.11 recommended)
- ~6GB disk space for models
- 8GB RAM minimum, 16GB recommended for LLM features
- NVIDIA GPU optional (speeds up LLM inference)

### Setup (Windows)

```bash
git clone https://github.com/your-username/HireMind.git
cd HireMind/ai_recruiter
setup.bat
```

This creates a virtual environment, installs dependencies, and downloads all models:
- **BGE-base-en-v1.5** — Sentence embedding model (~440MB)
- **en_core_web_sm** — spaCy NLP model (~12MB)
- **LLaMA 3 8B Instruct Q4** — Local LLM (~4.7GB)

### Manual setup

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Optional: CUDA-accelerated llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

python download_models.py
```

### Running

```bash
# Web UI — opens at http://localhost:5000
run.bat

# CLI mode — interactive terminal
python main.py

# FastAPI multi-tenant server
python -m backend.main
```

### Docker

```bash
docker build -t hiremind .
docker run -p 7860:7860 hiremind
```

Models are downloaded during the Docker build, so the container is fully self-contained.

---

## Architecture

```
                          Web UI (HTML/CSS/JS)
                                |
                           REST API
                                |
                 FastAPI / Flask Application Layer
                    |            |            |
               Auth (JWT)   Resume Mgmt   Analysis Pipeline
                                              |
                 +----------------------------+----------------------------+
                 |                            |                            |
          Resume Parser                Embedding Engine              LLM Engine
       (pdfplumber + spaCy)          (BGE + FAISS index)        (LLaMA 3 via llama.cpp)
                 |                            |                            |
          Skill Taxonomy              Composite Scorer              LLM Reranker
       (~500 skills, aliases,          (5-factor weighted)        (+ algorithmic fallback)
        category enrichment)                  |
                                    +---------+---------+
                                    |         |         |
                               Feedback   Memory    Personalization
                                Engine    Engine       Layer
                              (LightGBM) (per-user)  (score adjust)
```

The pipeline runs in stages: **parse → embed → index → score → rerank → personalize → export**. Each stage caches its output — parsed resume data, embeddings, and LLM evaluations are all persisted to disk, so re-runs on the same data are fast.

---

## Project structure

```
ai_recruiter/
├── app.py                     # Flask server (single-tenant mode)
├── main.py                    # CLI entry point
├── requirements.txt
├── Dockerfile
├── setup.bat / run.bat        # Windows setup and launch scripts
│
├── backend/                   # FastAPI multi-tenant backend
│   ├── main.py                #   App with auth-protected routes
│   ├── auth.py                #   JWT + bcrypt authentication
│   ├── database.py            #   SQLite user store
│   ├── user_context.py        #   Per-user directory isolation
│   ├── faiss_manager.py       #   Multi-tenant FAISS management
│   ├── parser.py              #   Tenant-scoped resume parsing
│   ├── scorer.py              #   Tenant-scoped scoring
│   ├── reranker.py            #   LLM reranking with user prefs
│   ├── llm_engine.py          #   Per-user LLM wrapper
│   ├── memory_engine.py       #   Feedback tracking and prefs
│   └── personalization.py     #   Score adjustment from prefs
│
├── embeddings.py              # BGE model loading + FAISS indexing
├── scorer.py                  # Composite scoring engine
├── parser.py                  # Resume/JD parsing
├── llm_client.py              # Embedded LLaMA client
├── llm_ranker.py              # LLM reranking + fallback evaluator
├── chat_engine.py             # Natural language query interface
├── feedback.py                # Feedback learning (LightGBM)
├── utils.py                   # Skill taxonomy, normalization, IO
├── download_models.py         # Model download automation
│
├── templates/index.html       # Main UI
├── static/                    # CSS, JS, favicon
├── models/                    # Downloaded AI models (gitignored)
├── resumes/                   # PDF input directory (gitignored)
└── data/                      # Caches, feedback, exports (gitignored)
```

---

## API endpoints

All endpoints under the multi-tenant backend require a Bearer JWT token (obtained via `/api/login`).

### Auth

- `POST /api/register` — Create account (email + password)
- `POST /api/login` — Get JWT access token

### Resumes

- `GET /api/resumes` — List uploaded resumes with metadata
- `POST /api/upload` — Upload PDF files (multipart)
- `DELETE /api/resumes/<filename>` — Remove a resume

### Analysis

- `POST /api/analyze` — Run the ranking pipeline. Body:
  ```json
  {
    "job_description": "...",
    "must_have_skills": ["Python", "Docker"],
    "use_llm_rerank": false
  }
  ```
- `GET /api/export/excel` — Download results as .xlsx

### Chat and feedback

- `POST /api/chat` — Natural language query (`{"message": "..."}`)
- `POST /api/feedback` — Record decision (`{"filename": "...", "action": "shortlisted"}`)
- `GET /api/feedback/stats` — Feedback counts and model status
- `POST /api/feedback/reset` — Clear all feedback data
- `POST /api/candidate/<filename>/interview-questions` — Generate interview questions

### Status

- `GET /api/llm/status` — LLM availability check

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (v1), FastAPI + Uvicorn (v2) |
| Embeddings | BAAI/bge-base-en-v1.5 via sentence-transformers |
| Vector search | FAISS (IndexFlatIP on L2-normalized vectors) |
| Local LLM | LLaMA 3 8B Q4_0 via llama-cpp-python |
| NLP | spaCy en_core_web_sm |
| PDF parsing | pdfplumber |
| Feedback ML | LightGBM (fallback: sklearn LogisticRegression) |
| Auth | PyJWT + bcrypt |
| User DB | SQLite |
| Frontend | Vanilla HTML/CSS/JS |
| Container | Docker (python:3.11-slim) |

---

## Contributing

Fork the repo, create a branch, make your changes, and open a pull request. If you're adding a new scoring factor or modifying the ranking formula, include before/after ranking comparisons on the sample data.

## License

MIT
