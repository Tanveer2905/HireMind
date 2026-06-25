<p align="center">
  <img src="static/favicon.png" alt="HireMind Logo" width="120" height="120" style="border-radius: 50%;">
</p>

<h1 align="center">🧠 HireMind</h1>

<p align="center">
  <strong>AI-Powered Hiring Copilot with LLM Reasoning, Semantic Search & Explainable Rankings</strong>
</p>

<p align="center">
  <em>Local-First · Fully Offline · Multi-Tenant · Free & Open Source</em>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-license">License</a>
</p>

---

## ✨ Features

### 🔍 Intelligent Candidate Ranking
- **Semantic Search** — Uses BGE embedding models + FAISS vector indexing to understand resume–JD fit beyond simple keyword matching
- **Composite Scoring** — Multi-factor scoring formula combining 5 dimensions:
  | Factor | Weight | Description |
  |--------|--------|-------------|
  | Semantic Similarity | 35% | Dense vector cosine similarity via FAISS |
  | Skill Match | 25% | Taxonomy-aware skill matching (~500 skills with aliases) |
  | Experience Relevance | 20% | Sigmoid-based scoring against JD requirements |
  | Recency | 10% | Exponential decay favoring recent experience |
  | Keyword Precision | 10% | Top-30 JD keyword coverage in resume |
- **Hard Filters** — Define must-have skills that automatically reject candidates who don't match
- **Skill Ontology** — Rich taxonomy of ~500 skills with alias resolution (e.g., `js` → `JavaScript`, `k8s` → `Kubernetes`) and category enrichment (e.g., `PyTorch` → also matches `Deep Learning`, `Machine Learning`)

### 🧠 LLM-Powered AI Reasoning
- **Embedded Local LLM** — Runs LLaMA 3 8B (Q4 quantized, ~4.7GB) directly in-process via `llama-cpp-python` — no external API calls, no cloud dependency
- **Deep Candidate Evaluation** — Evidence-based analysis covering skill match quality, experience relevance, project impact, and red flags
- **Structured Decisions** — Each candidate gets a `Strong Yes / Yes / Maybe / No` verdict with specific reasoning
- **Algorithmic Fallback** — When the LLM is unavailable (OOM, hardware mismatch), an evidence-based algorithmic evaluator produces differentiated analysis from structured data
- **Interview Question Generation** — Auto-generates targeted interview questions based on candidate gaps and strengths

### 💬 Conversational AI Copilot (Chat)
- Natural language queries like *"Find Python developers with 3+ years"* or *"Who has both React and Node.js?"*
- LLM-powered query parsing with rule-based fallback
- Supports filters on skills, experience ranges, titles, and sorting

### 📊 Feedback Learning System
- Record recruiter shortlist/reject decisions per candidate
- Trains a lightweight LightGBM classifier (fallback: Logistic Regression) after 20+ feedback samples
- Adjusts ranking scores by ±10% based on learned preferences
- Persists model to disk for cross-session learning

### 🏢 Multi-Tenant Architecture (Backend v2)
- **JWT Authentication** — User registration, login, and token-based auth via FastAPI + bcrypt
- **Per-User Data Isolation** — Each user gets their own resumes directory, FAISS index, feedback history, and preferences
- **Memory Engine** — Tracks per-user shortlisting patterns to build preference profiles
- **Personalization Layer** — Boosts/penalizes candidates based on learned recruiter preferences without requiring ML training

### 🎨 Premium Glassmorphism UI
- Modern glassmorphism design with animated gradient orbs
- Drag-and-drop PDF resume upload
- Interactive candidate cards with expandable detail modals
- Real-time stats bar, skill chip tags, and Excel/CSV export
- AI Chat drawer for conversational querying
- Auth modal with login/register flow
- Fully responsive layout

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        GLASSMORPHISM WEB UI                       │
│               (HTML + CSS + Vanilla JS, index.html)               │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ REST API
┌─────────────────────────────▼──────────────────────────────────────┐
│                    FastAPI / Flask Backend                         │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────────┐ │
│  │  Auth    │ │  Resume  │ │  Analyze  │ │  Chat / Feedback     │ │
│  │  (JWT)   │ │  Upload  │ │  Pipeline │ │  Endpoints           │ │
│  └──────────┘ └──────────┘ └─────┬─────┘ └──────────────────────┘ │
└──────────────────────────────────┼────────────────────────────────-┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
  ┌───────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │   Resume Parser   │ │  Embedding Engine │ │   LLM Engine     │
  │  (pdfplumber +    │ │  (BGE + FAISS)   │ │  (LLaMA 3 8B     │
  │   spaCy NER)      │ │                  │ │   via llama.cpp)  │
  └─────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
            │                    │                     │
            ▼                    ▼                     ▼
  ┌───────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │  Skill Taxonomy   │ │  Composite       │ │  LLM Reranker    │
  │  (~500 skills,    │ │  Scorer          │ │  + Algorithmic    │
  │  alias resolution)│ │  (5-factor)      │ │  Fallback         │
  └───────────────────┘ └────────┬─────────┘ └──────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
          ┌──────────────┐ ┌─────────┐ ┌───────────────┐
          │  Feedback    │ │ Memory  │ │ Personalization│
          │  Engine      │ │ Engine  │ │ Layer          │
          │  (LightGBM)  │ │ (prefs) │ │ (score adjust) │
          └──────────────┘ └─────────┘ └───────────────┘
```

### Pipeline Flow

1. **Ingest** — Upload PDF resumes via drag-and-drop or file picker
2. **Parse** — Extract text with `pdfplumber`, identify skills/experience/education with spaCy + rule-based extraction
3. **Embed** — Generate dense vectors using BGE-base-en-v1.5 (768-dim), cached to disk
4. **Index** — Build FAISS `IndexFlatIP` for cosine similarity search
5. **Score** — Composite 5-factor ranking with hard filter support
6. **Rerank** *(optional)* — LLM-powered deep evaluation with structured reasoning
7. **Personalize** — Apply learned recruiter preferences from feedback history
8. **Export** — Download results as CSV or Excel with candidate IDs and reasoning

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **~6GB disk space** (for models: LLaMA 3 ~4.7GB, BGE ~440MB, spaCy ~12MB)
- **8GB+ RAM** recommended (16GB for LLM features)
- *(Optional)* NVIDIA GPU with CUDA for faster LLM inference

### One-Command Setup (Windows)

```bash
# Clone the repository
git clone https://github.com/your-username/HireMind.git
cd HireMind/ai_recruiter

# Run the automated setup script
setup.bat
```

The setup script will:
1. Create a Python virtual environment
2. Install all dependencies from `requirements.txt`
3. Download all AI models (~5GB total):
   - `BAAI/bge-base-en-v1.5` — Embedding model (440MB)
   - `en_core_web_sm` — spaCy NLP model (12MB)
   - `Meta-Llama-3-8B-Instruct.Q4_0.gguf` — Local LLM (4.7GB)

### Manual Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install llama-cpp-python with CUDA support (optional)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118

# Download models
python download_models.py
```

### Running the Application

```bash
# Place resume PDFs in the resumes/ folder, then:

# Option 1: Web UI (recommended)
run.bat
# Opens at http://localhost:5000

# Option 2: CLI mode
python main.py
# Interactive terminal-based analysis

# Option 3: Direct backend launch
python -m backend.main
# FastAPI multi-tenant server on port 5000
```

### Docker Deployment

```bash
docker build -t hiremind .
docker run -p 7860:7860 hiremind
```

> **Note:** The Docker image pre-downloads all models during the build phase (~5GB). The resulting container is fully self-contained and works offline.

---

## ⚙️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Backend Framework** | Flask (v1) / FastAPI (v2 multi-tenant) |
| **Embedding Model** | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (768-dim) |
| **Vector Search** | FAISS (IndexFlatIP, cosine similarity) |
| **Local LLM** | LLaMA 3 8B Instruct (Q4_0 GGUF via llama-cpp-python) |
| **NLP** | spaCy (en_core_web_sm) + rule-based extraction |
| **PDF Parsing** | pdfplumber |
| **Feedback ML** | LightGBM / Scikit-learn Logistic Regression |
| **Auth** | JWT (PyJWT) + bcrypt password hashing |
| **Database** | SQLite (user management) |
| **Frontend** | Vanilla JS + CSS (glassmorphism design) |
| **Containerization** | Docker |

---

## 📡 API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/register` | Register a new user (`email`, `password`) |
| `POST` | `/api/login` | Login and receive JWT token |

### Resume Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/resumes` | List all uploaded resumes |
| `POST` | `/api/upload` | Upload PDF resumes (multipart form) |
| `DELETE` | `/api/resumes/<filename>` | Delete a specific resume |

### Analysis Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Run the full ranking pipeline |
| `GET` | `/api/export/excel` | Export results as Excel spreadsheet |

**Analyze Request Body:**
```json
{
  "job_description": "We are looking for a Senior Python Developer...",
  "must_have_skills": ["Python", "Docker"],
  "use_llm_rerank": true
}
```

### Chat & Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a natural language query |
| `POST` | `/api/feedback` | Record shortlist/reject decision |
| `GET` | `/api/feedback/stats` | Get feedback statistics |
| `POST` | `/api/feedback/reset` | Reset all feedback data |
| `POST` | `/api/candidate/<filename>/interview-questions` | Generate targeted interview questions |

### System Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/llm/status` | Check LLM availability and backend info |

---

## 📁 Project Structure

```
ai_recruiter/
├── app.py                    # Flask web server (single-tenant mode)
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container config
├── setup.bat                 # One-click Windows setup
├── run.bat                   # One-click Windows launcher
│
├── backend/                  # FastAPI multi-tenant backend (v2)
│   ├── main.py               #   FastAPI app with auth-protected routes
│   ├── auth.py               #   JWT authentication + bcrypt
│   ├── database.py           #   SQLite user management
│   ├── user_context.py       #   Per-user directory isolation
│   ├── faiss_manager.py      #   Multi-tenant FAISS index management
│   ├── parser.py             #   Multi-tenant resume parsing
│   ├── scorer.py             #   Multi-tenant scoring layer
│   ├── reranker.py           #   LLM reranking with user preferences
│   ├── llm_engine.py         #   Per-user LLM engine wrapper
│   ├── memory_engine.py      #   Feedback tracking & preference learning
│   └── personalization.py    #   Score adjustments from learned prefs
│
├── embeddings.py             # Embedding engine (BGE + FAISS)
├── scorer.py                 # 5-factor composite scoring
├── parser.py                 # Resume/JD parsing (pdfplumber + spaCy)
├── llm_client.py             # Embedded LLaMA client (llama-cpp-python)
├── llm_ranker.py             # LLM reranking + algorithmic fallback
├── chat_engine.py            # Conversational query interface
├── feedback.py               # Feedback learning (LightGBM)
├── utils.py                  # Skill taxonomy, normalization, helpers
├── download_models.py        # Model download script
│
├── templates/
│   └── index.html            # Main UI (glassmorphism design)
├── static/
│   ├── css/
│   │   ├── style-premium.css #   Premium glassmorphism styles
│   │   └── style.css         #   Base styles
│   ├── js/
│   │   └── app.js            #   Frontend application logic
│   └── favicon.png           #   App icon
│
├── models/                   # Downloaded AI models (gitignored)
│   ├── bge-base-en-v1.5/     #   BGE embedding model
│   ├── en_core_web_sm/       #   spaCy NLP model
│   └── llama3-8b-instruct-q4_0.gguf  # LLaMA 3 GGUF
│
├── resumes/                  # Upload PDF resumes here (gitignored)
├── data/                     # Runtime data, caches, feedback (gitignored)
│   ├── parsed_cache.json     #   Parsed resume cache
│   ├── embedding_cache.npz   #   Embedding vector cache
│   ├── feedback.json         #   Recruiter feedback history
│   ├── skill_ontology.json   #   Skill category ontology
│   ├── users.db              #   SQLite user database
│   └── results.csv           #   Latest ranking export
│
└── sample_candidates.json    # Sample candidate data for testing
```

---

## 🧮 Scoring Formula

Each candidate is scored on a 0–1 scale using a weighted composite formula:

```
final_score = 0.35 × semantic_similarity
            + 0.25 × skill_match
            + 0.20 × experience_relevance
            + 0.10 × recency
            + 0.10 × keyword_precision
```

**Sub-score Details:**

| Sub-score | Method | Range |
|-----------|--------|-------|
| **Semantic Similarity** | FAISS inner product on L2-normalized BGE embeddings | `[0, 1]` |
| **Skill Match** | Jaccard-like overlap with enriched skill categories + bonus for extra relevant skills | `[0, 1]` |
| **Experience Relevance** | Sigmoid function centered on the JD requirement with sweet spot at 80–150% | `[0, 1]` |
| **Recency** | Exponential decay `e^(-0.15 × years_ago)` averaged over top-3 most recent year mentions | `[0, 1]` |
| **Keyword Precision** | Fraction of top-30 meaningful JD keywords found in resume | `[0, 1]` |

After composite scoring, an optional **feedback adjustment** of ±10% is applied based on learned recruiter preferences (requires 20+ feedback samples).

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using local-first AI
</p>
