# AI Resume Screener

> no cap this thing actually slaps

y'all ever look at a job posting and think "there's literally no way my resume covers all that"?? yeah same. so i built this.

paste your resume + a job description and this ML-powered tool tells you **exactly** how cooked you are (or aren't). it's giving main character energy for your job hunt.

---

## what it does

- gives you a **match score (0-100)** so you know where you stand
- shows you **matched skills** you already got and the **missing ones** you need to cook up
- detects **key resume features** like education, experience, projects, certs
- maps **job requirements** to your resume so you know what's actually important
- gives **improvement suggestions** because we all need that push sometimes

built with **FastAPI** + **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic matching. basically it understands the *meaning* of your resume, not just keywords. big brain energy.

---

## features

- keyword + n-gram **skill extraction** from both resume and job description
- **semantic similarity** scoring using sentence embeddings (it actually understands context)
- match breakdown using:
  - overall semantic similarity
  - skill semantic match
  - exact overlap
  - semantic bonus (for skills that match by meaning, not just exact words)
  - experience signal (years detected)
- clean web UI at `/` (single-page frontend, dark mode because we're not cavemen)
- JSON API endpoint for programmatic use

---

## tech stack

| layer | tech |
|-------|------|
| backend | FastAPI + Uvicorn |
| ML/NLP | sentence-transformers, torch, numpy, nltk |
| API schema | Pydantic |
| frontend | Vanilla HTML/CSS/JS (served from `static/`) |

---

## project structure

```
.
├── main.py              # API routes + mounts the frontend
├── ml_engine.py         # core resume analysis logic (the brain)
├── requirements.txt     # dependencies
├── README.md            # you're reading this rn
└── static/
    └── index.html       # UI for pasting resume + job desc
```

---

## setup

### 1) create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) install dependencies

```bash
pip install -r requirements.txt
```

### 3) NLTK downloads

`ml_engine.py` uses NLTK resources (punkt, stopwords, wordnet) and will download them automatically at runtime if missing. no manual setup needed fr.

---

## run it

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

then hit **http://localhost:8000/** and vibe.

---

## API usage

### health check

```bash
curl http://localhost:8000/api/health
```

### analyze a resume

**endpoint:** `POST /api/analyze`

**body:**
- `resume_text` (string) — paste your resume
- `job_description` (string) — paste the job posting

**example:**

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Your resume text here...",
    "job_description": "Job description text here..."
  }'
```

**response fields:**

| field | type | what it means |
|-------|------|---------------|
| `match_score` | number | how well you match (0-100) |
| `matched_skills` | string[] | skills you got that they want |
| `missing_skills` | string[] | skills you're missing |
| `key_resume_features` | string[] | detected features (education, exp, etc) |
| `job_requirement_mapping` | array | requirements mapped to your resume |
| `resume_analysis_summary` | string | tl;dr of your analysis |
| `improvement_suggestions` | string[] | how to level up your resume |

---

## how the match score works

the score is a weighted combo of:

- **semantic similarity** (30%) — how well your resume *meaning* matches the job
- **skill semantic match** (25%) — semantic similarity of extracted skills
- **exact skill overlap** (30%) — direct keyword matches
- **semantic bonus** (10%) — skills that match by meaning, not exact words
- **experience bonus** (5%) — inferred from detected years of experience

final output is clamped to **0-100** so you never see weird numbers.

---

## limitations & tips

- accuracy depends on how well you paste your resume/job text
  - include sections like **Skills, Work Experience, Projects, Education**
- long inputs can increase compute time (semantic embeddings are expensive)
- paste the **full job description** and relevant resume sections for best results
- this is a screening tool, not a recruiter replacement (yet)

---

## future improvements

- [ ] PDF/DOCX file upload support (no more copy-paste grind)
- [ ] expanded skills taxonomy or user-defined skill sets
- [ ] cache embeddings for better performance
- [ ] persistence/history so you can track your progress over time

---

## license

nothing just clone it and use it. go get that bag.
