# AI Resume Screener

> A machine learning-powered tool that analyzes resumes against job descriptions and gives you actionable feedback.

Ever wonder why your resume isn't getting callbacks? This tool breaks down exactly where you stand — match score, missing skills, and what to improve. Built with semantic understanding, not just keyword matching.

---

## What It Does

- **Match Score (0–100)** — quantifies how well your resume aligns with a job posting
- **Skill Analysis** — identifies matched and missing skills between resume and job description
- **Resume Feature Detection** — surfaces education, experience, projects, and certifications
- **Job Requirement Mapping** — maps each requirement to your resume with a match percentage
- **Improvement Suggestions** — gives specific, actionable advice to level up your resume

---

## How It Works

The scoring engine combines multiple signals:

| Component | Weight | Description |
|-----------|--------|-------------|
| Semantic Similarity | 30% | How well your resume's meaning matches the job |
| Skill Semantic Match | 25% | Semantic similarity of extracted skills |
| Exact Skill Overlap | 30% | Direct keyword matches |
| Semantic Bonus | 10% | Skills that match by meaning, not just exact keywords |
| Experience Bonus | 5% | Inferred from detected years of experience |

Powered by `all-MiniLM-L6-v2` sentence embeddings for true contextual understanding.

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **ML/NLP:** sentence-transformers, PyTorch, NumPy, NLTK
- **API Schema:** Pydantic
- **Frontend:** Vanilla HTML/CSS/JS (dark mode UI)

---

## Project Structure

```
.
├── main.py              # FastAPI routes + frontend serving
├── ml_engine.py         # Core analysis engine (skill extraction, semantic matching, scoring)
├── requirements.txt     # Python dependencies
├── README.md
└── static/
    └── index.html       # Single-page web interface
```

---

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:KhizarDoingProgramming/Ai-Resume-Screener.git
cd Ai-Resume-Screener
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> NLTK resources (punkt, stopwords, wordnet) are downloaded automatically at runtime if missing.

---

## Running

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/** in your browser.

---

## API Reference

### `POST /api/analyze`

Analyzes a resume against a job description.

**Request Body:**

```json
{
  "resume_text": "Your resume content...",
  "job_description": "Job posting text..."
}
```

**Response:**

```json
{
  "match_score": 72.5,
  "matched_skills": ["python", "fastapi", "docker"],
  "missing_skills": ["kubernetes", "terraform"],
  "key_resume_features": ["Technical skills identified: python, fastapi", "Experience: 3+ years"],
  "job_requirement_mapping": [
    {
      "requirement": "Experience with cloud platforms",
      "importance": "high",
      "resume_match": 45.2
    }
  ],
  "resume_analysis_summary": "Moderate match (72%). 3/8 skills aligned. Semantic similarity: 68%.",
  "improvement_suggestions": [
    "Add or highlight these missing skills: kubernetes, terraform",
    "Add quantifiable achievements (e.g., 'Reduced latency by 40%')"
  ]
}
```

### `GET /api/health`

Returns service health status.

```json
{ "status": "healthy", "version": "1.0.0" }
```

---

## Tips for Best Results

- Paste full resume sections — Skills, Work Experience, Projects, Education
- Include the complete job description, not just the title
- The more detail you provide, the more accurate the analysis

---

## Known Limitations

- Requires manual copy-paste (no file upload yet)
- Long inputs increase processing time
- Accuracy depends on resume/job description quality

---

## Roadmap

- [ ] PDF/DOCX file upload support
- [ ] Configurable skill taxonomy
- [ ] Embedding caching for faster analysis
- [ ] Analysis history and tracking

---

## License

MIT — do whatever you want with it.
