import re
import math
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SKILL_DATABASE = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "ruby",
        "go", "golang", "rust", "php", "swift", "kotlin", "scala", "r",
        "matlab", "perl", "haskell", "elixir", "dart", "lua", "sql",
        "html", "css", "sass", "scss"
    ],
    "frameworks": [
        "react", "angular", "vue", "vue.js", "next.js", "nextjs", "nuxt",
        "django", "flask", "fastapi", "spring", "spring boot", "node.js",
        "nodejs", "express", "express.js", "rails", "ruby on rails",
        "laravel", "symfony", "dotnet", ".net", "asp.net", "tailwind",
        "bootstrap", "svelte", "remix", "astro"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "hugging face", "huggingface", "transformers", "bert", "gpt",
        "langchain", "llm", "large language model", "neural network",
        "cnn", "rnn", "lstm", "gan", "reinforcement learning",
        "xgboost", "random forest", "svm", "gradient boosting",
        "data science", "data analysis", "feature engineering",
        "model deployment", "mlops", "mlflow", "airflow"
    ],
    "cloud_devops": [
        "aws", "amazon web services", "azure", "microsoft azure",
        "gcp", "google cloud", "docker", "kubernetes", "k8s",
        "jenkins", "ci/cd", "terraform", "ansible", "puppet",
        "linux", "bash", "shell scripting", "nginx", "apache",
        "github actions", "gitlab ci", "circleci", "heroku", "vercel",
        "netlify", "cloudflare"
    ],
    "databases": [
        "sql", "mysql", "postgresql", "postgres", "mongodb", "nosql",
        "redis", "elasticsearch", "cassandra", "dynamodb", "sqlite",
        "oracle", "sql server", "firebase", "supabase", "neo4j",
        "couchdb", "mariadb", "clickhouse"
    ],
    "tools_misc": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "slack", "figma", "sketch", "adobe xd", "photoshop",
        "visual studio code", "vscode", "intellij", "eclipse",
        "postman", "swagger", "rest", "restful", "graphql", "grpc",
        "api", "microservices", "agile", "scrum", "kanban",
        "tdd", "test driven development", "bdd", "unit testing",
        "integration testing", "cypress", "selenium", "jest", "pytest",
        "junit", "mocha", "chai", "react testing library"
    ]
}

ALL_SKILLS = []
for category in SKILL_DATABASE.values():
    ALL_SKILLS.extend(category)
ALL_SKILLS = list(set(ALL_SKILLS))

SECTION_PATTERNS = {
    "skills": r"(?:skills?|technical\s+skills?|technologies|competencies|proficiencies)",
    "experience": r"(?:experience|work\s+experience|employment|work\s+history|professional\s+experience)",
    "education": r"(?:education|academic|qualification|degrees?)",
    "projects": r"(?:projects?|personal\s+projects?|key\s+projects?|portfolio)",
    "certifications": r"(?:certifications?|certificates?|licenses?)",
}

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


@dataclass
class ResumeAnalysis:
    match_score: float = 0.0
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    key_resume_features: List[str] = field(default_factory=list)
    job_requirement_mapping: List[Dict] = field(default_factory=list)
    resume_analysis_summary: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)


class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 1]
        return tokens

    def chunk_text(self, text: str, max_chunk_len: int = 200) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_len = 0
        for sent in sentences:
            words = len(sent.split())
            if current_len + words > max_chunk_len and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent]
                current_len = words
            else:
                current_chunk.append(sent)
                current_len += words
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks if chunks else [text]


class SkillExtractor:
    def __init__(self):
        self.text_processor = TextProcessor()

    def extract_skills(self, text: str) -> List[str]:
        found_skills = []
        text_lower = text.lower()

        for skill in ALL_SKILLS:
            pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
            if re.search(pattern, text_lower):
                found_skills.append(skill)

        tokens = self.text_processor.tokenize(text)
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]

        for ngram in bigrams + trigrams:
            for skill in ALL_SKILLS:
                if ngram == skill and skill not in found_skills:
                    found_skills.append(skill)

        return list(set(found_skills))

    def extract_experience_years(self, text: str) -> Optional[float]:
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s+)?experience',
            r'(\d+)\+?\s*yrs?\s*(?:of\s+)?experience',
            r'experience\s*:\s*(\d+)\+?\s*years?',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))
        return None


class SemanticEngine:
    def __init__(self):
        self.model = get_embedding_model()

    def compute_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def compute_chunk_similarities(self, resume_text: str, job_text: str) -> Dict[str, float]:
        resume_chunks = self.text_processor.chunk_text(resume_text) if hasattr(self, 'text_processor') else [resume_text]
        job_chunks = [job_text]

        resume_embeddings = self.model.encode(resume_chunks, convert_to_numpy=True)
        job_embeddings = self.model.encode(job_chunks, convert_to_numpy=True)

        similarities = []
        for r_emb in resume_embeddings:
            for j_emb in job_embeddings:
                sim = util.cos_sim(r_emb, j_emb).item()
                similarities.append(sim)

        return {
            "mean": float(np.mean(similarities)) if similarities else 0.0,
            "max": float(np.max(similarities)) if similarities else 0.0,
            "min": float(np.min(similarities)) if similarities else 0.0,
        }

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)
        return float(util.cos_sim(emb1, emb2).item())

    def compute_skill_semantic_match(self, resume_skills: List[str], job_skills: List[str]) -> float:
        if not resume_skills or not job_skills:
            return 0.0

        resume_text = ", ".join(resume_skills)
        job_text = ", ".join(job_skills)

        resume_emb = self.model.encode(resume_text, convert_to_numpy=True)
        job_emb = self.model.encode(job_text, convert_to_numpy=True)

        return float(util.cos_sim(resume_emb, job_emb).item())

    def find_semantic_matches(self, resume_text: str, job_skills: List[str], threshold: float = 0.35) -> List[str]:
        semantic_matches = []
        resume_emb = self.model.encode(resume_text, convert_to_numpy=True)

        for skill in job_skills:
            skill_emb = self.model.encode(skill, convert_to_numpy=True)
            sim = float(util.cos_sim(resume_emb, skill_emb).item())
            if sim >= threshold:
                semantic_matches.append(skill)

        return semantic_matches


class ResumeAnalyzer:
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.semantic_engine = SemanticEngine()
        self.text_processor = TextProcessor()

    def analyze(self, resume_text: str, job_description: str) -> ResumeAnalysis:
        resume_skills = self.skill_extractor.extract_skills(resume_text)
        job_skills = self.skill_extractor.extract_skills(job_description)

        exact_matched = list(set(resume_skills) & set(job_skills))
        exact_missing = list(set(job_skills) - set(resume_skills))

        semantic_matches = self.semantic_engine.find_semantic_matches(resume_text, exact_missing)
        all_matched = list(set(exact_matched + semantic_matches))
        still_missing = [s for s in exact_missing if s not in semantic_matches]

        overall_similarity = self.semantic_engine.compute_semantic_similarity(resume_text, job_description)
        skill_semantic_sim = self.semantic_engine.compute_skill_semantic_match(resume_skills, job_skills)

        exact_overlap = len(exact_matched) / len(job_skills) if job_skills else 0
        semantic_bonus = len(semantic_matches) / len(job_skills) if job_skills else 0

        experience_years = self.skill_extractor.extract_experience_years(resume_text)
        experience_bonus = 0.0
        if experience_years:
            if experience_years >= 5:
                experience_bonus = 1.0
            elif experience_years >= 3:
                experience_bonus = 0.6
            elif experience_years >= 1:
                experience_bonus = 0.3

        match_score = (
            overall_similarity * 30 +
            skill_semantic_sim * 25 +
            exact_overlap * 30 +
            semantic_bonus * 10 +
            experience_bonus * 5
        )
        match_score = min(max(match_score, 0), 100)

        job_requirements = self._extract_job_requirements(job_description, resume_text)
        key_features = self._extract_key_features(resume_text, resume_skills, experience_years)
        suggestions = self._generate_suggestions(all_matched, still_missing, resume_text, job_description, overall_similarity)
        summary = self._generate_summary(match_score, len(all_matched), len(still_missing), len(job_skills), overall_similarity)

        return ResumeAnalysis(
            match_score=round(match_score, 1),
            matched_skills=all_matched,
            missing_skills=still_missing,
            key_resume_features=key_features,
            job_requirement_mapping=job_requirements,
            resume_analysis_summary=summary,
            improvement_suggestions=suggestions
        )

    def _extract_job_requirements(self, job_description: str, resume_text: str) -> List[Dict]:
        requirements = []
        lines = job_description.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(r'^[\-\*\•\✓\►\→\d+\.\)]+\s*', line):
                clean = re.sub(r'^[\-\*\•\✓\►\→\d+\.\)]+\s*', '', line).strip()
                if len(clean) > 5:
                    sim = self.semantic_engine.compute_semantic_similarity(clean, resume_text)
                    requirements.append({
                        "requirement": clean[:120],
                        "importance": "high" if any(kw in clean.lower() for kw in ["must", "required", "essential", "minimum"]) else "medium",
                        "resume_match": round(sim * 100, 1)
                    })

        return sorted(requirements, key=lambda x: x["resume_match"], reverse=True)[:15]

    def _extract_key_features(self, resume_text: str, skills: List[str], experience_years: Optional[float]) -> List[str]:
        features = []
        if skills:
            features.append(f"Technical skills identified: {', '.join(skills[:12])}")

        if experience_years:
            features.append(f"Experience: {int(experience_years)}+ years")

        education_match = re.search(r'(?:bachelor|master|phd|b\.s\.|m\.s\.|b\.tech|m\.tech|degree)', resume_text.lower())
        if education_match:
            features.append("Education mentioned")

        projects = re.findall(r'(?:project|built|developed|created|implemented)', resume_text.lower())
        if projects:
            features.append(f"{len(projects)} project-related mentions found")

        certs = re.findall(r'(?:certified|certification|certificate)', resume_text.lower())
        if certs:
            features.append(f"{len(certs)} certification mentions")

        return features

    def _generate_suggestions(self, matched, missing, resume_text, job_description, similarity) -> List[str]:
        suggestions = []

        if missing:
            top_missing = missing[:5]
            suggestions.append(f"Add or highlight these missing skills: {', '.join(top_missing)}")

        if similarity < 0.3:
            suggestions.append("Resume content is semantically distant from job requirements — consider rewriting experience bullets to mirror job language")

        if not re.search(r'(?:quantified|reduced|increased|improved|achieved|\d+%|\d+\s*%)', resume_text.lower()):
            suggestions.append("Add quantifiable achievements (e.g., 'Reduced latency by 40%')")

        if not re.search(r'(?:github|portfolio|deployed|live demo)', resume_text.lower()):
            suggestions.append("Include links to GitHub, portfolio, or deployed projects")

        if len(matched) < 3:
            suggestions.append("Tailor resume keywords to match job description more closely")

        if not re.search(r'(?:certified|certification|aws certified|google cloud)', resume_text.lower()):
            if re.search(r'(?:aws|azure|gcp|cloud|docker|kubernetes)', job_description.lower()):
                suggestions.append("Consider adding relevant certifications (AWS, Azure, GCP)")

        if not re.search(r'(?:api|rest|microservice|backend|frontend)', resume_text.lower()):
            if re.search(r'(?:api|rest|microservice|backend|frontend)', job_description.lower()):
                suggestions.append("Highlight experience with APIs, microservices, or system design")

        if not suggestions:
            suggestions.append("Resume appears well-aligned with the job description")

        return suggestions

    def _generate_summary(self, score, matched_count, missing_count, total_job_skills, similarity) -> str:
        score_rounded = round(score)
        sim_pct = round(similarity * 100)

        if score_rounded >= 80:
            return f"Strong match ({score_rounded}%). {matched_count}/{total_job_skills} skills aligned. Semantic similarity: {sim_pct}%. Resume is well-suited for this role."
        elif score_rounded >= 60:
            return f"Moderate match ({score_rounded}%). {matched_count}/{total_job_skills} skills matched, {missing_count} gaps to address. Semantic similarity: {sim_pct}%."
        elif score_rounded >= 40:
            return f"Weak match ({score_rounded}%). {matched_count}/{total_job_skills} skills matched. {missing_count} gaps. Semantic similarity: {sim_pct}%. Resume needs tailoring."
        else:
            return f"Low match ({score_rounded}%). {matched_count}/{total_job_skills} skills matched. Semantic similarity: {sim_pct}%. Consider upskilling or targeting a different role."
