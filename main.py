from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from ml_engine import ResumeAnalyzer

app = FastAPI(title="AI Resume Screener", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = ResumeAnalyzer()


class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str


class AnalysisResponse(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    key_resume_features: List[str]
    job_requirement_mapping: List[Dict]
    resume_analysis_summary: str
    improvement_suggestions: List[str]


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_resume(request: AnalysisRequest):
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text is required")
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required")

    result = analyzer.analyze(request.resume_text, request.job_description)

    return AnalysisResponse(
        match_score=result.match_score,
        matched_skills=result.matched_skills,
        missing_skills=result.missing_skills,
        key_resume_features=result.key_resume_features,
        job_requirement_mapping=result.job_requirement_mapping,
        resume_analysis_summary=result.resume_analysis_summary,
        improvement_suggestions=result.improvement_suggestions
    )


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Resume Screener API is running. Use POST /api/analyze"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
