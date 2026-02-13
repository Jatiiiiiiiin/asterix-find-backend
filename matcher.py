from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# -----------------------------
# Load model ONCE (CRITICAL)
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Request schema
# -----------------------------
class MatchRequest(BaseModel):
    resume_text: str
    job: Dict  # single job per request (recommended)

class MatchResponse(BaseModel):
    score: float


# -----------------------------
# Helper: clean & structure text
# -----------------------------
def build_job_text(job: Dict) -> str:
    """
    Forces semantic diversity between jobs
    """
    title = job.get("title", "")
    skills = ", ".join(job.get("skills", []))
    experience = job.get("experience", "")
    description = job.get("description", "") or job.get("jd", "")

    job_text = f"""
    Role: {title}
    Required Skills: {skills}
    Experience Level: {experience}
    Job Description: {description}
    """

    return job_text.strip()


def preprocess_resume(text: str) -> str:
    """
    Prevent resume dominance
    """
    if not text:
        return ""

    # HARD LIMIT (you already discovered this 👍)
    return text[:1200]


# -----------------------------
# Core matcher
# -----------------------------
@app.post("/match", response_model=MatchResponse)
def match_resume_to_job(payload: MatchRequest):

    resume_text = preprocess_resume(payload.resume_text)
    job_text = build_job_text(payload.job)

    if not resume_text or not job_text:
        raise HTTPException(status_code=400, detail="Resume or Job text missing")

    try:
        # Embed resume ONCE
        resume_vec = model.encode(
            resume_text,
            normalize_embeddings=True
        )

        # Embed job
        job_vec = model.encode(
            job_text,
            normalize_embeddings=True
        )

        # Cosine similarity
        similarity = cosine_similarity(
            [resume_vec],
            [job_vec]
        )[0][0]

        # ---- UX score shaping (optional but recommended) ----
        # Spread scores visually so UI feels responsive
        similarity = float(similarity ** 1.15)

        score = round(similarity * 100, 2)

        return MatchResponse(score=score)

    except Exception as e:
        print("MATCH ERROR:", str(e))
        raise HTTPException(status_code=500, detail="Matching failed")