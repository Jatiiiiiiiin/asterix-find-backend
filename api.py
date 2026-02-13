from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pdfplumber
import numpy as np
import json
import math

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ================= APP =================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://asterix-find.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= MODELS =================

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)


# ================= HELPERS =================

def extract_pdf_text(file: UploadFile) -> str:
    file.file.seek(0)
    text = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_job_text(title: str, description: str) -> str:
    return f"""
Role: {title}

Job Description:
{description}
""".strip()


def compute_skill_mastery(
    skill: str,
    resume_vec: np.ndarray,
    job_vec: np.ndarray,
    profile_vec: np.ndarray
) -> int:
    skill_vec = embedder.encode(
        skill.lower(),
        normalize_embeddings=True
    )

    resume_sim = float(np.dot(skill_vec, resume_vec))
    job_sim = float(np.dot(skill_vec, job_vec))
    profile_sim = float(np.dot(skill_vec, profile_vec))

    raw = (
        0.50 * resume_sim +
        0.30 * job_sim +
        0.20 * profile_sim
    )

    mastery = 1 - math.exp(-3 * raw)
    mastery = max(0.0, min(1.0, mastery))

    return round(mastery * 100)


# ================= MATCH =================

@app.post("/match")
async def match_resume(
    resume: UploadFile = File(...),
    jobTitle: str = Form(...),
    jobDescription: str = Form(...),
    candidateSkills: str = Form(...),
    profileText: str = Form(...),
    auditSkills: str = Form(None)
):
    # -------- Text preparation --------
    resume_text = extract_pdf_text(resume)[:1200]
    job_text = build_job_text(
        jobTitle,
        jobDescription[:1000]
    )
    profile_text = profileText[:800]

    if len(resume_text) < 50:
        return {
            "fidelityScore": 0,
            "skillAudit": [],
            "breakdown": {}
        }

    # -------- Embeddings (normalized) --------
    resume_vec = embedder.encode(
        resume_text,
        normalize_embeddings=True
    )
    job_vec = embedder.encode(
        job_text,
        normalize_embeddings=True
    )
    profile_vec = embedder.encode(
        profile_text,
        normalize_embeddings=True
    )

    # -------- Semantic scores --------
    resume_score = cosine_sim(resume_vec, job_vec)
    profile_score = cosine_sim(profile_vec, job_vec)

    # -------- Skill bonus --------
    skills = json.loads(candidateSkills) if candidateSkills else []
    skill_hits = 0

    for s in skills:
        name = (s.get("skill") or "").lower()
        weight = int(s.get("weight") or 0)
        if name and name in job_text.lower():
            skill_hits += weight

    skill_bonus = min(skill_hits / 500, 0.15)

    # ================= SCORE LOGIC (UNCHANGED) =================

    raw_score = (
        1.00 * resume_score +
        0.20 * profile_score +
        0.20 * skill_bonus
    )

    MAX_SCORE = 1.60
    normalized = raw_score / MAX_SCORE

    final_score = 1 - math.exp(-3 * normalized)
    final_score = max(0.0, min(1.0, final_score))

    # ==========================================================

    # -------- Skill Audit --------
    skill_audit = []
    audit_list = json.loads(auditSkills) if auditSkills else []

    for skill in audit_list:
        mastery = compute_skill_mastery(
            skill,
            resume_vec,
            job_vec,
            profile_vec
        )

        skill_audit.append({
            "skill": skill.upper(),
            "score": mastery
        })

    return {
        "fidelityScore": round(final_score * 100),
        "skillAudit": skill_audit,
        "matchHighlights": [
            "Resume relevance",
            "Profile alignment",
            "Skill overlap"
        ],
        "breakdown": {
            "resume_semantic": round(resume_score * 100),
            "profile_semantic": round(profile_score * 100),
            "skills_bonus": round(skill_bonus * 100)
        }
    }


# ================= INSIGHTS =================

@app.post("/insights")
async def generate_insights(
    candidateName: str = Form(...),
    jobTitle: str = Form(...)
):
    prompt = f"""
Give 3 bullet points explaining why {candidateName}
is a good fit for {jobTitle}.
Each bullet under 12 words.
"""

    result = generator(
        prompt,
        max_length=128
    )[0]["generated_text"]

    bullets = [
        l.strip("-• ").strip()
        for l in result.split("\n")
        if len(l.strip()) > 8
    ]

    return {"points": bullets[:3]}


# ================= SUMMARY =================

@app.post("/summary")
async def generate_summary(
    jobDescription: str = Form(...)
):
    prompt = f"""
Extract top 3 technical skills.
Return comma separated list only.

{jobDescription[:800]}
"""

    result = generator(
        prompt,
        max_length=64
    )[0]["generated_text"]

    skills = [
        s.strip()
        for s in result.split(",")
        if len(s.strip()) > 2
    ]

    return {
        "requirements": skills[:3],
        "estimatedMatchPool": 15
    }


# ================= CHAT =================

class ChatRequest(BaseModel):
    jobTitle: str
    jobDescription: str
    question: str
    history: List[dict]


@app.post("/chat")
async def job_chat(request: ChatRequest):
    prompt = f"""
You are a professional job assistant.

Use ONLY the information below.
Be concise.
Use bullet points.

Job Title: {request.jobTitle}

Job Description:
{request.jobDescription[:1000]}

Question:
{request.question}

Answer:
"""

    result = generator(
        prompt,
        max_length=256
    )[0]["generated_text"]

    return {
        "answer": result.replace(prompt, "").strip()
    }
