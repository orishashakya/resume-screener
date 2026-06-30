"""
Routes for job seekers to upload a resume against a specific job.
AI scoring (parser -> features -> matcher) is wired in here.
"""

import os
import sys
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

# the ai/ folder is a sibling of backend/, so add it to the path to import from it
AI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ai")
sys.path.insert(0, AI_DIR)
from parser import parse_resume          # noqa: E402
from matcher import score_candidate      # noqa: E402

router = APIRouter(prefix="/candidates", tags=["candidates"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@router.post("/upload", response_model=schemas.CandidateResponse)
def upload_candidate(
    job_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job = db.query(models.Job).filter(models.Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf and .docx files are supported")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        resume_text = parse_resume(save_path)
        result = score_candidate(resume_text, job.description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI scoring failed: {str(e)}")

    new_candidate = models.Candidate(
        job_id=job_id,
        filename=file.filename,
        resume_text=resume_text,
        extracted_skills=", ".join(result["explanation"]["matched_skills"]),
        experience_years=result["explanation"]["resume_experience_years"],
        match_score=result["match_score"],
        prediction=result["prediction"],
        explanation=str(result["explanation"]),
    )
    db.add(new_candidate)
    db.commit()
    db.refresh(new_candidate)
    return new_candidate


@router.get("/job/{job_id}", response_model=List[schemas.CandidateResponse])
def list_candidates_for_job(job_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Candidate)
        .filter(models.Candidate.job_id == job_id)
        .order_by(models.Candidate.match_score.desc())
        .all()
    )