"""
Routes for job seekers to upload a resume against a specific job.

NOTE: AI scoring (parser -> extractor -> features -> matcher) gets wired in
once Person C's ai/ module is ready. For now this saves the file and a
placeholder candidate record so the frontend can be built against a real
working endpoint immediately.
"""

import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List

from database import get_db
import models
import schemas

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

    # --- TODO (Person C): replace this placeholder block with real AI scoring ---
    # from ai.parser import parse_resume
    # from ai.extractor import extract_fields
    # from ai.matcher import score_candidate
    # resume_text = parse_resume(save_path)
    # fields = extract_fields(resume_text)
    # result = score_candidate(resume_text, job.description)
    resume_text_placeholder = None
    extracted_skills_placeholder = None
    experience_years_placeholder = None
    match_score_placeholder = 0.0
    prediction_placeholder = "Pending"
    explanation_placeholder = "AI scoring not yet connected"
    # --- end placeholder ---

    new_candidate = models.Candidate(
        job_id=job_id,
        filename=file.filename,
        resume_text=resume_text_placeholder,
        extracted_skills=extracted_skills_placeholder,
        experience_years=experience_years_placeholder,
        match_score=match_score_placeholder,
        prediction=prediction_placeholder,
        explanation=explanation_placeholder,
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
