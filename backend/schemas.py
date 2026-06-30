"""
Pydantic schemas - define what data the API accepts and returns.
Keeps validation separate from the database models (good FastAPI practice).
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# ---------- Job schemas ----------

class JobCreate(BaseModel):
    title: str
    description: str
    required_skills: Optional[str] = None      # recruiter can leave blank; AI module can extract it
    min_experience_years: Optional[int] = 0


class JobResponse(BaseModel):
    id: int
    title: str
    description: str
    required_skills: Optional[str]
    min_experience_years: int
    created_at: datetime

    class Config:
        from_attributes = True  # allows creating this from an ORM object directly


# ---------- Candidate schemas ----------

class CandidateResponse(BaseModel):
    id: int
    job_id: int
    filename: Optional[str]
    extracted_skills: Optional[str]
    experience_years: Optional[int]
    match_score: Optional[float]
    prediction: Optional[str]
    explanation: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True