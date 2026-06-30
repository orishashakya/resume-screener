"""
SQLAlchemy ORM models.

Job: a job posting created by a recruiter.
Candidate: a resume uploaded by a job seeker against a specific job.
"""

from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(Text)            # comma-separated, parsed from description
    min_experience_years = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    candidates = relationship("Candidate", back_populates="job")


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    filename = Column(String)
    resume_text = Column(Text)
    extracted_skills = Column(Text)
    experience_years = Column(Integer)
    match_score = Column(Float)               # similarity/confidence score from the model
    prediction = Column(String)                # "Yes" or "No"
    explanation = Column(Text)                  # matched/missing skills, human-readable
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("Job", back_populates="candidates")