"""
Pydantic models for data validation (Agent 2 output).
"""
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class QualityFlag(BaseModel):
    field: str
    issue: str
    severity: str  # "warning" | "error"


class DataQualityReport(BaseModel):
    ticker: str
    quality_score: float = Field(ge=0.0, le=1.0, description="0=unusable, 1=perfect")
    flags: list[QualityFlag] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    stale_fields: list[str] = Field(default_factory=list)
    outlier_fields: list[str] = Field(default_factory=list)
    currency: str = "USD"
    passed: bool = True
    message: str = ""


class ValidationResult(BaseModel):
    primary: DataQualityReport
    competitors: list[DataQualityReport] = Field(default_factory=list)
    overall_passed: bool = True
    overall_quality_score: float = 1.0
