"""Veri modelleri."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class CancerCategory(str, Enum):
    ADRENAL_GLAND = "Adrenal_Gland"
    BILE_DUCT = "Bile_Duct"
    BLADDER = "Bladder"
    BLOOD = "Blood"
    BONE = "Bone"
    BONE_MARROW = "Bone_Marrow"
    BRAIN = "Brain"
    BREAST = "Breast"
    CANCER_ALL = "Cancer_all"
    CERVIX = "Cervix"
    COLORECTAL = "Colorectal"
    ESOPHAGUS = "Esophagus"
    EYE = "Eye"
    HEAD_AND_NECK = "Head_and_Neck"
    INFLAMMATORY = "Inflammatory"
    INTRAHEPATIC = "Intrahepatic"
    KIDNEY = "Kidney"
    LIVER = "Liver"
    LUNG = "Lung"
    LYMPH_NODES = "Lymph_Nodes"
    NERVOUS_SYSTEM = "Nervous_System"
    OTHER = "Other"
    OVARY = "Ovary"
    PANCREAS = "Pancreas"
    PLEURA = "Pleura"
    PROSTATE = "Prostate"
    SKIN = "Skin"
    SOFT_TISSUE = "Soft_Tissue"
    STOMACH = "Stomach"
    TESTIS = "Testis"
    THYMUS = "Thymus"
    THYROID = "Thyroid"
    UTERUS = "Uterus"


class ClassificationResult(BaseModel):
    """LLM sınıflandırma sonucu."""
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    histological_type: Optional[str] = None
    primary_site: Optional[str] = None
    extracted_text_summary: Optional[str] = None


class ValidationResult(BaseModel):
    """Doğrulama sonucu."""
    is_valid: bool
    original_category: str
    validated_category: str
    keyword_matches: list[str] = []
    warnings: list[str] = []
    confidence_adjusted: float


class AnonymizationRecord(BaseModel):
    """Anonimizasyon işlem kaydı (loglama için)."""
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    filename: str
    file_type: str
    original_length: int = 0
    anonymized_length: int = 0
    fields_removed_count: int = 0
    fields_removed: str = ""
    fields_generalized: str = ""
    warnings: str = ""
    processing_time_seconds: float = 0.0
    method: str = "regex"  # "regex" veya "ner"


class ProcessingRecord(BaseModel):
    """İşleme kaydı (loglama için)."""
    id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    filename: str
    file_type: str
    input_mode: str = "vision"  # "vision" veya "text"
    classification_category: str
    classification_confidence: float
    validation_passed: bool
    final_category: str
    processing_time_seconds: float
    warnings: str = ""
    llm_model: str = ""
