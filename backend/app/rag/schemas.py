from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, allow_inf_nan=False)


class PolicyDocument(Model):
    doc_id: str = Field(pattern=r"^POL-[A-Z]+-\d{3}$")
    title: str = Field(min_length=1, max_length=160)
    version: str = Field(pattern=r"^\d+\.\d+$")
    effective_date: date
    department: str = Field(min_length=1, max_length=80)
    policy_type: str = Field(min_length=1, max_length=80)
    source_path: str
    content_hash: str
    body: str = Field(min_length=1)


class PolicyChunk(Model):
    chunk_id: str
    doc_id: str
    title: str
    section_id: str
    section_title: str
    chunk_index: int
    text: str
    source_path: str
    version: str
    effective_date: date
    department: str
    policy_type: str
    content_hash: str


class PolicyCitation(Model):
    policy_evidence_id: str
    chunk_id: str
    doc_id: str
    title: str
    section_title: str
    source_path: str
    version: str
    effective_date: date


class PolicyHit(Model):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    section_title: str
    excerpt: str = Field(max_length=800)
    vector_rank: int | None
    bm25_rank: int | None
    vector_distance: float | None
    bm25_score: float | None
    rrf_score: float
    metadata: dict
    citation: PolicyCitation


class PolicySearchResult(Model):
    query: str
    retrieval_mode: Literal["HYBRID"] = "HYBRID"
    hits: list[PolicyHit] = Field(min_length=1, max_length=8)


class PolicySearchInput(Model):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=8, strict=True)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("Query must contain at least two nonblank characters")
        return value.strip()


class PolicyEvidence(Model):
    evidence_id: str
    evidence_kind: Literal["POLICY"] = "POLICY"
    excerpt: str = Field(max_length=800)
    citation: PolicyCitation


class RetrievalError(Exception):
    def __init__(self, code, *, retryable=False):
        self.code, self.retryable = code, retryable
        super().__init__(code)
