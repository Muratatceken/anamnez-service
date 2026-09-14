"""API şemaları."""

from typing import Any, Optional

from pydantic import BaseModel


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobListItem(BaseModel):
    id: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: str
    filename: str            # "<job_id_ilk8>.<uzantı>" — gerçek dosya adı saklanmaz
    ref: Optional[str] = ""  # X-Ref başlığıyla verilen opak müşteri referansı
    size_bytes: Optional[int] = None
    error: Optional[str] = None


class JobResponse(JobListItem):
    file_sha256: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    ok: bool
    components: dict[str, Any]
