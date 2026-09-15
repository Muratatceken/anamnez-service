"""Egress gateway — sunucudan dışarı çıkan TEK yol.

Kural: Buluta yalnızca (1) kapıdan geçmiş, (2) bilinen hiçbir PII adayını içermeyen, (3) maske etiketi
dışında 11 haneli sayı / tam tarih / e-posta / telefon içermeyen anonim metin gider. Her çıkış
denetim kaydına (hash, boyut, hedef, model) yazılır — metin asla yazılmaz.

Ağ seviyesinde de aynı kural: nftables + squid allowlist yalnızca api.anthropic.com:443 (bkz. deploy/).
"""

import hashlib
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .anonymization.gate import GateResult, tr_fold

logger = logging.getLogger(__name__)

ALLOWED_PROVIDERS = {"anthropic": "api.anthropic.com"}

# Son savunma: maske etiketleri dışında bariz PII kalıpları
_FINAL_RULES = [
    ("tc", re.compile(r"\b[1-9]\d{10}\b")),
    ("date", re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](?:19|20)\d{2}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("phone", re.compile(r"(?<!\d)0?\s?\(?5\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)")),
    ("url", re.compile(r"https?://\S+|www\.\S+")),
]


class EgressBlocked(Exception):
    pass


@dataclass
class EgressDecision:
    allowed: bool
    reasons: list[str]
    text_sha256: str
    chars: int


class EgressGateway:
    def __init__(self, cfg: dict, audit_db_path: Optional[str] = None):
        self.enabled = bool(cfg.get("enabled", False))
        self.provider = cfg.get("provider", "anthropic")
        if self.provider not in ALLOWED_PROVIDERS:
            raise ValueError(f"izin verilmeyen sağlayıcı: {self.provider}")
        self.host = ALLOWED_PROVIDERS[self.provider]
        self.model = cfg.get("model", "claude-opus-5")
        self.audit_db_path = audit_db_path
        self._lock = threading.Lock()
        if audit_db_path:
            Path(audit_db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(audit_db_path) as c:
                c.execute("""CREATE TABLE IF NOT EXISTS egress_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, job_id TEXT, allowed INTEGER NOT NULL,
                    reasons TEXT, text_sha256 TEXT, chars INTEGER, host TEXT, model TEXT)""")

    def decide(self, text: str, gate: GateResult, candidates: set[str]) -> EgressDecision:
        reasons: list[str] = []
        if not self.enabled:
            reasons.append("egress devre dışı")
        if not gate.passed:
            reasons.append("kapı geçilmedi")
        folded = tr_fold(text)
        for cand in candidates:
            c = tr_fold(cand.strip())
            if len(c) >= 3 and re.search(r"(?<![\wçğıöşü])" + re.escape(c) + r"(?![\wçğıöşü])", folded):
                reasons.append("PII adayı çıktıda mevcut")
                break
        for name, rx in _FINAL_RULES:
            if rx.search(text):
                reasons.append(f"son kontrol: {name}")
                break
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return EgressDecision(allowed=not reasons, reasons=reasons, text_sha256=sha, chars=len(text))

    def authorize(self, text: str, gate: GateResult, candidates: set[str], job_id: str = "") -> EgressDecision:
        """İzin ver ya da EgressBlocked fırlat; her iki durumda denetim kaydı yaz."""
        d = self.decide(text, gate, candidates)
        self._audit(job_id, d)
        if not d.allowed:
            logger.warning("Egress ENGELLENDİ (%s): %s", job_id[:8], "; ".join(d.reasons))
            raise EgressBlocked("; ".join(d.reasons))
        return d

    def _audit(self, job_id: str, d: EgressDecision) -> None:
        if not self.audit_db_path:
            return
        with self._lock, sqlite3.connect(self.audit_db_path) as c:
            c.execute(
                "INSERT INTO egress_audit (ts, job_id, allowed, reasons, text_sha256, chars, host, model) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), job_id, int(d.allowed),
                 "; ".join(d.reasons), d.text_sha256, d.chars, self.host, self.model),
            )
