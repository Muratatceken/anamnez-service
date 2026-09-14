#!/usr/bin/env python3
"""Anamnez Anonimizasyon Servisi — Python istemcisi (yalnızca standart kütüphane).

Kütüphane:
    from anamnez_client import AnamnezClient
    c = AnamnezClient("http://10.0.0.5:8080", api_key="...")
    job_id = c.submit("rapor.pdf", ref="HBYS-123")
    result = c.wait(job_id)          # {'status': 'done'|'needs_review'|'failed', 'result': {...}}

CLI:
    python client/anamnez_client.py --url http://10.0.0.5:8080 --key $KEY rapor.pdf [rapor2.pdf ...]
    python client/anamnez_client.py --url ... --key ... --status <job_id>
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


class AnamnezClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _req(self, method: str, path: str, data: Optional[bytes] = None, headers: Optional[dict] = None) -> dict:
        h = {"X-API-Key": self.api_key, **(headers or {})}
        req = urllib.request.Request(self.base_url + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} {path}: {body[:300]}") from None

    def health(self) -> dict:
        return self._req("GET", "/health")

    def submit(self, file_path: str, ref: str = "") -> str:
        p = Path(file_path)
        headers = {"X-Filename": p.name, "Content-Type": "application/octet-stream"}
        if ref:
            headers["X-Ref"] = ref
        return self._req("POST", "/jobs", data=p.read_bytes(), headers=headers)["job_id"]

    def get(self, job_id: str) -> dict:
        return self._req("GET", f"/jobs/{job_id}")

    def wait(self, job_id: str, poll: float = 5.0, timeout: float = 1800.0) -> dict:
        t0 = time.time()
        while True:
            j = self.get(job_id)
            if j["status"] not in ("queued", "processing"):
                return j
            if time.time() - t0 > timeout:
                raise TimeoutError(f"iş {job_id} {timeout}s içinde bitmedi (durum: {j['status']})")
            time.sleep(poll)

    def list(self, status: Optional[str] = None, limit: int = 50) -> list:
        q = f"?limit={limit}" + (f"&status={status}" if status else "")
        return self._req("GET", "/jobs" + q)


def main() -> int:
    ap = argparse.ArgumentParser(description="Anamnez anonimizasyon servisi istemcisi")
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--ref", default="", help="X-Ref: kendi opak referansınız")
    ap.add_argument("--status", metavar="JOB_ID", help="yalnızca durumu sorgula")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--out", help="sonuçları bu dizine JSON olarak yaz")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    c = AnamnezClient(a.url, a.key)

    if a.status:
        print(json.dumps(c.get(a.status), ensure_ascii=False, indent=1))
        return 0
    if not a.files:
        ap.error("dosya verin veya --status kullanın")

    h = c.health()
    if not h.get("ok"):
        print("UYARI: servis sağlıklı değil:", json.dumps(h, ensure_ascii=False)[:300], file=sys.stderr)

    ids = {f: c.submit(f, ref=a.ref or Path(f).stem) for f in a.files}
    for f, jid in ids.items():
        print(f"gönderildi  {jid}  {f}")
    if a.no_wait:
        return 0
    rc = 0
    for f, jid in ids.items():
        j = c.wait(jid)
        r = j.get("result") or {}
        cls = (r.get("classification") or {})
        print(f"{j['status']:13} {jid}  {f}  →  {cls.get('validated_category', '-')} "
              f"({cls.get('adjusted_confidence', '-')})  kapı={'PASS' if (r.get('gate') or {}).get('passed') else 'FAIL'}")
        if j["status"] != "done":
            rc = 2
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            Path(a.out, f"{jid}.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
