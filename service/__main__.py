"""python -m service  → uvicorn ile servisi başlat."""

import uvicorn

from .config import load_config


def main() -> None:
    cfg = load_config()
    srv = cfg.get("server", {})
    uvicorn.run(
        "service.api:app",
        host=srv.get("host", "127.0.0.1"),
        port=int(srv.get("port", 8080)),
        workers=1,          # iş kuyruğu süreç içi; çoklu uvicorn worker KULLANMA
        log_level="info",
        access_log=False,   # erişim logunda dosya adı vb. görünmesin
    )


if __name__ == "__main__":
    main()
