from __future__ import annotations

import argparse
import threading
import traceback
from http.server import HTTPServer
from pathlib import Path

from backend_config import add_backend_args, apply_backend_config, apply_cli_overrides, print_config, resolve_backend_config
import captcha_server as srv

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "captcha_server_headless_internal.log"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the CNCAPTCHA headless backend.")
    add_backend_args(parser)
    args = parser.parse_args(argv)
    apply_cli_overrides(args)
    config = resolve_backend_config(source="headless-cli")
    apply_backend_config(config)
    srv.BACKEND_CONFIG = config
    srv.HOST = config.host
    srv.PORT = config.port
    print_config(config)

    log("[server] headless main entered")
    server = HTTPServer((srv.HOST, srv.PORT), srv.CaptchaHandler)
    log(f"[server] listening on http://127.0.0.1:{srv.PORT}")

    def preload() -> None:
        try:
            log("[server] preloading worker...")
            srv._get_worker()
            log("[server] worker ready")
        except Exception as exc:
            log(f"[server] worker preload failed: {exc}")
            log(traceback.format_exc())

    threading.Thread(target=preload, daemon=True).start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("[server] fatal")
        log(traceback.format_exc())
        raise
