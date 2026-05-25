from __future__ import annotations

import argparse
import sys

from backend_config import add_backend_args, apply_backend_config, apply_cli_overrides, print_config, resolve_backend_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the CNCAPTCHA backend with automatic device selection.")
    parser.add_argument("--headless", action="store_true", help="Run without the Tk GUI window")
    add_backend_args(parser)
    args = parser.parse_args(argv)

    apply_cli_overrides(args)
    config = resolve_backend_config(source="start-backend-cli")
    apply_backend_config(config)
    print_config(config)

    if args.headless:
        import captcha_server_headless

        return captcha_server_headless.main([])

    import captcha_server

    return captcha_server.main([])


if __name__ == "__main__":
    raise SystemExit(main())
