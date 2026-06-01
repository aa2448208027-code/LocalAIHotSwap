from __future__ import annotations

import argparse
import json
import urllib.request

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hotmodel")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="run the HotModelReplacement HTTP proxy")
    serve.add_argument("--config", required=True)

    switch = subcommands.add_parser("switch", help="switch the active model through a running proxy")
    switch.add_argument("model")
    switch.add_argument("--config", required=True)

    models = subcommands.add_parser("models", help="list configured models")
    models.add_argument("--config", required=True)

    args = parser.parse_args(argv)

    if args.command == "serve":
        return _serve(args.config)
    if args.command == "switch":
        return _switch(args.config, args.model)
    if args.command == "models":
        return _models(args.config)
    raise AssertionError(args.command)


def _serve(config_path: str) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the server") from exc
    config = load_config(config_path)
    from .api import create_app

    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
    return 0


def _switch(config_path: str, model: str) -> int:
    config = load_config(config_path)
    body = json.dumps({"model": model}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{config.host}:{config.port}/admin/switch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        print(response.read().decode("utf-8"))
    return 0


def _models(config_path: str) -> int:
    config = load_config(config_path)
    for name in sorted(config.models):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
