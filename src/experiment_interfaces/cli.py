"""Explicit file-driven commands; outputs never overwrite an existing delivery."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path


def _new_output(path: str | Path, *, directory: bool) -> Path:
    target = Path(path).resolve()
    root = Path(__file__).resolve().parents[2]
    for name in ("experiments", "assets", "policies", "weights", "results", "configs"):
        if target.is_relative_to(root / name):
            raise ValueError("Interface outputs must be outside published experiment directories")
    if directory:
        target.mkdir(parents=True, exist_ok=False)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_json(path: Path, document, *, default=None):
    # Serialize first: a contract failure must not leave a partial JSON document.
    content = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=default)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content + "\n")


def _make_env(config):
    from .envs import RobotEnv, MetaWorldEnv

    factories = {"metaworld": MetaWorldEnv, "robot": RobotEnv}
    return factories[config["backend"]](**config["kwargs"])


def _gpt6(args):
    from .gpt6 import GPT6Client, GenerateRequest

    request = GenerateRequest(**json.loads(Path(args.request).read_text()))
    provider = {} if args.provider is None else json.loads(Path(args.provider).read_text())
    allowed = {"base_url", "api_key_env", "http_headers"}
    if not isinstance(provider, dict) or set(provider) - allowed:
        raise ValueError("Provider JSON accepts only base_url, api_key_env and http_headers; use environment credentials")
    output = _new_output(args.output, directory=True)

    def save_receipt(body, response, http_status):
        _write_json(output / "receipt.json", dict(request=body, response=response, http_status=http_status))

    result = GPT6Client(response_sink=save_receipt, **provider).generate(request)
    _write_json(output / "result.json", dict(
        status=result.status, response_id=result.response_id, model=result.model,
        requested_model=result.request["model"], requested_reasoning_effort=result.request["reasoning"]["effort"],
        returned_reasoning=result.response.get("reasoning"),
        usage=result.usage, output_text=result.output_text,
        provenance="openai_responses_api", prior_pipeline_stage="external_call_only"))


def _describe_env(args):
    config = json.loads(Path(args.config).read_text())
    target = _new_output(args.output, directory=False)
    with closing(_make_env(config)) as env:
        _write_json(target, env.spec.as_dict())
def main(argv=None):
    parser = argparse.ArgumentParser(description="GPT-6 and env interfaces; run through Pixi")
    commands = parser.add_subparsers(dest="command", required=True)
    api = commands.add_parser("gpt6", help="One API request; save raw receipt and validated text")
    api.add_argument("--request", required=True)
    api.add_argument("--provider", help="Explicit provider JSON with URL, environment credential name and headers")
    api.add_argument("--output", required=True)
    api.set_defaults(run=_gpt6)
    describe = commands.add_parser("describe-env", help="Export the exact available simulator contract")
    describe.add_argument("--config", required=True)
    describe.add_argument("--output", required=True)
    describe.set_defaults(run=_describe_env)
    args = parser.parse_args(argv)
    args.run(args)
