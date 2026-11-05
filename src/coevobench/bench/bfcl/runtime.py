"""Isolated BFCL generation followed by trusted official per-record grading."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from coevobench.bench.bfcl.provenance import MANIFEST, VENDOR_ROOT, digest, verify_upstream

ENGINE_MODULE = "coevobench.bench.bfcl.grading"
GENERATION_IMAGE = "coevobench-bfcl:latest"


def engine_available() -> bool:
    return (VENDOR_ROOT / "bfcl_eval/data/BFCL_v4_memory.json").is_file()


def settings(
    *,
    model_name="qwen3.5-9b",
    base_url="http://127.0.0.1:8101/v1",
    tokenizer="",
    temperature=0.001,
    max_tokens=4096,
    max_steps=20,
    context_limit=65536,
) -> dict:
    if (
        not math.isfinite(temperature)
        or not 0 <= temperature <= 2
        or not 1 <= max_tokens <= 4096
        or max_steps < 1
        or context_limit < 2
    ):
        raise ValueError("invalid BFCL temperature/output/step configuration")
    if not tokenizer:
        raise ValueError("BFCL needs the served candidate's tokenizer directory")
    return {
        "name": model_name,
        "base_url": base_url,
        "tokenizer": tokenizer,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_steps": max_steps,
        "context_limit": context_limit,
    }


def image_identity(image: str = GENERATION_IMAGE) -> dict:
    info = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    labels = info["Config"].get("Labels") or {}
    if (
        labels.get("coevobench.bfcl.manifest") != digest(MANIFEST)
        or labels.get("coevobench.bfcl.grading") != digest(Path(__file__).with_name("grading.py"))
        or labels.get("coevobench.bfcl.generation")
        != digest(Path(__file__).with_name("generation.py"))
    ):
        raise ValueError("BFCL generation image is stale; run scripts/build-bfcl-image.py")
    return {"image_id": info["Id"], "labels": labels}


def grading_env(project: Path) -> dict:
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(project),
        "BFCL_PROJECT_ROOT": str(project),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]) + os.pathsep + str(VENDOR_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def run_repeat(
    ids: list[str],
    sample: int,
    options: dict,
    harness_dir: Path,
    private_dir: Path,
    workers: int = 8,
    image: str = GENERATION_IMAGE,
    owner_dir: Path | None = None,
) -> Path:
    from coevobench.bench.bfcl.tasks import generation_plan, source_records
    from coevobench.runtime.proxy.completions import completion_bridge
    from coevobench.runtime.run_resources import run_container

    provenance = verify_upstream()
    identity = image_identity(image)
    records = source_records()
    entries = generation_plan(ids, records)
    project = private_dir / f"repeat_{sample}"
    project.mkdir(parents=True, exist_ok=False, mode=0o700)
    (project / "grading-input.json").write_text(
        json.dumps({"records": {key: records[key] for key in ids}, "entries": entries})
    )
    (project / "provenance.json").write_text(
        json.dumps({**provenance, **identity, "options": options}, indent=2)
    )
    with tempfile.TemporaryDirectory(prefix="ceb-bfcl-") as scratch:
        root = Path(scratch)
        inputs, output, policy, tokenizer = [
            root / name for name in ("input", "output", "policy", "tokenizer")
        ]
        for path in (inputs, output, policy, tokenizer):
            path.mkdir(mode=0o755)
        os.chown(output, 1000, 1000)
        for name in ("default_prompts.py", "weak_policy.py"):
            source = harness_dir / name
            if source.is_symlink() or not source.is_file():
                raise ValueError("BFCL policy must contain regular Python files")
            shutil.copyfile(source, policy / name)
        for pattern in ("*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"):
            for path in Path(options["tokenizer"]).glob(pattern):
                if path.is_file():
                    shutil.copyfile(path, tokenizer / path.name)
        (inputs / "tasks.json").write_text(json.dumps(entries))
        public_options = {
            key: value for key, value in options.items() if key not in ("base_url", "tokenizer")
        }
        (inputs / "options.json").write_text(json.dumps(public_options))
        name = "ceb-bfcl-" + uuid.uuid4().hex[:16]
        socket_path = root / "model.sock"
        with completion_bridge(
            socket_path,
            options["base_url"],
            options["name"],
            options["max_tokens"],
            options["temperature"],
        ):
            command = [
                "docker",
                "run",
                "--rm",
                "--init",
                "--name",
                name,
                "--network",
                "none",
                "--user",
                "1000:1000",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--pids-limit",
                "512",
                "--memory",
                "16g",
                "--cpus",
                "8",
                "--tmpfs",
                "/tmp:rw,nosuid,size=2g",
            ]
            for source, target, writable in (
                (inputs, "/input", False),
                (output, "/output", True),
                (policy, "/policy", False),
                (tokenizer, "/tokenizer", False),
                (socket_path, "/model.sock", False),
            ):
                command += ["-v", f"{source}:{target}" + ("" if writable else ":ro")]
            command += [identity["image_id"], str(workers)]
            try:
                with (project / "generate.log").open("w") as log:
                    run_container(command, name, owner_dir, log, timeout=86400)
            finally:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        generated = output / "generation.jsonl"
        if (
            generated.is_symlink()
            or not generated.is_file()
            or generated.stat().st_size > 8 * 1024**3
        ):
            raise ValueError("missing, unsafe, or oversized BFCL generation output")
        shutil.copyfile(generated, project / "generation.jsonl")
    verify_upstream()
    with tempfile.TemporaryDirectory(prefix="ceb-bfcl-grade-") as temporary:
        destination = Path(temporary)
        destination.chmod(0o755)
        os.chown(destination, 1000, 1000)
        project.chmod(0o755)
        for path in project.iterdir():
            if path.is_file():
                path.chmod(0o444)
        try:
            grade_name = "ceb-bfcl-grade-" + uuid.uuid4().hex[:16]
            with (project / "evaluate.log").open("w") as log:
                run_container(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--init",
                        "--name",
                        grade_name,
                        "--network",
                        "none",
                        "--user",
                        "1000:1000",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--read-only",
                        "--tmpfs",
                        "/tmp:rw,nosuid,size=2g",
                        "--memory",
                        "8g",
                        "--cpus",
                        "4",
                        "--pids-limit",
                        "256",
                        "-v",
                        f"{project}:/project:ro",
                        "-v",
                        f"{destination}:/score",
                        "--entrypoint",
                        "python",
                        identity["image_id"],
                        "/opt/grading.py",
                        "/project",
                        "/score/verdicts.jsonl",
                    ],
                    grade_name,
                    owner_dir,
                    log,
                    timeout=7200,
                )
            verdict_file = destination / "verdicts.jsonl"
            if verdict_file.is_symlink() or not verdict_file.is_file():
                raise ValueError("unsafe or missing official verdict file")
            shutil.copyfile(verdict_file, project / "verdicts.jsonl")
        finally:
            project.chmod(0o700)
    return project


def token_usage(project: Path) -> dict:
    """Report scoring and prerequisite token usage without counting prerequisites as trials."""
    requested = set(json.loads((project / "grading-input.json").read_text())["records"])
    totals = {
        name: {"input_tokens": 0, "output_tokens": 0, "records": 0}
        for name in ("scoring", "prerequisites")
    }

    def total(value):
        if isinstance(value, list):
            return sum(total(item) for item in value)
        return int(value) if isinstance(value, (int, float)) and value >= 0 else 0

    for line in (project / "generation.jsonl").open():
        row = json.loads(line)
        target = totals["scoring" if row["id"] in requested else "prerequisites"]
        target["records"] += 1
        target["input_tokens"] += total(row.get("input_token_count", []))
        target["output_tokens"] += total(row.get("output_token_count", []))
    return totals


def read_results(project: Path) -> list[dict]:
    rows = [json.loads(line) for line in (project / "verdicts.jsonl").read_text().splitlines()]
    expected = set(json.loads((project / "grading-input.json").read_text())["records"])
    if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError("incomplete or duplicate official BFCL verdicts")
    if any(type(row.get("correct")) is not bool for row in rows):
        raise ValueError("invalid official BFCL correctness payload")
    return rows
