"""Run registry: ~/.ceb/runs/<id>/ is the single source of truth (Section 8.5).

Layout per run: manifest.json, lifecycle.json, workspace/ (bind-mounted
into the run envelope, paper Appendix A contract), audit/, select_runs/,
final/. run ids align with the paper's Table 2 numbering: <researcher>_<bench>_<seq>.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunManifest:
    run_id: str
    bench: str
    researcher_model: str
    researcher_cli: str
    replicate: int
    budget_hours: float
    gpus: tuple[int, ...]
    image: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    phase: str = "registered"
    container: str | None = None
    # the host-side truth of the envelope's read-only /base mount
    base_dir: str | None = None
    runhost: str = "docker"
    process_start: str | None = None
    submission: dict | None = None


class RunRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or (Path.home() / ".ceb")) / "runs"

    def dir_for(self, run_id: str) -> Path:
        return self.root / run_id

    def workspace_dir(self, run_id: str) -> Path:
        return self.dir_for(run_id) / "workspace"

    def audit_dir(self, run_id: str) -> Path:
        return self.dir_for(run_id) / "audit"

    def select_dir(self, run_id: str) -> Path:
        return self.dir_for(run_id) / "select_runs"

    def register(self, manifest: RunManifest) -> Path:
        target = self.dir_for(manifest.run_id)
        target.mkdir(parents=True, exist_ok=True)
        for sub in ("workspace", "audit", "select_runs"):
            (target / sub).mkdir(exist_ok=True)
        self.save_manifest(manifest)
        return target

    def save_manifest(self, manifest: RunManifest) -> None:
        payload = asdict(manifest)
        payload["gpus"] = list(manifest.gpus)
        (self.dir_for(manifest.run_id) / "manifest.json").write_text(
            json.dumps(payload, indent=2)
        )

    def load_manifest(self, run_id: str) -> RunManifest | None:
        path = self.dir_for(run_id) / "manifest.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        data["gpus"] = tuple(data.get("gpus", ()))
        return RunManifest(**data)

    def list_runs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            d.name for d in self.root.iterdir() if (d / "manifest.json").is_file()
        )

    def next_replicate(self, researcher: str, bench: str) -> int:
        prefix = f"{researcher}_{bench}_"
        used = [int(r.removeprefix(prefix)) for r in self.list_runs() if r.startswith(prefix)]
        return max(used, default=0) + 1


__all__ = ["RunManifest", "RunRegistry"]
