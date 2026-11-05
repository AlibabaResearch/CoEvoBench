"""Bench adapter SPI and shared rollout/judge plumbing (Section 4.1).

A bench adapter owns three things: the public view of a task (gold
stripped), spawning the rollout sandbox, and judging a submission in a
fresh sandbox (grading separated from rollout, so a dirtied workspace
cannot influence the verdict).

Concrete benches subclass this skeleton; the toy bench is the reference
implementation of the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from coevobench.core.protocol.spec import TaskView, Verdict
from coevobench.core.protocol.splits import Splits


class BenchAdapter(Protocol):
    name: str

    def task_ids(self, split: str) -> list[str]: ...

    def public_view(self, task_id: str) -> TaskView: ...

    def judge(self, task_id: str, submission: str, sandbox: object) -> Verdict: ...


class FileBench:
    """Base for benches whose splits live as a JSON file on disk.

    Layout: <data_root>/<bench>_splits.json with {"visible": [...],
    "selection": [...], "test": [...]}.
    """

    name = ""

    def __init__(self, data_root: Path) -> None:
        self._root = data_root
        self._splits: Splits | None = None

    @property
    def splits(self) -> Splits:
        if self._splits is None:
            import json

            path = self._root / f"{self.name}_splits.json"
            if not path.is_file():
                # the splits' source of truth is the repo's frozen
                # definition (bench/<name>/splits.json); data-root copies
                # are acceleration, never a hard dependency
                from coevobench.core.protocol.protocol_assets import frozen_splits

                self._splits = frozen_splits(self.name)
                if self._splits is None:
                    raise FileNotFoundError(
                        f"no {self.name}_splits.json under {self._root} and no "
                        "frozen split definition in the package"
                    )
                return self._splits
            payload = json.loads(path.read_text())
            self._splits = Splits(
                visible=tuple(payload.get("visible", [])),
                selection=tuple(payload.get("selection", [])),
                test=tuple(payload.get("test", [])),
            )
        return self._splits

    def task_ids(self, split: str) -> list[str]:
        return list(self.splits.ids(split))



__all__ = ["BenchAdapter", "FileBench"]
