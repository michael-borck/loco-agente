"""Calibration capture — records user picks/rejects/edits as JSONL.

Every interaction is logged in machine-readable form so downstream studies
(see studies/context/) can analyse model self-confidence vs human-validated
outcomes. The data capture starts on day one; the analysis is downstream
research.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance — small implementation, no dependency."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (ca != cb)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


@dataclass
class CalibrationLog:
    """Append-only JSONL log of user decisions over harness output."""

    path: Path

    def _append(self, entry: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry["timestamp"] = time.time()
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def record_pick(
        self,
        *,
        run_id: str,
        variant_index: int,
        confidence: float,
        frame_name: str,
    ) -> None:
        self._append(
            {
                "event": "pick",
                "run_id": run_id,
                "variant_index": variant_index,
                "confidence": confidence,
                "frame_name": frame_name,
            }
        )

    def record_reject_all(self, *, run_id: str, reason: str = "") -> None:
        self._append(
            {"event": "reject_all", "run_id": run_id, "reason": reason}
        )

    def record_edit(
        self,
        *,
        run_id: str,
        variant_index: int,
        original_text: str,
        edited_text: str,
    ) -> None:
        self._append(
            {
                "event": "edit",
                "run_id": run_id,
                "variant_index": variant_index,
                "original_length": len(original_text),
                "edited_length": len(edited_text),
                "edit_distance": _edit_distance(original_text, edited_text),
            }
        )
