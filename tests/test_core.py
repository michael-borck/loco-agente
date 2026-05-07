"""Tests for harness.core — Variant, Uncertainty, parser, generate_variants."""
from __future__ import annotations

import pytest

from harness.core import Uncertainty, Variant


def test_variant_dataclass_minimal_fields() -> None:
    v = Variant(
        text="hello",
        rationale="testing",
        uncertainty=Uncertainty(confidence=0.5, flags=[], verification_hooks=[]),
        metadata={"frame_name": "test"},
    )
    assert v.text == "hello"
    assert v.uncertainty.confidence == 0.5
    assert v.metadata["frame_name"] == "test"


def test_uncertainty_default_relative_rank_is_none() -> None:
    u = Uncertainty(confidence=0.7, flags=["a"], verification_hooks=["b"])
    assert u.relative_rank is None


def test_uncertainty_accepts_relative_rank() -> None:
    u = Uncertainty(
        confidence=0.7, relative_rank=2, flags=[], verification_hooks=[]
    )
    assert u.relative_rank == 2
