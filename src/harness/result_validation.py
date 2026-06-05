from __future__ import annotations

import math
from typing import Any

from src.utils.metrics import REQUIRED_PAPER_METRICS, REQUIRED_VALIDATION_FIELDS

VALIDATION_TOLERANCE = 1e-6


class ResultValidationError(ValueError):
    """Raised when a formal evaluation/benchmark result is not paper-ready."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_episode_result(result: dict[str, Any], *, allow_debug_truncation: bool = False, tolerance: float = VALIDATION_TOLERANCE) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    required_all = REQUIRED_PAPER_METRICS + REQUIRED_VALIDATION_FIELDS
    missing = [k for k in required_all if k not in result]
    if missing:
        errors.append(f"Missing required metrics/fields: {missing}")

    numeric_metrics = REQUIRED_PAPER_METRICS + ["episode_end_time", "operating_horizon_min"]
    for key in numeric_metrics:
        if key not in result:
            continue
        try:
            value = float(result[key])
        except (TypeError, ValueError):
            errors.append(f"Metric '{key}' must be numeric, got {result[key]!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"Metric '{key}' must be finite, got {value}")

    end_time = float(result.get("episode_end_time", 0.0))
    horizon = float(result.get("operating_horizon_min", 0.0))
    termination_reason = str(result.get("termination_reason", ""))
    truncated_by_max_steps = _as_bool(result.get("truncated_by_max_steps", False))
    full_horizon_completed = _as_bool(result.get("full_horizon_completed", False))

    if end_time > horizon + tolerance:
        errors.append(f"episode_end_time ({end_time}) exceeds operating_horizon_min ({horizon})")
    if termination_reason == "horizon_reached" and abs(end_time - horizon) > tolerance:
        errors.append(f"termination_reason=horizon_reached requires episode_end_time≈operating_horizon_min, got {end_time} vs {horizon}")

    if truncated_by_max_steps:
        msg = "Episode was truncated_by_max_steps=true"
        if allow_debug_truncation:
            warnings.append(msg)
        else:
            errors.append(msg)

    expected_full_horizon = (termination_reason == "horizon_reached") and (abs(end_time - horizon) <= tolerance) and (not truncated_by_max_steps)
    if full_horizon_completed != expected_full_horizon:
        errors.append(
            "full_horizon_completed is inconsistent with termination_reason/episode_end_time/truncated_by_max_steps "
            f"(got {full_horizon_completed}, expected {expected_full_horizon})"
        )

    nonnegative_substrings = ("cost", "delay", "lateness", "duration", "holding_time", "overflow", "stockout")
    for key, raw in result.items():
        if not any(token in key for token in nonnegative_substrings):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric, got {raw!r}")
            continue
        if value < -tolerance:
            errors.append(f"{key} must be nonnegative physical metric, got {raw}")

    for key in ["undelivered_parcel_count", "terminal_undelivered_penalty", "late_delivery_count"]:
        if key in result:
            try:
                if float(result[key]) < 0.0:
                    errors.append(f"{key} must be nonnegative, got {result[key]}")
            except (TypeError, ValueError):
                errors.append(f"{key} must be numeric, got {result[key]!r}")

    status = "success" if not errors else "failed"
    out = dict(result)
    out["paper_ready"] = bool(status == "success" and not warnings)
    out["validation_status"] = status
    out["validation_errors"] = " | ".join(errors) if errors else ""
    out["validation_warnings"] = " | ".join(warnings) if warnings else ""

    if errors and not allow_debug_truncation:
        raise ResultValidationError(out["validation_errors"])

    if errors and allow_debug_truncation:
        out["paper_ready"] = False
    if warnings:
        out["paper_ready"] = False
    return out
