from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.harness.benchmark_runner import build_policy
from src.harness.evaluator import evaluate_policy
from src.harness.methods import normalize_method_name
from src.harness.result_aggregator import aggregate
from src.utils.metrics import REQUIRED_PAPER_METRICS

FACTOR_PATHS = {
    "passenger_intensity": ("passenger", "demand_intensity_factor"),
    "base_load_perturbation_intensity": ("power", "disturbance_std_kw"),
    "num_customers": ("generation", "num_customers"),
    "customer_number": ("generation", "num_customers"),
    "parcel_intensity": ("parcel", "demand_intensity_factor"),
    "chargers_per_station": ("charging", "chargers_per_station"),
    "drones_per_station": ("drone", "drones_per_station"),
    "locker_capacity": ("parcel", "locker_capacity_kg"),
    "bus_freight_capacity": ("bus", "freight_capacity_kg"),
    "station_unloading_capacity": ("parcel", "unloading_capacity_kg_per_stop"),
    "station_power_capacity": ("power", "station_capacity_kw"),
    "charging_power": ("charging", "pantograph_power_kw"),
    "initial_full_batteries": ("battery", "initial_fully_charged_per_station"),
    "max_charging_duration": ("charging", "max_single_stop_seconds"),
    "freight_trip_availability": ("generation", "freight_trip_availability"),
}

# Generated instances copy the bus/charging/drone/parcel/power sections and bake
# station resources into each station. Passenger and power scenarios are generated
# files too. Rebuilding all supported factors is therefore the safe, explicit rule.
REQUIRES_INSTANCE_REGEN = set(FACTOR_PATHS)
REQUIRES_OFFLINE_RESOLVE = {
    "num_customers",
    "customer_number",
    "parcel_intensity",
    "locker_capacity",
    "drones_per_station",
    "bus_freight_capacity",
    "station_unloading_capacity",
    "integrated_station_set",
    "freight_trip_availability",
}


def _factor_flags(factor: str) -> dict[str, bool]:
    regenerate = factor in REQUIRES_INSTANCE_REGEN
    return {
        "regenerate_instance": regenerate,
        "resolve_offline": factor in REQUIRES_OFFLINE_RESOLVE,
        "rebuild_env": regenerate,
    }


def _config_value(cfg: dict, factor: str) -> Any:
    section, key = FACTOR_PATHS[factor]
    return cfg.get(section, {}).get(key)


def _log_application(
    *,
    cli_parameter: str,
    factor: str,
    value: Any,
    seed: int,
    actual: Any,
    stage: str,
) -> None:
    print(f"[SENSITIVITY] parameter={cli_parameter} stage={stage}", flush=True)
    print(f"[SENSITIVITY] internal_key={factor} stage={stage}", flush=True)
    print(f"[SENSITIVITY] value={value} stage={stage}", flush=True)
    print(f"[SENSITIVITY] seed={seed} stage={stage}", flush=True)
    print(
        f"[SENSITIVITY] applied_config_or_instance_value={actual} stage={stage}",
        flush=True,
    )


def _failure_row(
    *,
    cli_parameter: str,
    factor: str,
    value: Any,
    instance_name: str,
    method: str,
    seed: int,
    offline_status: str,
    regenerated: bool,
    offline_resolved: bool,
    scenario_token: dict,
    smoke_test: bool,
) -> dict:
    row = {metric: None for metric in REQUIRED_PAPER_METRICS}
    row.update(
        {
            "sensitivity_parameter": cli_parameter,
            "sensitivity_internal_key": factor,
            "sensitivity_name": factor,  # Backward-compatible export name.
            "sensitivity_value": value,
            "instance": instance_name,
            "instance_size": instance_name,
            "method": method,
            "seed": seed,
            "offline_status": offline_status,
            "whether_instance_regenerated": regenerated,
            "whether_offline_resolved": offline_resolved,
            "whether_policy_retrained": False,
            "policy_source": "not_run",
            "full_horizon_completed": False,
            "termination_reason": "offline_infeasible",
            "runtime_sec": 0.0,
            "scenario_token": json.dumps(scenario_token, sort_keys=True),
            "smoke_mode": bool(smoke_test),
        }
    )
    return row


def _write_rows(rows: list[dict], out_csv: str) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    for required in (
        "sensitivity_parameter",
        "sensitivity_internal_key",
        "sensitivity_value",
        "method",
        "instance",
        "seed",
        "total_cost",
        "total_reward",
        *REQUIRED_PAPER_METRICS,
    ):
        if required not in fieldnames:
            fieldnames.append(required)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_combinations(
    rows: list[dict], values: list[Any], seeds: list[int], methods: list[str]
) -> None:
    expected = len(values) * len(seeds) * len(methods)
    if len(rows) < expected:
        raise AssertionError(
            "Sensitivity summary is incomplete: "
            f"expected at least {expected} rows "
            f"({len(values)} values x {len(seeds)} seeds x {len(methods)} methods), "
            f"got {len(rows)}."
        )
    observed = {
        (str(row.get("sensitivity_value")), int(row["seed"]), row["method"])
        for row in rows
    }
    missing = [
        (value, seed, method)
        for value in values
        for seed in seeds
        for method in methods
        if (str(value), int(seed), method) not in observed
    ]
    if missing:
        raise AssertionError(f"Sensitivity summary is missing combinations: {missing}")


def run_sensitivity(
    methods,
    out_csv: str,
    env_builder,
    instance_name: str,
    test_seeds: list[int],
    cfg: dict,
    factor: str,
    values: list[float],
    smoke_test: bool = False,
    train_if_missing: bool = False,
    sensitivity_parameter: str | None = None,
):
    if factor not in FACTOR_PATHS:
        raise ValueError(f"Unsupported sensitivity factor: {factor}")
    methods = [normalize_method_name(method) for method in methods]
    cli_parameter = sensitivity_parameter or factor
    rows: list[dict] = []
    eval_episodes = int(
        cfg.get("rl", {}).get(
            "benchmark_eval_episodes", cfg.get("rl", {}).get("evaluation_episodes", 1)
        )
    )
    section, key = FACTOR_PATHS[factor]
    flags = _factor_flags(factor)
    hooks = cfg.get("_sensitivity_hooks", {})
    regen_cb = hooks.get("regenerate_instance")
    resolve_cb = hooks.get("resolve_offline")
    retrain_cb = hooks.get("retrain_policy")

    for value in values:
        cfg_mod = deepcopy(cfg)
        cfg_mod.pop("_sensitivity_hooks", None)
        cfg_mod.setdefault(section, {})[key] = value
        for seed in test_seeds:
            scenario_token = {
                "seed": seed,
                "factor": factor,
                "value": value,
                "instance": instance_name,
            }
            offline_status = "not_required"
            regenerated = False
            offline_resolved = False
            actual = _config_value(cfg_mod, factor)
            _log_application(
                cli_parameter=cli_parameter,
                factor=factor,
                value=value,
                seed=seed,
                actual=actual,
                stage="before_instance_generation",
            )

            if flags["regenerate_instance"]:
                if regen_cb is None:
                    raise RuntimeError(
                        f"Sensitivity factor '{factor}' requires instance regeneration, "
                        "but no regenerate_instance hook was configured."
                    )
                callback_actual = regen_cb(cfg_mod, instance_name, seed, factor, value)
                regenerated = True
                if callback_actual is not None:
                    actual = callback_actual

            _log_application(
                cli_parameter=cli_parameter,
                factor=factor,
                value=value,
                seed=seed,
                actual=actual,
                stage="before_offline_assignment",
            )
            if flags["resolve_offline"]:
                offline_resolved = True
                offline_status = (
                    resolve_cb(cfg_mod, instance_name, seed, factor, value)
                    if resolve_cb is not None
                    else "failed"
                )
                if offline_status in {"infeasible", "failed"}:
                    for method in methods:
                        rows.append(
                            _failure_row(
                                cli_parameter=cli_parameter,
                                factor=factor,
                                value=value,
                                instance_name=instance_name,
                                method=method,
                                seed=seed,
                                offline_status=offline_status,
                                regenerated=regenerated,
                                offline_resolved=offline_resolved,
                                scenario_token=scenario_token,
                                smoke_test=smoke_test,
                            )
                        )
                    continue

            for method in methods:
                _log_application(
                    cli_parameter=cli_parameter,
                    factor=factor,
                    value=value,
                    seed=seed,
                    actual=actual,
                    stage="before_evaluation",
                )
                started = time.time()
                env = env_builder(seed, cfg_mod)
                retrained = (
                    bool(retrain_cb(cfg_mod, instance_name, seed, factor, value, method))
                    if retrain_cb is not None
                    else False
                )
                policy = build_policy(
                    method,
                    env,
                    out_root=cfg["paths"]["outputs"],
                    train_if_missing=train_if_missing,
                    smoke_test=smoke_test,
                    cfg=cfg_mod,
                    seed=seed,
                    instance_name=instance_name,
                )
                metrics = evaluate_policy(
                    env,
                    policy,
                    episodes=eval_episodes,
                    max_steps=10 if smoke_test else None,
                    allow_debug_truncation=bool(smoke_test),
                )
                metrics.update(
                    {
                        "sensitivity_parameter": cli_parameter,
                        "sensitivity_internal_key": factor,
                        "sensitivity_name": factor,
                        "sensitivity_value": value,
                        "method": method,
                        "instance": instance_name,
                        "instance_size": instance_name,
                        "seed": seed,
                        "runtime_sec": time.time() - started,
                        "smoke_mode": bool(smoke_test),
                        "whether_instance_regenerated": regenerated,
                        "whether_offline_resolved": offline_resolved,
                        "offline_status": offline_status,
                        "whether_policy_retrained": retrained,
                        "policy_source": "retrained" if retrained else "reused_default",
                        "scenario_token": json.dumps(scenario_token, sort_keys=True),
                    }
                )
                rows.append(metrics)

    if not rows:
        raise ValueError("Sensitivity produced no rows.")
    _validate_combinations(rows, values, test_seeds, methods)
    _write_rows(rows, out_csv)

    grouped = {}
    for method in methods:
        for value in values:
            subset = [
                row
                for row in rows
                if row["method"] == method
                and str(row["sensitivity_value"]) == str(value)
                and row.get("offline_status") not in {"infeasible", "failed"}
            ]
            if subset:
                grouped[f"{method}:{value}"] = aggregate(subset)
    Path(out_csv).with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": {
                    "sensitivity_parameter": cli_parameter,
                    "sensitivity_internal_key": factor,
                    "values": values,
                    "seeds": test_seeds,
                    "methods": methods,
                },
                "aggregated": grouped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    related = {
        "passenger_intensity": "onboard_passenger_delay_passenger_min",
        "station_power_capacity": "station_power_overload_amount",
        "chargers_per_station": "charger_utilization",
    }.get(factor)
    if related:
        related_values = [
            float(row.get(related, 0.0))
            for row in rows
            if row.get("offline_status") not in {"infeasible", "failed"}
        ]
        if related_values and len({round(item, 10) for item in related_values}) == 1:
            print(
                f"[SENSITIVITY][warning] factor={factor} produced identical related "
                f"metric '{related}' across tested values.",
                flush=True,
            )
    return rows
