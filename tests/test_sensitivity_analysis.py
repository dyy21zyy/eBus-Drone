from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from src.harness.sensitivity_runner import run_sensitivity
from src.main import main


def _metric_for_env(env, *_args, **_kwargs):
    value = float(env.applied_value)
    return {
        "total_cost": value * 10.0,
        "total_reward": -value,
        "full_horizon_completed": True,
        "termination_reason": "horizon_reached",
    }


def test_two_values_two_seeds_export_all_combinations(tmp_path, monkeypatch):
    monkeypatch.setattr("src.harness.sensitivity_runner.evaluate_policy", _metric_for_env)
    regenerated = []

    def regenerate(cfg, _instance, seed, _factor, value):
        regenerated.append((value, seed))
        return cfg["parcel"]["locker_capacity_kg"]

    cfg = {
        "paths": {"outputs": str(tmp_path)},
        "_sensitivity_hooks": {
            "regenerate_instance": regenerate,
            "resolve_offline": lambda *_args: "resolved",
        },
    }
    out = tmp_path / "summary.csv"
    rows = run_sensitivity(
        ["no_charging"],
        str(out),
        env_builder=lambda _seed, config: SimpleNamespace(
            applied_value=config["parcel"]["locker_capacity_kg"]
        ),
        instance_name="small",
        test_seeds=[1, 2],
        cfg=cfg,
        factor="locker_capacity",
        values=[15.0, 30.0],
        sensitivity_parameter="locker_capacity",
    )

    exported = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == len(exported) == 4
    assert set(exported[0]) >= {
        "sensitivity_parameter",
        "sensitivity_internal_key",
        "sensitivity_value",
        "method",
        "instance",
        "seed",
        "total_cost",
        "total_reward",
    }
    assert {float(row["sensitivity_value"]) for row in exported} == {15.0, 30.0}
    assert set(regenerated) == {(15.0, 1), (15.0, 2), (30.0, 1), (30.0, 2)}
    assert {float(row["total_cost"]) for row in exported} == {150.0, 300.0}


def test_failed_value_does_not_evaluate_or_copy_stale_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("src.harness.sensitivity_runner.evaluate_policy", _metric_for_env)
    evaluated = []

    def env_builder(seed, config):
        value = config["parcel"]["locker_capacity_kg"]
        evaluated.append((value, seed))
        return SimpleNamespace(applied_value=value)

    cfg = {
        "paths": {"outputs": str(tmp_path)},
        "_sensitivity_hooks": {
            "regenerate_instance": lambda cfg, *_args: cfg["parcel"]["locker_capacity_kg"],
            "resolve_offline": lambda _cfg, _instance, _seed, _factor, value: (
                "resolved" if value == 15.0 else "failed"
            ),
        },
    }
    out = tmp_path / "failed.csv"
    rows = run_sensitivity(
        ["no_charging"],
        str(out),
        env_builder=env_builder,
        instance_name="small",
        test_seeds=[1],
        cfg=cfg,
        factor="locker_capacity",
        values=[15.0, 30.0],
    )

    assert evaluated == [(15.0, 1)]
    failed = next(row for row in rows if row["sensitivity_value"] == 30.0)
    assert failed["offline_status"] == "failed"
    assert failed["total_cost"] is None
    assert failed["total_reward"] is None


def test_pipeline_sensitivity_routes_values_to_sensitivity_runner(tmp_path, monkeypatch):
    monkeypatch.setattr("src.main.run_generate", lambda *_args: None)
    monkeypatch.setattr("src.main.run_offline", lambda *_args: None)

    def fake_run_sensitivity(methods, out_csv, **kwargs):
        rows = []
        for value in kwargs["values"]:
            for seed in kwargs["test_seeds"]:
                for method in methods:
                    rows.append(
                        {
                            "sensitivity_parameter": kwargs["sensitivity_parameter"],
                            "sensitivity_internal_key": kwargs["factor"],
                            "sensitivity_value": value,
                            "method": method,
                            "instance": kwargs["instance_name"],
                            "seed": seed,
                            "total_cost": value,
                            "total_reward": -value,
                        }
                    )
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        return rows

    monkeypatch.setattr("src.main.run_sensitivity", fake_run_sensitivity)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--mode",
            "pipeline",
            "--experiment",
            "sensitivity",
            "--config",
            "configs/default.yaml",
            "--instance",
            "small",
            "--seeds",
            "1",
            "2",
            "--methods",
            "no_charging",
            "--parameter",
            "locker_capacity",
            "--values",
            "15",
            "30",
            "--output-dir",
            str(tmp_path),
            "--overwrite",
        ],
    )
    main()

    summary = tmp_path / "results" / "sensitivity" / "small" / "summary.csv"
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    assert len(rows) == 4
    assert {float(row["sensitivity_value"]) for row in rows} == {15.0, 30.0}
