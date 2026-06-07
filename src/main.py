from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

from src.data_generation.instance_writer import write_instance
from src.data_generation.scenario_generator import generate_instance, generate_scenario
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.ablation_runner import run_ablation
from src.harness.benchmark_runner import build_policy, normalize_method_name, run_benchmark, uniform_seconds_from_method
from src.harness.evaluator import evaluate_policy, save_eval_metrics
from src.harness.curve_export import export_eval_curves, aggregate_curves
from src.harness.sensitivity_runner import FACTOR_PATHS, run_sensitivity
from src.harness.trainer import train_agent
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_io import load_offline_assignment, write_assignment
from src.offline.assignment_solver import solve_assignment
from src.offline.assignment_validator import summarize_assignment_flows
from src.utils.config import load_instance, load_scenario, load_yaml, validate_config
from src.utils.metrics import REQUIRED_PAPER_METRICS
from src.utils.random_seed import set_seed

SENSITIVITY_PARAMETER_ALIASES = {
    "passenger_demand_intensity": "passenger_intensity",
    "chargers_per_station": "chargers_per_station",
    "charging_power": "charging_power",
    "station_power_capacity": "station_power_capacity",
    "parcel_demand": "num_customers",
    "trip_freight_capacity": "bus_freight_capacity",
    "drone_resources": "drones_per_station",
    "locker_capacity": "locker_capacity",
    "freight_trip_availability": "freight_trip_availability",
}

VALID_METHODS = {
    "uniform", "uniform_15", "uniform_30", "uniform_45", "uniform_60", "uniform_120",
    "dwell_based_greedy", "dwell_greedy", "battery_threshold",
    "dqn_dr", "ddqn_dr", "am_ddqn_dr", "am_dueling_ddqn_dr",
    "no_charging", "max_feasible",
}


def _require_exists(path: Path, msg: str):
    if not path.exists():
        raise FileNotFoundError(msg)


def run_generate(cfg, instance_name: str, seed: int):
    cfg_file = Path(f"configs/instances/{instance_name}.yaml")
    _require_exists(cfg_file, f"Missing instance file: {cfg_file}")
    instance_cfg = load_yaml(str(cfg_file))
    out_root = f"{cfg['paths']['data_generated']}/{instance_name}"
    instance = generate_instance(cfg, instance_cfg, seed)
    write_instance(instance, out_root, f"instance_seed_{seed}")
    write_instance(generate_scenario(cfg, instance, seed, 0), out_root, f"scenario_0_seed_{seed}")


def run_offline(cfg, instance_name: str, seed: int):
    off = cfg.get("offline", {})
    solver = str(off.get("solver", "scipy"))
    solver_log = bool(off.get("solver_log", True))
    print(f"[offline] instance={instance_name} seed={seed} phase=load_instance started", flush=True)
    instance = load_instance(instance_name, seed, cfg["paths"]["data_generated"])
    print(f"[offline] instance={instance_name} seed={seed} phase=load_instance completed", flush=True)
    print(f"[offline] instance={instance_name} seed={seed} phase=build_assignment_data started", flush=True)
    data = build_assignment_data(instance)
    print(f"[offline] instance={instance_name} seed={seed} phase=build_assignment_data completed customers={len(data.customers)} freight_trips={len(data.trips)} stations={len(data.stations)}", flush=True)
    print(f"[offline] instance={instance_name} seed={seed} phase=solve_assignment started solver={solver}", flush=True)
    result = solve_assignment(
        data,
        allow_greedy_fallback=bool(off.get("allow_greedy_fallback", False)),
        solver=solver,
        time_limit_sec=off.get("time_limit_sec"),
        mip_gap=off.get("mip_gap"),
        solver_disp=bool(off.get("solver_disp", False)),
        solver_log=solver_log,
        auto_fallback_solver=off.get("auto_fallback_solver"),
    )
    print(f"[offline] instance={instance_name} seed={seed} phase=solve_assignment completed solver={solver} status={result.status} objective={result.objective_value} runtime_sec={result.runtime_sec} assigned={result.number_assigned_customers}/{result.number_customers}", flush=True)
    out = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    metadata = {"instance_name": instance_name, "seed": seed, **summarize_assignment_flows(data, result), **(result.metadata or {})}
    write_assignment(result, str(out), metadata=metadata)
    print(f"[offline] instance={instance_name} seed={seed} phase=write_assignment completed path={out}", flush=True)


def _read_applied_sensitivity_value(instance: dict, factor: str):
    stations = instance.get("stations", {}).get("stations", [])
    station = stations[0] if stations else {}
    readers = {
        "passenger_intensity": lambda: instance.get("_sensitivity", {}).get("passenger_intensity"),
        "num_customers": lambda: len(instance.get("customers", [])),
        "customer_number": lambda: len(instance.get("customers", [])),
        "chargers_per_station": lambda: station.get("chargers"),
        "drones_per_station": lambda: station.get("drones"),
        "locker_capacity": lambda: station.get("locker_capacity_kg"),
        "bus_freight_capacity": lambda: instance.get("bus", {}).get("freight_capacity_kg"),
        "station_unloading_capacity": lambda: instance.get("parcel", {}).get("unloading_capacity_kg_per_stop"),
        "station_power_capacity": lambda: station.get("station_power_capacity_kw"),
        "charging_power": lambda: station.get("pantograph_power_kw"),
        "initial_full_batteries": lambda: station.get("initial_fully_charged_batteries"),
        "max_charging_duration": lambda: instance.get("charging", {}).get("max_single_stop_seconds"),
        "freight_trip_availability": lambda: instance.get("network", {}).get("num_freight_carrying_trips"),
        "parcel_intensity": lambda: instance.get("_sensitivity", {}).get("parcel_intensity"),
        "base_load_perturbation_intensity": lambda: instance.get("power", {}).get("disturbance_std_kw"),
    }
    reader = readers.get(factor)
    return reader() if reader is not None else None


def _sensitivity_hooks(base_cfg: dict):
    def _regen(cfg_mod, instance_name: str, seed: int, factor: str, value: float):
        cfg_file = Path(f"configs/instances/{instance_name}.yaml")
        instance_cfg = load_yaml(str(cfg_file))
        if factor in {"num_customers", "customer_number", "parcel_intensity"}:
            baseline = int(instance_cfg.get("num_customers", 0))
            target = int(round(float(value))) if factor in {"num_customers", "customer_number"} else int(round(baseline * float(value)))
            instance_cfg["num_customers"] = max(1, target)
        if factor == "freight_trip_availability":
            scheduled = int(instance_cfg.get("num_scheduled_trips", instance_cfg["num_scheduled_bus_trips"]))
            numeric_value = float(value)
            target = int(round(scheduled * numeric_value)) if 0.0 <= numeric_value <= 1.0 else int(round(numeric_value))
            instance_cfg["num_freight_carrying_trips"] = max(0, min(scheduled, target))

        generated_root = Path(cfg_mod["paths"]["data_generated"]) / instance_name
        instance_path = generated_root / f"instance_seed_{seed}.json"
        scenario_path = generated_root / f"scenario_0_seed_{seed}.json"
        # Remove generated data first so a failed generation cannot be confused with
        # a completed run. Structural factors separately clear their assignment before
        # re-solving; scenario-only factors intentionally reuse the baseline assignment.
        for artifact in (instance_path, scenario_path):
            artifact.unlink(missing_ok=True)

        instance = generate_instance(cfg_mod, instance_cfg, seed)
        instance.setdefault("_sensitivity", {})[factor] = value
        if factor in {"num_customers", "customer_number"} and len(instance.get("customers", [])) != int(instance_cfg["num_customers"]):
            raise ValueError("Sensitivity num_customers failed: generated customer count mismatch.")
        write_instance(instance, str(generated_root), f"instance_seed_{seed}")
        write_instance(generate_scenario(cfg_mod, instance, seed, 0), str(generated_root), f"scenario_0_seed_{seed}")
        actual = _read_applied_sensitivity_value(instance, factor)
        if actual is None:
            section, key = FACTOR_PATHS[factor]
            actual = cfg_mod.get(section, {}).get(key)
        return actual

    def _resolve(cfg_mod, instance_name: str, seed: int, _factor: str, _value: float):
        assignment_path = Path(cfg_mod["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
        assignment_path.unlink(missing_ok=True)
        try:
            run_offline(cfg_mod, instance_name, seed)
            return "resolved"
        except Exception as exc:
            assignment_path.unlink(missing_ok=True)
            print(f"[SENSITIVITY] offline_assignment_failed={type(exc).__name__}: {exc}", flush=True)
            return "failed"

    return {"regenerate_instance": _regen, "resolve_offline": _resolve}


def build_env(cfg, instance_name: str, seed: int, smoke_test: bool) -> EBusDroneEnv:
    assign_path = Path(cfg["paths"]["outputs"]) / "assignments" / f"offline_assignment_{instance_name}_seed_{seed}.json"
    _require_exists(assign_path, f"Missing offline assignment: {assign_path}. Run --mode offline first.")
    return EBusDroneEnv(config=cfg, instance=load_instance(instance_name, seed, cfg["paths"]["data_generated"]), scenario=load_scenario(instance_name, seed, generated_root=cfg["paths"]["data_generated"]), assignment=load_offline_assignment(instance_name, seed, outputs_root=cfg["paths"]["outputs"]), smoke_test=smoke_test)


def _save_run_metadata(run_dir: Path, cfg: dict, args: argparse.Namespace, plan: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "cli_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def _write_named_summary(source_csv: Path, target_csv: Path):
    _require_exists(source_csv, f"Missing source summary: {source_csv}")
    rows = list(csv.DictReader(source_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty summaries are invalid: {source_csv}")
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    with target_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _validate_args(args):
    if args.method and args.methods and args.method != "no_charging":
        raise ValueError("Use either --method or --methods, not both.")
    sensitivity_mode = args.mode == "sensitivity" or (args.mode == "pipeline" and args.experiment == "sensitivity")
    if not sensitivity_mode and (args.sensitivity or args.parameter or args.values):
        raise ValueError("Sensitivity arguments require --mode sensitivity or --mode pipeline --experiment sensitivity.")
    if sensitivity_mode and not args.values:
        raise ValueError("Sensitivity experiments require at least one value in --values.")
    if args.sensitivity and args.sensitivity not in FACTOR_PATHS:
        raise ValueError(f"Unknown sensitivity variable: {args.sensitivity}")
    methods_to_validate = list(args.methods or []) + ([args.method] if args.method else [])
    for m in methods_to_validate:
        nm = normalize_method_name(m)
        if m.startswith("uniform_") or nm.startswith("uniform_"):
            uniform_seconds_from_method(m)
            continue
        if m not in VALID_METHODS and nm not in VALID_METHODS:
            raise ValueError(f"Unknown method: {m}")
    if args.mode in ('benchmark', 'ablation', 'sensitivity', 'pipeline'):
        if args.smoke:
            pass
        elif args.max_steps is not None and not args.allow_truncated_for_testing:
            raise ValueError("Formal experiments require full horizon: --max-steps is disallowed unless --allow-truncated-for-testing is set.")


def _resolve_plan(args):
    instances = args.instances or [args.instance]
    seeds = args.seeds or ([args.seed] if args.seed is not None else [1])
    methods = args.methods or ([args.method] if args.method else ["no_charging"])
    if args.experiment == "scalability":
        instances = ["small", "medium", "large"]
    return {"mode": args.mode, "experiment": args.experiment, "instances": instances, "seeds": seeds, "methods": methods, "max_steps": args.max_steps, "smoke": bool(args.smoke)}


def _run_eval(cfg, instance, seed, method, args, rows):
    env = build_env(cfg, instance, seed, args.smoke)
    try:
        pol = build_policy(method, env=env, out_root=cfg['paths']['outputs'], checkpoint=args.checkpoint, train_if_missing=args.train_if_missing, smoke_test=args.smoke, cfg=cfg, seed=seed, instance_name=instance)
    except TypeError:
        pol = build_policy(method)
    eval_eps = int(args.episodes) if args.episodes is not None else int(cfg.get("rl", {}).get("evaluation_episodes", 1))
    per_ep = []
    for ep_i in range(eval_eps):
        rec = evaluate_policy(env, pol, episodes=1, max_steps=args.max_steps if args.max_steps is not None else (10 if args.smoke else None), allow_debug_truncation=bool(args.smoke or args.allow_truncated_for_testing))
        rec.update({"eval_episode": ep_i + 1, "method": normalize_method_name(method), "instance": instance, "seed": seed})
        per_ep.append(rec)
    m = {}
    numeric_keys = {k for r in per_ep for k, v in r.items() if isinstance(v, (int, float))}
    for k in per_ep[0].keys():
        if k in numeric_keys:
            vals = [float(r[k]) for r in per_ep]
            m[k] = sum(vals) / len(vals)
            m[f"{k}_std"] = math.sqrt(sum((x - m[k]) ** 2 for x in vals) / len(vals))
        else:
            m[k] = per_ep[-1][k]
    def _mean_std(key: str, prefix: str):
        vals = [float(r.get(key, 0.0)) for r in per_ep]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
        m[f"mean_{prefix}"] = mean
        m[f"std_{prefix}"] = std
        return mean
    mtc = _mean_std("total_cost", "total_cost")
    _mean_std("total_reward", "total_reward")
    _mean_std("onboard_passenger_delay_passenger_min", "onboard_passenger_delay_passenger_min")
    _mean_std("parcel_lateness_parcel_min", "parcel_lateness_parcel_min")
    mmb = _mean_std("minimum_bus_battery", "minimum_bus_battery")
    _mean_std("total_energy_consumption", "total_energy_consumption")
    m["success_rate"] = sum(1.0 for r in per_ep if str(r.get("termination_reason", "")) == "horizon_reached") / len(per_ep)
    m["battery_depletion_rate"] = sum(1.0 for r in per_ep if float(r.get("minimum_bus_battery", 1.0)) <= 0.0) / len(per_ep)
    m.update({'method': normalize_method_name(method), 'instance': instance, 'seed': seed, 'smoke': bool(args.smoke), 'sum_total_cost': sum(float(r.get("total_cost", 0.0)) for r in per_ep), 'sum_total_reward': sum(float(r.get("total_reward", 0.0)) for r in per_ep)})
    export_eval_curves(cfg['paths']['outputs'], instance, normalize_method_name(method), int(seed), per_ep, smoke=bool(args.smoke))
    rows.append(m)


def _export_tables(output_dir: Path, experiment_name: str, include_smoke: bool = False):
    def _read_rows(path: Path):
        _require_exists(path, f"Missing results CSV for export_tables: {path}")
        out_rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        if not out_rows:
            raise ValueError(f"Empty results: export_tables requires non-empty metrics rows: {path}")
        for row in out_rows:
            if "method" in row:
                row["method"] = normalize_method_name(row["method"])
        return out_rows

    def _validate_required(rows, *, experiment: str, instance: str, path: Path):
        missing = [k for k in REQUIRED_PAPER_METRICS if k not in rows[0]]
        if missing:
            raise KeyError(f"Missing metrics for export_tables at {path} (experiment={experiment}, instance={instance}): {missing}")

    def _is_full_horizon(row):
        if not include_smoke and str(row.get("smoke_mode", row.get("smoke", "false"))).lower() in {"1", "true", "yes"}:
            return False
        if str(row.get("offline_status", "")).lower() in {"failed", "infeasible"}:
            return False
        if str(row.get("full_horizon_completed", "false")).lower() in {"1", "true", "yes"}:
            return True
        return False

    def _agg(rows, group_keys):
        grouped = {}
        for r in rows:
            key = tuple(r.get(k, "") for k in group_keys)
            grouped.setdefault(key, []).append(r)
        out = []
        for key, rs in grouped.items():
            rec = {k: v for k, v in zip(group_keys, key)}
            rec["successful_full_horizon_runs"] = sum(1 for r in rs if _is_full_horizon(r))
            rec["failed_or_truncated_runs"] = len(rs) - rec["successful_full_horizon_runs"]
            for m in REQUIRED_PAPER_METRICS:
                vals = []
                for r in rs:
                    if not _is_full_horizon(r):
                        continue
                    try:
                        vals.append(float(r[m]))
                    except (TypeError, ValueError):
                        continue
                if not vals:
                    rec[f"{m}_mean"] = "MISSING"
                    rec[f"{m}_std"] = "MISSING"
                else:
                    mean = sum(vals) / len(vals)
                    std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
                    rec[f"{m}_mean"] = f"{mean:.6g}"
                    rec[f"{m}_std"] = f"{std:.6g}"
            out.append(rec)
        return out

    def _write_table(rows, out_csv: Path, out_tex: Path):
        if not rows:
            raise ValueError(f"No aggregated rows for export: {out_csv}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        header = " & ".join(fields) + " \\\\"
        tex = ["\\begin{tabular}{" + "l" * len(fields) + "}", header, "\\hline"]
        for r in rows:
            tex.append(" & ".join(str(r.get(c, "")) for c in fields) + " \\\\")
        tex.append("\\end{tabular}")
        out_tex.write_text("\n".join(tex), encoding="utf-8")

    exp_root = output_dir / "results"
    if experiment_name in {"benchmark", "overall"}:
        bench_root = exp_root / "benchmark"
        instance_files = list(bench_root.glob("*/summary.csv"))
        if not instance_files and (bench_root / "summary.csv").exists():
            rows = _read_rows(bench_root / "summary.csv")
            _validate_required(rows, experiment="benchmark", instance="unknown", path=bench_root / "summary.csv")
            _write_table(_agg(rows, ["instance", "method"]), output_dir / "tables" / "benchmark_overall.csv", output_dir / "tables" / "benchmark_overall.tex")
            return
        for p in instance_files:
            instance = p.parent.name
            rows = _read_rows(p)
            _validate_required(rows, experiment="benchmark", instance=instance, path=p)
            _write_table(_agg(rows, ["method"]), output_dir / "tables" / f"benchmark_{instance}_overall.csv", output_dir / "tables" / f"benchmark_{instance}_overall.tex")
    if experiment_name == "ablation":
        for p in (exp_root / "ablation").glob("*/summary.csv"):
            instance = p.parent.name
            rows = _read_rows(p)
            _validate_required(rows, experiment="ablation", instance=instance, path=p)
            _write_table(_agg(rows, ["method"]), output_dir / "tables" / f"ablation_{instance}.csv", output_dir / "tables" / f"ablation_{instance}.tex")
    if experiment_name == "sensitivity":
        for p in (exp_root / "sensitivity").glob("*/*.csv"):
            instance = p.parent.name
            factor = p.stem
            rows = _read_rows(p)
            _validate_required(rows, experiment="sensitivity", instance=instance, path=p)
            _write_table(_agg(rows, ["method", "sensitivity_value"]), output_dir / "tables" / f"sensitivity_{instance}_{factor}.csv", output_dir / "tables" / f"sensitivity_{instance}_{factor}.tex")
    if experiment_name == "scalability":
        rows_all = []
        for p in (exp_root / "scalability").glob("*/summary.csv"):
            instance = p.parent.name
            rows = _read_rows(p)
            _validate_required(rows, experiment="scalability", instance=instance, path=p)
            rows_all.extend(rows)
        _write_table(_agg(rows_all, ["instance", "method"]), output_dir / "tables" / "scalability_summary.csv", output_dir / "tables" / "scalability_summary.tex")




def _has_non_smoke_results(output_root: Path) -> bool:
    for csv_path in (output_root / "results").glob("**/*.csv"):
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
        for row in rows:
            is_smoke = str(row.get("smoke_mode", row.get("smoke", "false"))).lower() in {"1", "true", "yes"}
            if not is_smoke:
                return True
    return False


def _run_formal_preflight(plan: dict, args: argparse.Namespace, cfg: dict):
    if plan.get("smoke"):
        raise ValueError("Formal experiment preflight failed: smoke must be false.")
    if plan.get("max_steps") is not None:
        raise ValueError("Formal experiment preflight failed: max_steps must be None for full-horizon runs.")
    if not REQUIRED_PAPER_METRICS:
        raise ValueError("Formal experiment preflight failed: required metrics list is empty.")
    if not plan.get("seeds"):
        raise ValueError("Formal experiment preflight failed: random seed list must be non-empty.")
    if not plan.get("methods"):
        raise ValueError("Formal experiment preflight failed: method list must be non-empty.")
    if any(int(s) <= 0 for s in plan["seeds"]):
        raise ValueError(f"Formal experiment preflight failed: invalid seeds {plan['seeds']}.")
    if any(not str(i).strip() for i in plan["instances"]):
        raise ValueError("Formal experiment preflight failed: instance size/name must be non-empty.")
    output_root = Path(cfg['paths']['outputs'])
    if (output_root / 'results').exists() and not _has_non_smoke_results(output_root):
        raise ValueError("Formal experiment preflight failed: output directory contains only smoke/stale smoke results.")


def _run_smoke_validation(cfg: dict, args: argparse.Namespace):
    smoke_cfg = deepcopy(cfg)
    smoke_cfg['paths']['outputs'] = str(Path(cfg['paths']['outputs']) / 'smoke')
    instance = 'small'
    seed = 1
    run_generate(smoke_cfg, instance, seed)
    run_offline(smoke_cfg, instance, seed)
    env = build_env(smoke_cfg, instance, seed, True)
    obs, _ = env.reset(seed=seed)
    if obs is None or len(obs) == 0:
        raise ValueError('Smoke validation failed: environment reset did not return a valid observation.')
    mask = env.get_action_mask()
    if mask is None or len(mask) == 0 or sum(1 for v in mask if int(v) > 0) <= 0:
        raise ValueError('Smoke validation failed: action mask has no feasible action.')
    _, _, _, _, info = env.step(0)
    if 'reward_components' not in info:
        raise KeyError('Smoke validation failed: env.step() info missing reward_components.')
    if not isinstance(info['reward_components'], dict):
        raise TypeError('Smoke validation failed: reward_components must be a dict.')
    from src.low_level.drone_dispatch_solver import solve_station_dispatch
    assignments, n_assigned = solve_station_dispatch(idle_drone_ids=['d1'], full_batteries=1, feasible_waiting=[{'id': 1, 'request_time': 0.0, 'deadline': 10.0}], now=0.0)
    if n_assigned <= 0 or not assignments:
        raise ValueError('Smoke validation failed: low-level dispatch could not assign parcel with available drone and battery.')
    rows = []
    _run_eval(smoke_cfg, instance, seed, 'uniform_30', args, rows)
    missing = [k for k in REQUIRED_PAPER_METRICS if k not in rows[0]]
    if missing:
        raise KeyError(f'Smoke validation failed: missing required paper metrics: {missing}')
    out = Path(smoke_cfg['paths']['outputs']) / 'metrics' / 'validate_pipeline.csv'
    save_eval_metrics(rows, str(out))
    content = out.read_text(encoding='utf-8').strip()
    if not content or 'smoke' not in content.lower():
        raise ValueError('Smoke validation failed: result writer did not produce non-empty smoke output.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['generate', 'offline', 'train', 'eval', 'benchmark', 'ablation', 'sensitivity', 'pipeline', 'validate_pipeline', 'export_tables'])
    ap.add_argument('--experiment', choices=['overall', 'ablation', 'scalability', 'sensitivity'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--instance', default='small')
    ap.add_argument('--instances', nargs='+')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', nargs='+', type=int)
    ap.add_argument('--method', default='no_charging')
    ap.add_argument('--methods', nargs='+')
    ap.add_argument('--episodes', type=int)
    ap.add_argument('--smoke', action='store_true'); ap.add_argument('--smoke-test', action='store_true')
    ap.add_argument('--checkpoint'); ap.add_argument('--train-if-missing', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--factor'); ap.add_argument('--values', nargs='+', type=float)
    ap.add_argument('--parameter')
    ap.add_argument('--sensitivity')
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--allow-truncated-for-testing', action='store_true')
    ap.add_argument('--output-dir')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--include-smoke', action='store_true')
    ap.add_argument('--log-interval', type=int, default=10)

    ap.add_argument('--offline-solver', choices=['gurobi', 'scipy'])
    ap.add_argument('--offline-time-limit', type=float)
    ap.add_argument('--offline-mip-gap', type=float)
    ap.add_argument('--offline-log', action='store_true')
    ap.add_argument('--no-offline-log', action='store_true')
    ap.add_argument('--offline-solver-disp', action='store_true')
    args = ap.parse_args()
    if args.smoke_test:
        args.smoke = True
    if args.parameter and args.sensitivity:
        raise ValueError("Use either --parameter or --sensitivity, not both.")
    if args.parameter:
        if args.parameter not in SENSITIVITY_PARAMETER_ALIASES:
            raise ValueError(f"Unknown --parameter: {args.parameter}")
        args.sensitivity = SENSITIVITY_PARAMETER_ALIASES[args.parameter]
    _validate_args(args)
    cfg = load_yaml(args.config)
    validate_config(cfg)
    if args.output_dir:
        cfg['paths']['outputs'] = args.output_dir
    cfg.setdefault('offline', {})
    if args.offline_solver:
        cfg['offline']['solver'] = args.offline_solver
    if args.offline_time_limit is not None:
        cfg['offline']['time_limit_sec'] = args.offline_time_limit
    if args.offline_mip_gap is not None:
        cfg['offline']['mip_gap'] = args.offline_mip_gap
    if args.offline_solver_disp:
        cfg['offline']['solver_disp'] = True
    if args.offline_log:
        cfg['offline']['solver_log'] = True
    if args.no_offline_log:
        cfg['offline']['solver_log'] = False
    plan = _resolve_plan(args)
    print(json.dumps(plan, indent=2))
    run_dir = Path(cfg['paths']['outputs']) / 'runs' / f"{plan['mode']}_{(args.experiment or 'single')}"
    _save_run_metadata(run_dir, cfg, args, plan)

    if args.mode == 'generate':
        for i in plan['instances']:
            for s in plan['seeds']:
                run_generate(cfg, i, s)
        return
    if args.mode == 'offline':
        for i in plan['instances']:
            for s in plan['seeds']:
                run_offline(cfg, i, s)
        return
    if args.mode == 'train':
        for i in plan['instances']:
            for seed in plan['seeds']:
                set_seed(seed, deterministic=bool(cfg.get('rl', {}).get('deterministic', False)))
                env = build_env(cfg, i, seed, args.smoke)
                train_agent(
                    env,
                    method=plan['methods'][0],
                    episodes=args.episodes if args.episodes is not None else (2 if args.smoke else None),
                    max_steps=args.max_steps if args.max_steps is not None else (10 if args.smoke else None),
                    smoke_test=args.smoke,
                    out_root=cfg['paths']['outputs'],
                    cfg=cfg,
                    seed=seed,
                    instance_name=i,
                    resume=bool(args.resume),
                    checkpoint=args.checkpoint,
                    log_interval=args.log_interval,
                )
        return
    if args.mode == 'eval':
        rows = []
        for i in plan['instances']:
            for seed in plan['seeds']:
                _run_eval(cfg, i, seed, plan['methods'][0], args, rows)
        raw_method = plan['methods'][0]
        out_method = normalize_method_name(raw_method)
        out = Path(cfg['paths']['outputs']) / 'metrics' / f"eval_{out_method}_{plan['instances'][0]}_seed_{plan['seeds'][0]}.csv"
        save_eval_metrics(rows, str(out))
        if str(raw_method).startswith("uniform_") and str(raw_method) != out_method:
            legacy_out = Path(cfg['paths']['outputs']) / 'metrics' / f"eval_{raw_method}_{plan['instances'][0]}_seed_{plan['seeds'][0]}.csv"
            save_eval_metrics(rows, str(legacy_out))
        print(json.dumps(rows)); return
    if args.mode == 'validate_pipeline':
        _run_smoke_validation(cfg, args)
        return
    if args.mode in ('benchmark', 'ablation', 'sensitivity', 'pipeline'):
        if args.smoke:
            cfg['paths']['outputs'] = str(Path(cfg['paths']['outputs']) / 'smoke')
        elif args.max_steps is not None and not args.allow_truncated_for_testing:
            raise ValueError("Formal experiments require full horizon with --max-steps unset. Use --allow-truncated-for-testing to override for tests.")
        if args.mode in ('benchmark', 'ablation', 'sensitivity') and not args.allow_truncated_for_testing:
            _run_formal_preflight(plan, args, cfg)
        pipeline_sensitivity = args.mode == 'pipeline' and args.experiment == 'sensitivity'
        if args.mode == 'pipeline':
            for i in plan['instances']:
                for s in plan['seeds']:
                    print(f"[pipeline] instance={i} seed={s} phase=generate started", flush=True)
                    run_generate(cfg, i, s)
                    print(f"[pipeline] instance={i} seed={s} phase=offline started", flush=True)
                    run_offline(cfg, i, s)
        if args.mode == 'benchmark' or (args.mode == 'pipeline' and not pipeline_sensitivity):
            for i in plan['instances']:
                out = Path(cfg['paths']['outputs']) / 'results' / (args.experiment or 'benchmark') / i / 'summary.csv'
                if out.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists and overwrite is disabled: {out}")
                if args.mode == 'pipeline':
                    for s in plan['seeds']:
                        for m in [normalize_method_name(x) for x in plan['methods']]:
                            if m in {"dqn_dr","ddqn_dr","am_ddqn_dr","am_dueling_ddqn_dr"} and args.train_if_missing:
                                print(f"[pipeline] instance={i} seed={s} method={m} phase=train started", flush=True)
                            print(f"[pipeline] instance={i} seed={s} method={m} phase=eval started", flush=True)
                run_benchmark(plan['methods'], str(out), env_builder=lambda sd, _i=i: build_env(cfg, _i, sd, args.smoke), instance_name=i, test_seeds=plan['seeds'], cfg=cfg, smoke_test=args.smoke, train_if_missing=args.train_if_missing, log_interval=args.log_interval)
                for _m in [normalize_method_name(x) for x in plan['methods']]:
                    aggregate_curves(cfg['paths']['outputs'], i, _m, 'train')
                    aggregate_curves(cfg['paths']['outputs'], i, _m, 'eval')
                _write_named_summary(out, Path(cfg['paths']['outputs']) / "results" / "overall_performance_summary.csv")
                if args.mode == 'pipeline':
                    for s in plan['seeds']:
                        print(f"[pipeline] instance={i} seed={s} completed", flush=True)
        if args.mode == 'ablation':
            for i in plan['instances']:
                out = Path(cfg['paths']['outputs']) / 'results' / 'ablation' / i / 'summary.csv'
                run_ablation(str(out), env_builder=lambda sd, _i=i: build_env(cfg, _i, sd, args.smoke), instance_name=i, test_seeds=plan['seeds'], cfg=cfg, smoke_test=args.smoke, train_if_missing=args.train_if_missing)
                _write_named_summary(out, Path(cfg['paths']['outputs']) / "results" / "ablation_summary.csv")
        if args.mode == 'sensitivity' or pipeline_sensitivity:
            vals = args.values
            cfg['_sensitivity_hooks'] = _sensitivity_hooks(cfg)
            factors = [args.sensitivity] if args.sensitivity else list(FACTOR_PATHS.keys())
            all_summary_rows = []
            for i in plan['instances']:
                instance_rows = []
                sensitivity_dir = Path(cfg['paths']['outputs']) / 'results' / 'sensitivity' / i
                summary_out = sensitivity_dir / 'summary.csv'
                if summary_out.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists and overwrite is disabled: {summary_out}")
                for factor in factors:
                    out = sensitivity_dir / f'{factor}.csv'
                    rows = run_sensitivity(
                        plan['methods'],
                        str(out),
                        env_builder=lambda sd, c, _i=i: build_env(c, _i, sd, args.smoke),
                        instance_name=i,
                        test_seeds=plan['seeds'],
                        cfg=cfg,
                        factor=factor,
                        values=vals,
                        smoke_test=args.smoke,
                        train_if_missing=args.train_if_missing,
                        sensitivity_parameter=args.parameter or factor,
                    )
                    instance_rows.extend(rows)
                save_eval_metrics(instance_rows, str(summary_out))
                all_summary_rows.extend(instance_rows)
            save_eval_metrics(all_summary_rows, str(Path(cfg['paths']['outputs']) / "results" / "sensitivity_summary.csv"))
        return
    if args.mode == 'export_tables':
        _export_tables(Path(cfg['paths']['outputs']), args.experiment or 'benchmark', include_smoke=bool(args.include_smoke))
        return


if __name__ == '__main__':
    main()
