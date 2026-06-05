from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _safe_num(v):
    try:
        if v is None or v == "":
            return math.nan
        return float(v)
    except Exception:
        return math.nan


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _atomic_to_csv(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def export_training_curve(out_root: str, instance: str, method: str, seed: int, rows: list[dict], smoke: bool = False) -> Path:
    curve_dir = Path(out_root) / "curves" / "train" / instance
    _ensure_dir(curve_dir)
    path = curve_dir / f"{method}_seed_{seed}_train_curve.csv"
    df = pd.DataFrame(rows)
    required = [
        "episode","method","instance","seed","episode_reward","episode_cost","total_reward","total_cost",
        "moving_avg_reward_10","moving_avg_reward_50","moving_avg_cost_10","moving_avg_cost_50","epsilon",
        "loss_mean","loss_last","steps","episode_length_decisions","termination_reason","minimum_bus_battery",
        "battery_safety_violation_count","onboard_passenger_delay_passenger_min","parcel_lateness_parcel_min","late_delivery_count","runtime_sec","smoke",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = math.nan
    df["smoke"] = bool(smoke)
    _atomic_to_csv(df, path)
    _plot_training(df, Path(out_root), instance, method, seed)
    return path


def _plot_training(df: pd.DataFrame, out_root: Path, instance: str, method: str, seed: int):
    fig_dir = out_root / "figures" / "train" / instance
    _ensure_dir(fig_dir)
    x = pd.to_numeric(df.get("episode"), errors="coerce")
    def _save_plot(filename, ys, title, ylabel):
        plt.figure(figsize=(8, 4.5))
        for label, y in ys:
            yy = pd.to_numeric(y, errors="coerce")
            if yy.notna().any():
                plt.plot(x, yy, label=label)
        plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        if len(ys) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
    _save_plot(f"{method}_seed_{seed}_training_reward_curve.png", [("episode_reward", df.get("episode_reward")), ("total_reward", df.get("total_reward"))], f"Training Reward Curve ({method})", "Reward")
    _save_plot(f"{method}_seed_{seed}_training_cost_curve.png", [("episode_cost", df.get("episode_cost")), ("total_cost", df.get("total_cost"))], f"Training Cost Curve ({method})", "Cost")
    if pd.to_numeric(df.get("loss_mean"), errors="coerce").notna().any() or pd.to_numeric(df.get("loss_last"), errors="coerce").notna().any():
        _save_plot(f"{method}_seed_{seed}_training_loss_curve.png", [("loss_mean", df.get("loss_mean")), ("loss_last", df.get("loss_last"))], f"Training Loss Curve ({method})", "Loss")
    if pd.to_numeric(df.get("minimum_bus_battery"), errors="coerce").notna().any() or pd.to_numeric(df.get("battery_safety_violation_count"), errors="coerce").notna().any():
        _save_plot(f"{method}_seed_{seed}_training_battery_safety_curve.png", [("minimum_bus_battery", df.get("minimum_bus_battery")), ("battery_safety_violation_count", df.get("battery_safety_violation_count"))], f"Training Battery Safety ({method})", "Battery/Safety")


def export_eval_curves(out_root: str, instance: str, method: str, seed: int, rows: list[dict], smoke: bool = False):
    curve_dir = Path(out_root) / "curves" / "eval" / instance
    _ensure_dir(curve_dir)
    eval_path = curve_dir / f"{method}_seed_{seed}_eval_curve.csv"
    cum_path = curve_dir / f"{method}_seed_{seed}_eval_cumulative_curve.csv"
    df = pd.DataFrame(rows)
    required = ["eval_episode","method","instance","seed","total_reward","total_cost","onboard_passenger_delay_passenger_min","total_bus_operating_delay_min","parcel_lateness_parcel_min","late_delivery_count","undelivered_parcel_count","minimum_bus_battery","battery_safety_violation_count","total_energy_consumption","average_charging_duration_min","average_positive_charging_duration_min","total_charging_duration_min","mean_requested_charging_duration_min","mean_executed_charging_duration_min","mean_requested_action_index","mean_executed_action_index","valid_charging_opportunity_count","charger_utilization","station_power_overload_amount","station_power_overload_duration","locker_overflow_amount","locker_overflow_duration","termination_reason","full_horizon_completed","runtime_sec","smoke"]
    for c in required:
        if c not in df.columns:
            df[c] = math.nan
    df["smoke"] = bool(smoke)
    _atomic_to_csv(df, eval_path)
    cdf = pd.DataFrame({"eval_episode": df["eval_episode"]})
    for col, out in [("total_cost","cumulative_mean_total_cost"), ("total_reward","cumulative_mean_total_reward")]:
        s = pd.to_numeric(df[col], errors="coerce")
        cdf[out] = s.expanding(min_periods=1).mean()
        cdf[out.replace("mean", "std")] = s.expanding(min_periods=1).std(ddof=0)
    cdf["cumulative_mean_passenger_delay"] = pd.to_numeric(df["onboard_passenger_delay_passenger_min"], errors="coerce").expanding(min_periods=1).mean()
    cdf["cumulative_mean_parcel_lateness"] = pd.to_numeric(df["parcel_lateness_parcel_min"], errors="coerce").expanding(min_periods=1).mean()
    cdf["cumulative_mean_minimum_bus_battery"] = pd.to_numeric(df["minimum_bus_battery"], errors="coerce").expanding(min_periods=1).mean()
    fh = pd.to_numeric(df["full_horizon_completed"], errors="coerce").fillna(0.0)
    cdf["cumulative_success_rate_full_horizon"] = fh.expanding(min_periods=1).mean()
    _atomic_to_csv(cdf, cum_path)
    _plot_eval(df, cdf, Path(out_root), instance, method, seed)
    return eval_path, cum_path


def _plot_eval(df, cdf, out_root, instance, method, seed):
    fig_dir = out_root / "figures" / "eval" / instance
    _ensure_dir(fig_dir)
    x = pd.to_numeric(df["eval_episode"], errors="coerce")
    def _plot(path, ys, title, ylabel):
        plt.figure(figsize=(8, 4.5))
        for lab, y in ys:
            yy = pd.to_numeric(y, errors="coerce")
            if yy.notna().any():
                plt.plot(x if len(yy)==len(x) else cdf["eval_episode"], yy, label=lab)
        plt.title(title); plt.xlabel("Evaluation episode"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        if len(ys) > 1: plt.legend()
        plt.tight_layout(); plt.savefig(fig_dir / path, dpi=300, bbox_inches="tight"); plt.close()
    _plot(f"{method}_seed_{seed}_eval_total_cost_curve.png", [("total_cost", df["total_cost"])], f"Evaluation Total Cost ({method})", "Total cost")
    _plot(f"{method}_seed_{seed}_eval_cumulative_mean_cost_curve.png", [("cumulative_mean_total_cost", cdf["cumulative_mean_total_cost"])], f"Evaluation Cumulative Mean Cost ({method})", "Cumulative mean cost")
    _plot(f"{method}_seed_{seed}_eval_reward_curve.png", [("total_reward", df["total_reward"])], f"Evaluation Reward ({method})", "Total reward")
    _plot(f"{method}_seed_{seed}_eval_service_quality_curve.png", [("parcel_lateness", df["parcel_lateness_parcel_min"]), ("late_delivery_count", df["late_delivery_count"])], f"Evaluation Service Quality ({method})", "Service quality")
    _plot(f"{method}_seed_{seed}_eval_battery_safety_curve.png", [("minimum_bus_battery", df["minimum_bus_battery"]), ("battery_safety_violation_count", df["battery_safety_violation_count"])], f"Evaluation Battery Safety ({method})", "Battery/Safety")


def aggregate_curves(out_root: str, instance: str, method: str, curve_type: str):
    idx_col = "episode" if curve_type == "train" else "eval_episode"
    suffix = "train_curve" if curve_type == "train" else "eval_curve"
    curve_dir = Path(out_root) / "curves" / curve_type / instance
    files = sorted(curve_dir.glob(f"{method}_seed_*_{suffix}.csv"))
    if len(files) < 2:
        return None
    dfs = [pd.read_csv(f) for f in files if f.exists()]
    merged = None
    for i, df in enumerate(dfs):
        if idx_col not in df.columns:
            continue
        cols = [idx_col] + [c for c in ["total_cost", "total_reward", "episode_reward", "minimum_bus_battery", "battery_safety_violation_count"] if c in df.columns]
        sdf = df[cols].copy().rename(columns={c: f"{c}_s{i}" for c in cols if c != idx_col})
        merged = sdf if merged is None else merged.merge(sdf, on=idx_col, how="outer")
    if merged is None:
        return None
    out = pd.DataFrame({idx_col: merged[idx_col]})
    for base in ["total_cost", "total_reward", "episode_reward", "minimum_bus_battery", "battery_safety_violation_count"]:
        seed_cols = [c for c in merged.columns if c.startswith(f"{base}_s")]
        if seed_cols:
            out[f"mean_{base}"] = merged[seed_cols].mean(axis=1, skipna=True)
            out[f"std_{base}"] = merged[seed_cols].std(axis=1, ddof=0, skipna=True)
    out = out.sort_values(idx_col)
    agg_name = f"{method}_{'train_curve_agg.csv' if curve_type=='train' else 'eval_curve_agg.csv'}"
    agg_path = curve_dir / agg_name
    _atomic_to_csv(out, agg_path)
    _plot_agg(out_root, instance, method, curve_type, out, idx_col)
    return agg_path


def _plot_agg(out_root, instance, method, curve_type, out, idx_col):
    fig_dir = Path(out_root) / "figures" / curve_type / instance
    _ensure_dir(fig_dir)
    x = out[idx_col]
    def _line_band(mean_col, std_col, fname, title, ylabel):
        if mean_col not in out.columns:
            return
        y = pd.to_numeric(out[mean_col], errors="coerce")
        s = pd.to_numeric(out.get(std_col), errors="coerce")
        plt.figure(figsize=(8, 4.5)); plt.plot(x, y, label="mean")
        if s.notna().any():
            plt.fill_between(x, y-s, y+s, alpha=0.2, label="±1 std")
        plt.title(title); plt.xlabel(idx_col); plt.ylabel(ylabel); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / fname, dpi=300, bbox_inches="tight"); plt.close()
    if curve_type == "train":
        _line_band("mean_total_reward", "std_total_reward", f"{method}_training_reward_curve_agg.png", f"Aggregated Training Reward ({method})", "Reward")
        _line_band("mean_total_cost", "std_total_cost", f"{method}_training_cost_curve_agg.png", f"Aggregated Training Cost ({method})", "Cost")
    else:
        _line_band("mean_total_cost", "std_total_cost", f"{method}_eval_total_cost_curve_agg.png", f"Aggregated Eval Total Cost ({method})", "Cost")
        _line_band("mean_total_cost", "std_total_cost", f"{method}_eval_cumulative_cost_curve_agg.png", f"Aggregated Eval Cumulative Cost ({method})", "Cost")
