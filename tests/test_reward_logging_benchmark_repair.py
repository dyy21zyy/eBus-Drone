import csv
from pathlib import Path

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.trainer import train_agent
from src.harness.evaluator import evaluate_policy
from src.harness.benchmark_runner import run_benchmark


class ZeroPolicy:
    def select_action(self, obs, mask, info_ctx):
        return 0


def test_terminal_penalty_changes_final_reward():
    env = EBusDroneEnv(smoke_test=True, config={"reward": {"eta_l_term": 1.0, "eta_u_term": 1.0}})
    before = {
        "passenger_delay": 0.0, "parcel_lateness": 0.0, "late_delivery_count": 0.0, "delivered_count": 0.0,
        "energy_consumption": 0.0, "power_overload": 0.0, "battery_violation": 0.0, "locker_overflow": 0.0,
        "bus_charging_energy_kwh": 0.0, "drone_charging_energy_kwh": 0.0, "power_overload_duration": 0.0,
        "locker_overflow_duration": 0.0, "locker_overflow_amount": 0.0,
    }
    after = dict(before)
    r0, _ = env._build_transition_reward(before, after, terminal_penalty=0.0)
    r1, rc1 = env._build_transition_reward(before, after, terminal_penalty=3.0)
    assert rc1["parcel_lateness"] == 3.0
    assert r1 < r0


def test_train_log_reads_real_episode_metrics(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    cfg = {"paths": {"outputs": str(tmp_path)}, "rl": {"episodes": 1, "max_steps_per_episode": 2, "device": "cpu"}}
    train_agent(env, method="dqn_dr", episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg=cfg, seed=1, instance_name="small")
    log = tmp_path / "metrics" / "train_log_dqn_dr_small_seed_1.csv"
    row = next(csv.DictReader(log.open()))
    episode_metrics = env.get_episode_metrics()
    assert float(row["onboard_passenger_delay_passenger_min"]) == float(episode_metrics["onboard_passenger_delay_passenger_min"])
    assert float(row["total_energy_consumption"]) == float(episode_metrics["total_energy_consumption"])


def test_eval_output_includes_required_fields():
    env = EBusDroneEnv(smoke_test=True)
    out = evaluate_policy(env, ZeroPolicy(), episodes=1, max_steps=3, allow_debug_truncation=True)
    for k in [
        "total_cost", "total_reward", "onboard_passenger_delay_passenger_min", "average_excess_dwell_time_min",
        "total_bus_operating_delay_min", "parcel_lateness_parcel_min", "late_delivery_count", "undelivered_parcel_count",
        "bus_battery_violation" if "bus_battery_violation" in out else "battery_safety_violation_count",
        "station_power_overload_amount", "locker_overflow_amount", "total_energy_consumption",
        "average_charging_duration_min", "valid_charging_opportunity_count", "episode_length_decisions",
    ]:
        assert k in out


def test_summary_csv_uses_canonical_method_names(tmp_path):
    cfg = {"paths": {"outputs": str(tmp_path)}, "rl": {"benchmark_eval_episodes": 1}}
    out_csv = tmp_path / "summary.csv"
    run_benchmark(["proposed"], str(out_csv), env_builder=lambda seed: EBusDroneEnv(smoke_test=True), instance_name="small", test_seeds=[1], cfg=cfg, smoke_test=True, train_if_missing=True)
    row = next(csv.DictReader(out_csv.open()))
    assert row["method"] == "am_dueling_ddqn_dr"
    assert "total_cost" in row and "onboard_passenger_delay_passenger_min" in row
