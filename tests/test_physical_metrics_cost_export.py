import csv

import pytest

from src.env.ebus_drone_env import EBusDroneEnv
from src.env.reward import compute_reward
from src.harness.benchmark_runner import run_benchmark


class ZeroPolicy:
    def select_action(self, obs, mask, info_ctx):
        return 0


def _metrics_env(seed=1):
    return EBusDroneEnv(smoke_test=True)


def test_physical_cost_uses_nonnegative_components_not_final_clipping():
    reward, rc = compute_reward(
        {
            "passenger_delay": 10.0,
            "parcel_lateness": -3.0,
            "energy_cost": 2.0,
            "power_overload": -5.0,
            "battery_safety": 1.0,
            "locker_overflow": -7.0,
        },
        {"alpha_1": 1.0, "alpha_2": 1.0, "alpha_3": 1.0, "alpha_4": 1.0, "alpha_5": 1.0, "alpha_6": 1.0},
    )
    assert rc["parcel_lateness"] == 0.0
    assert rc["power_overload"] == 0.0
    assert rc["locker_overflow"] == 0.0
    assert rc["total_cost"] == pytest.approx(13.0)
    assert rc["step_physical_cost"] == pytest.approx(13.0)
    assert reward == pytest.approx(-13.0)


def test_transition_cost_is_not_negative_when_signed_deltas_decrease():
    env = EBusDroneEnv(smoke_test=True)
    before = {
        "passenger_delay": 5.0,
        "parcel_lateness": 4.0,
        "late_delivery_count": 1.0,
        "delivered_count": 2.0,
        "energy_consumption": 3.0,
        "power_overload": 10.0,
        "battery_violation": 2.0,
        "locker_overflow": 8.0,
        "bus_charging_energy_kwh": 2.0,
        "drone_charging_energy_kwh": 1.0,
        "power_overload_duration": 6.0,
        "locker_overflow_duration": 5.0,
        "locker_overflow_amount": 4.0,
    }
    after = dict(before)
    after.update({"power_overload": 1.0, "locker_overflow": 0.0, "battery_violation": 0.0})
    reward, rc = env._build_transition_reward(before, after, terminal_penalty=0.0)
    assert rc["power_overload"] == 0.0
    assert rc["locker_overflow"] == 0.0
    assert rc["battery_safety"] == 0.0
    assert rc["total_cost"] == 0.0
    assert reward == 0.0


def test_benchmark_csv_exports_unit_aware_physical_metric_names(tmp_path):
    out_csv = tmp_path / "benchmark.csv"
    rows = run_benchmark(
        ["no_charging"],
        str(out_csv),
        env_builder=_metrics_env,
        instance_name="small",
        test_seeds=[1],
        cfg={"paths": {"outputs": str(tmp_path)}, "rl": {"benchmark_eval_episodes": 1}},
        smoke_test=True,
    )
    assert rows[0]["total_cost"] >= 0.0
    assert "total_reward" in rows[0]
    header = next(csv.reader(out_csv.open(newline="")))
    for col in [
        "onboard_passenger_delay_passenger_min",
        "average_excess_dwell_time_min",
        "total_bus_operating_delay_min",
        "parcel_lateness_parcel_min",
        "average_locker_holding_time_min",
        "average_charging_duration_min",
        "average_positive_charging_duration_min",
        "total_charging_duration_min",
        "mean_requested_charging_duration_min",
        "mean_executed_charging_duration_min",
        "mean_requested_action_index",
        "mean_executed_action_index",
    ]:
        assert col in header
    assert "average_charging_duration" not in header
    assert "mean_requested_action" not in header
    assert "mean_executed_action" not in header
    for col in header:
        if "action_index" not in col:
            assert not col.endswith("_action")
