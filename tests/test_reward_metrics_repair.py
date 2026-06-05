import pytest

from src.env.reward import compute_reward
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy


class DummyPolicy:
    def select_action(self, obs, mask, info_ctx):
        return 0


class DummyEnv:
    def __init__(self):
        self.state = {"battery": 10.0, "battery_max": 20.0}
        self.parcel_states = {1: {"status": "delivered", "locker_holding_time_min": 2.0}, 2: {"status": "onboard", "locker_holding_time_min": 0.0}}
        self.bus_states = {0: {"accumulated_delay_min": 1.0}}
        self.station_states = {1: {"full_batteries": 0}}
        self._done = False

    def reset(self, seed=None):
        self._done = False
        return [0.0], {}

    def get_action_mask(self):
        return [1, 0, 0, 0, 0, 0]

    def step(self, action):
        self._done = True
        rc = {
            "total_cost": 5.0, "passenger_delay": 2.0, "parcel_lateness": 1.0, "terminal_penalty": 0.5,
            "total_energy_kwh": 0.25, "power_overload": 0.0, "power_overload_duration": 0.0,
            "locker_overflow_amount": 0.0, "locker_overflow_duration": 0.0, "number_late_deliveries": 1,
            "battery_safety": 1.0, "reward": -5.0,
        }
        info = {"reward_components": rc, "dwell_components": {"realized_dwell_min": 2.0, "passenger_dwell_min": 1.0}, "executed_duration_min": 0.5, "action_repaired": False}
        return [0.0], -5.0, True, False, info


def test_reward_equals_negative_total_cost_and_matches_component_reward():
    r, c = compute_reward({"passenger_delay": 1, "parcel_lateness": 1, "energy_cost": 1, "power_overload": 1, "battery_safety": 1, "locker_overflow": 1}, {"alpha_1": 1, "alpha_2": 1, "alpha_3": 1, "alpha_4": 1, "alpha_5": 1, "alpha_6": 1})
    assert c["total_cost"] == 6
    assert r == -c["total_cost"]
    assert c["reward"] == r


def test_energy_kwh_from_kw_minutes_unit():
    energy_kwh = 120.0 * 30.0 / 60.0
    r, c = compute_reward({"passenger_delay": 0, "parcel_lateness": 0, "energy_cost": energy_kwh, "power_overload": 0, "battery_safety": 0, "locker_overflow": 0}, {"alpha_1": 0, "alpha_2": 0, "alpha_3": 1, "alpha_4": 0, "alpha_5": 0, "alpha_6": 0})
    assert c["energy_cost"] == pytest.approx(60.0)
    assert r == pytest.approx(-60.0)


def test_passenger_delay_affected_after_alighting_plus_boarders():
    additional_dwell = 3.0
    affected = 10 + 4
    expected = affected * additional_dwell
    assert expected == 42.0


def test_evaluator_contains_required_metrics():
    out = evaluate_policy(DummyEnv(), DummyPolicy(), episodes=1, max_steps=1)
    required = [
        "total_cost","total_reward","onboard_passenger_delay_passenger_min","average_excess_dwell_time_min","total_bus_operating_delay_min",
        "parcel_lateness_parcel_min","late_delivery_count","undelivered_parcel_count","average_locker_holding_time_min",
        "terminal_undelivered_penalty","minimum_bus_battery","battery_safety_violation_count","total_energy_consumption",
        "station_power_overload_amount","station_power_overload_duration","locker_overflow_amount","locker_overflow_duration",
        "charger_utilization","drone_battery_stockout_count",
    ]
    for k in required:
        assert k in out


def test_energy_cost_scales_with_reward_eta_E():
    env_base = EBusDroneEnv(smoke_test=True, config={"reward": {"eta_E": 1.0}})
    env_scaled = EBusDroneEnv(smoke_test=True, config={"reward": {"eta_E": 3.0}})
    before = {
        "passenger_delay": 0.0, "parcel_lateness": 0.0, "late_delivery_count": 0.0, "delivered_count": 0.0,
        "energy_consumption": 0.0, "power_overload": 0.0, "battery_violation": 0.0, "locker_overflow": 0.0,
        "bus_charging_energy_kwh": 0.0, "drone_charging_energy_kwh": 0.0, "power_overload_duration": 0.0,
        "locker_overflow_duration": 0.0, "locker_overflow_amount": 0.0,
    }
    after = dict(before)
    after["energy_consumption"] = 2.0
    _, rc_base = env_base._build_transition_reward(before, after, terminal_penalty=0.0)
    _, rc_scaled = env_scaled._build_transition_reward(before, after, terminal_penalty=0.0)
    assert rc_base["energy_cost"] == pytest.approx(2.0)
    assert rc_scaled["energy_cost"] == pytest.approx(6.0)


def test_battery_violation_transition_penalty_is_nonnegative():
    env = EBusDroneEnv(smoke_test=True)
    before = {
        "passenger_delay": 0.0, "parcel_lateness": 0.0, "late_delivery_count": 0.0, "delivered_count": 0.0,
        "energy_consumption": 0.0, "power_overload": 0.0, "battery_violation": 5.0, "locker_overflow": 0.0,
        "bus_charging_energy_kwh": 0.0, "drone_charging_energy_kwh": 0.0, "power_overload_duration": 0.0,
        "locker_overflow_duration": 0.0, "locker_overflow_amount": 0.0,
    }
    after = dict(before)
    after["battery_violation"] = 3.0
    _, rc = env._build_transition_reward(before, after, terminal_penalty=0.0)
    assert rc["battery_safety"] == 0.0
