from __future__ import annotations


REQUIRED_PAPER_METRICS = [
    "total_cost", "total_reward",
    "onboard_passenger_delay_passenger_min", "average_excess_dwell_time_min",
    "total_bus_operating_delay_min", "parcel_lateness_parcel_min",
    "late_delivery_count", "undelivered_parcel_count",
    "average_locker_holding_time_min", "terminal_undelivered_penalty",
    "minimum_bus_battery", "battery_safety_violation_count",
    "total_energy_consumption", "station_power_overload_amount",
    "station_power_overload_duration", "locker_overflow_amount",
    "locker_overflow_duration", "charger_utilization",
    "drone_battery_stockout_count", "average_charging_duration_min",
    "average_positive_charging_duration_min", "total_charging_duration_min",
    "mean_requested_charging_duration_min", "mean_executed_charging_duration_min",
    "charging_event_count", "zero_charging_event_count",
    "mean_requested_action_index", "mean_executed_action_index",
]

LEGACY_METRIC_ALIASES = {
    "onboard_passenger_delay": "onboard_passenger_delay_passenger_min",
    "average_excess_dwell_time": "average_excess_dwell_time_min",
    "total_bus_operating_delay": "total_bus_operating_delay_min",
    "parcel_lateness": "parcel_lateness_parcel_min",
    "average_locker_holding_time": "average_locker_holding_time_min",
    "average_charging_duration": "average_charging_duration_min",
    "mean_requested_action": "mean_requested_action_index",
    "mean_executed_action": "mean_executed_action_index",
}

REQUIRED_VALIDATION_FIELDS = [
    "episode_end_time",
    "operating_horizon_min",
    "termination_reason",
    "full_horizon_completed",
    "truncated_by_max_steps",
]


def add_legacy_metric_aliases(m: dict) -> dict:
    for old, new in LEGACY_METRIC_ALIASES.items():
        if new in m:
            m[old] = m[new]
        elif old in m:
            m[new] = m[old]
    return m


def init_metrics():
    m = {k: 0.0 for k in REQUIRED_PAPER_METRICS}
    m.update({
        "steps": 0.0,
        "infeasible_actions": 0.0,
        "repaired_actions": 0.0,
        "requested_action_index_sum": 0.0,
        "executed_action_index_sum": 0.0,
        "action_gap_sum": 0.0,
        "requested_charging_duration_min_sum": 0.0,
        "executed_charging_duration_min_sum": 0.0,
        "positive_executed_charging_duration_min_sum": 0.0,
        "invalid_action_count": 0.0,
        "action_repair_count": 0.0,
    })
    return add_legacy_metric_aliases(m)


def finalize_metrics(m: dict):
    steps = max(1, int(m.get("steps", 0)))
    m["average_excess_dwell_time_min"] = float(m.get("average_excess_dwell_time_min", m.get("average_excess_dwell_time", 0.0))) / steps
    m["infeasible_action_rate"] = m.get("infeasible_actions", 0.0) / steps
    repairs = float(m.get("action_repair_count", m.get("repaired_actions", 0.0)))
    m["action_repair_rate"] = repairs / steps
    m["mean_requested_action_index"] = float(m.get("requested_action_index_sum", m.get("requested_action_sum", 0.0))) / steps
    m["mean_executed_action_index"] = float(m.get("executed_action_index_sum", m.get("executed_action_sum", 0.0))) / steps
    m["mean_action_gap_after_repair"] = float(m.get("action_gap_sum", 0.0)) / steps
    m["mean_requested_charging_duration_min"] = float(m.get("requested_charging_duration_min_sum", 0.0)) / steps
    m["mean_executed_charging_duration_min"] = float(m.get("executed_charging_duration_min_sum", 0.0)) / steps
    m["total_charging_duration_min"] = float(m.get("executed_charging_duration_min_sum", m.get("total_charging_duration_min", 0.0)))
    m["average_charging_duration_min"] = m["mean_executed_charging_duration_min"]
    positive_events = max(1.0, float(m.get("charging_event_count", 0.0)))
    m["average_positive_charging_duration_min"] = float(m.get("positive_executed_charging_duration_min_sum", 0.0)) / positive_events if m.get("charging_event_count", 0.0) else 0.0
    return add_legacy_metric_aliases(m)
