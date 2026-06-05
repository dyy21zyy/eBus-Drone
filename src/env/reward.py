from __future__ import annotations


def _required_float(mapping: dict, key: str) -> float:
    if key not in mapping:
        raise KeyError(f"Missing reward component: {key}")
    return float(mapping[key])


def _nonnegative_component(components: dict, key: str) -> float:
    value = _required_float(components, key)
    # These inputs are physical quantities (delay, lateness, energy, overload,
    # safety deficit, overflow).  Negative values can only come from signed
    # cumulative-delta bookkeeping or tiny floating-point noise, not from the
    # physical definition.
    return max(0.0, value)


def compute_reward(components: dict, alphas: dict) -> tuple[float, dict]:
    """Compute RL reward from non-negative physical cost components.

    ``total_cost`` is the physical weighted operating/penalty cost for this
    transition.  It is not inferred from reward; reward is derived from cost.
    Keeping both fields makes future reward shaping safe because benchmark
    exporters can continue to accumulate ``total_cost`` directly.
    """
    d_p = _nonnegative_component(components, "passenger_delay")
    d_l = _nonnegative_component(components, "parcel_lateness")
    d_e = _nonnegative_component(components, "energy_cost")
    d_pwr = _nonnegative_component(components, "power_overload")
    d_b = _nonnegative_component(components, "battery_safety")
    d_k = _nonnegative_component(components, "locker_overflow")

    physical_cost_components = {
        "passenger_delay_cost": float(alphas.get("alpha_1", 1.0)) * d_p,
        "parcel_lateness_cost": float(alphas.get("alpha_2", 1.0)) * d_l,
        "energy_cost_component": float(alphas.get("alpha_3", 1.0)) * d_e,
        "power_overload_cost": float(alphas.get("alpha_4", 1.0)) * d_pwr,
        "battery_safety_cost": float(alphas.get("alpha_5", 1.0)) * d_b,
        "locker_overflow_cost": float(alphas.get("alpha_6", 1.0)) * d_k,
    }
    total_cost = sum(physical_cost_components.values())
    reward = -float(total_cost)
    out = dict(components)
    out.update({
        "passenger_delay": d_p,
        "parcel_lateness": d_l,
        "energy_cost": d_e,
        "power_overload": d_pwr,
        "battery_safety": d_b,
        "locker_overflow": d_k,
        "step_physical_cost": float(total_cost),
        "total_cost": float(total_cost),
        "reward": reward,
    })
    out.update(physical_cost_components)
    return reward, out
