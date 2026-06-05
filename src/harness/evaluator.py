from __future__ import annotations

import csv
from pathlib import Path
from src.utils.metrics import REQUIRED_PAPER_METRICS, init_metrics, finalize_metrics, add_legacy_metric_aliases, LEGACY_METRIC_ALIASES
from src.harness.result_validation import validate_episode_result

def _required_component(rc: dict, key: str) -> float:
    if key not in rc:
        raise KeyError(f"Missing reward component in info['reward_components']: {key}")
    return float(rc[key])

def evaluate_policy(env, policy, episodes: int = 1, max_steps: int | None = None, *, allow_debug_truncation: bool = False):
    metrics = init_metrics()
    min_bat = float("inf")
    termination_reason = None
    terminated_by_env = False
    truncated_by_max_steps = False
    terminal_undelivered_count_total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        step_idx = 0
        while True:
            if max_steps is not None and step_idx >= int(max_steps):
                truncated_by_max_steps = True
                termination_reason = 'max_steps_truncated'
                break
            mask = env.get_action_mask()
            event = getattr(env, 'current_event', None)
            getv = (lambda k, d=0.0: event.get(k, d)) if isinstance(event, dict) else (lambda k, d=0.0: getattr(event, k, d) if event is not None else d)
            preview = getv('passenger_service_preview', {}) or {}
            t_p_est = float(preview.get('passenger_dwell_min', getv('passenger_dwell_min', 0.0)))
            unloading_volume_kg = float(getv('unloading_volume_kg', 0.0))
            unloading_time_sec_per_kg = float(getattr(env, 'instance', {}).get('parcel', {}).get('unloading_time_sec_per_kg', 0.0))
            t_f = unloading_volume_kg * unloading_time_sec_per_kg / 60.0
            trip_id = getv('trip_id', None)
            bus = None
            if trip_id is not None and hasattr(env, 'bus_states'):
                bus = env.bus_states.get(int(trip_id))
            if bus is None and hasattr(env, 'state'):
                bus = env.state.get('current_bus_state')
            bus_cfg = getattr(env, 'config', {}).get('bus', {})
            e_min = float(bus_cfg.get('safety_battery_kwh', 40.0))
            e_max = float(bus_cfg.get('battery_capacity_kwh', 160.0))
            info_ctx = {
                'E_current': float(bus.get('battery_kwh', env.state.get('battery', 0.0))) if isinstance(bus, dict) else float(env.state.get('battery', 0.0)),
                'E_min': e_min,
                'E_max': float(bus.get('battery_capacity_kwh', e_max)) if isinstance(bus, dict) else e_max,
                'T_P_est': t_p_est,
                'T_F': t_f,
                'action_mask': mask,
                'event': event,
            }
            action = policy.select_action(obs, mask, info_ctx)
            step_idx += 1
            if mask[action] == 0:
                metrics['infeasible_actions'] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            rc = info.get('reward_components', {})
            if not rc:
                metrics['total_reward'] += reward
                if terminated or truncated:
                    terminated_by_env = bool(terminated)
                    termination_reason = info.get('termination_reason') or ('truncated' if truncated else termination_reason)
                    break
                continue
            metrics['total_reward'] += reward
            metrics['total_cost'] += _required_component(rc, 'total_cost')
            metrics['onboard_passenger_delay_passenger_min'] += max(0.0, _required_component(rc, 'passenger_delay'))
            metrics['parcel_lateness_parcel_min'] += max(0.0, _required_component(rc, 'parcel_lateness'))
            metrics['terminal_undelivered_penalty'] += _required_component(rc, 'terminal_penalty')
            metrics['total_energy_consumption'] += _required_component(rc, 'total_energy_kwh')
            metrics['station_power_overload_amount'] += _required_component(rc, 'power_overload')
            metrics['station_power_overload_duration'] += _required_component(rc, 'power_overload_duration')
            metrics['locker_overflow_amount'] += _required_component(rc, 'locker_overflow_amount')
            metrics['locker_overflow_duration'] += _required_component(rc, 'locker_overflow_duration')
            metrics['late_delivery_count'] += float(info.get('late_delivery_count_delta', _required_component(rc, 'number_late_deliveries')))
            metrics['battery_safety_violation_count'] += int(_required_component(rc, 'battery_safety') > 0)
            dwell = info.get('dwell_components', {})
            extra_dwell = float(
                dwell.get(
                    'additional_dwell_min',
                    max(0.0, float(dwell.get('realized_dwell_min', 0.0)) - float(dwell.get('passenger_dwell_min', 0.0))),
                )
            )
            metrics['average_excess_dwell_time_min'] += max(0.0, extra_dwell)
            metrics['total_bus_operating_delay_min'] += max(0.0, float(info.get('bus_operating_delay_delta', extra_dwell)))
            metrics['steps'] += 1
            metrics['repaired_actions'] += int(info.get('action_repaired', False))
            metrics['invalid_action_count'] += float(info.get('invalid_action', False))
            metrics['action_repair_count'] += float(info.get('was_action_repaired', info.get('action_repaired', False)))
            metrics['requested_action_index_sum'] += float(info.get('requested_action', action))
            metrics['executed_action_index_sum'] += float(info.get('executed_action', action))
            metrics['action_gap_sum'] += abs(float(info.get('requested_action', action)) - float(info.get('executed_action', action)))
            metrics['requested_charging_duration_min_sum'] += max(0.0, float(info.get('requested_duration_min', info.get('executed_duration_min', 0.0))))
            executed_duration_min = max(0.0, float(info.get('executed_duration_min', 0.0)))
            metrics['executed_charging_duration_min_sum'] += executed_duration_min
            if executed_duration_min > 0.0:
                metrics['charging_event_count'] += 1.0
                metrics['positive_executed_charging_duration_min_sum'] += executed_duration_min
            else:
                metrics['zero_charging_event_count'] += 1.0
            min_bat = min(min_bat, float(env.state.get('battery', 0)))
            if terminated or truncated:
                terminated_by_env = bool(terminated)
                terminal_undelivered_count_total += float(info.get('undelivered_terminal_count', 0.0))
                termination_reason = info.get('termination_reason') or ('truncated' if truncated else termination_reason)
                break

    metrics['minimum_bus_battery'] = 0.0 if min_bat == float('inf') else min_bat
    end_time = float(getattr(env, 'state', {}).get('time', 0.0))
    metrics['undelivered_parcel_count'] = float(terminal_undelivered_count_total) if terminal_undelivered_count_total > 0.0 else (float(sum(1 for p in env.parcel_states.values() if p.get('delivery_completion_time_min') is None or float(p.get('delivery_completion_time_min')) > end_time + 1e-9)) if hasattr(env, 'parcel_states') else 0.0)
    delivered_parcels = [
        p for p in getattr(env, 'parcel_states', {}).values()
        if p.get('delivery_completion_time_min') is not None
        and float(p.get('delivery_completion_time_min')) <= end_time + 1e-9
    ]
    locker_holding_times = [max(0.0, float(p['locker_holding_time_min'])) for p in delivered_parcels if p.get('locker_holding_time_min') is not None]
    metrics['average_locker_holding_time_min'] = float(sum(locker_holding_times)) / float(len(locker_holding_times)) if locker_holding_times else 0.0
    if metrics['total_bus_operating_delay_min'] <= 0.0:
        metrics['total_bus_operating_delay_min'] = max(0.0, float(sum(b.get('accumulated_operating_delay_min', 0.0) for b in getattr(env, 'bus_states', {}).values())))
    stations = list(getattr(env, 'station_states', {}).values())
    occupied_time = sum(float(st.get('charger_occupied_time_min', 0.0)) for st in stations)
    charger_capacity_time = sum(float(max(0, int(st.get('charging_slots', len(st.get('charger_release_times_min', [])) or 0)))) * float(getattr(env, 'horizon', getattr(env, 'state', {}).get('horizon', 0.0))) for st in stations)
    metrics['charger_utilization'] = (occupied_time / charger_capacity_time) if charger_capacity_time > 0.0 else 0.0
    # Definition: stockout frequency counts dispatch-trigger opportunities where waiting feasible parcels and idle drones exist but no full battery is available.
    metrics['drone_battery_stockout_count'] = float(sum(float(st.get('dispatch_stockout_count', 0.0)) for st in stations))

    env_episode_metrics = env.get_episode_metrics() if hasattr(env, 'get_episode_metrics') else {}
    for k in ['total_cost','total_reward','onboard_passenger_delay_passenger_min','average_excess_dwell_time_min','total_bus_operating_delay_min','parcel_lateness_parcel_min','late_delivery_count','undelivered_parcel_count','terminal_undelivered_penalty','minimum_bus_battery','battery_safety_violation_count','total_energy_consumption','station_power_overload_amount','locker_overflow_amount','station_power_overload_duration','locker_overflow_duration','average_charging_duration_min','average_positive_charging_duration_min','total_charging_duration_min','mean_requested_charging_duration_min','mean_executed_charging_duration_min','mean_requested_action_index','mean_executed_action_index','charging_event_count','zero_charging_event_count','invalid_action_count']:
        if k in env_episode_metrics:
            metrics[k] = float(env_episode_metrics[k])
    out = finalize_metrics(metrics)
    out['max_steps'] = max_steps
    out['episode_end_time'] = end_time
    out['operating_horizon_min'] = float(getattr(env, 'horizon', getattr(env, 'state', {}).get('horizon', 0.0)))
    out['termination_reason'] = termination_reason or ('horizon_reached' if out['episode_end_time'] >= out['operating_horizon_min'] else 'unknown')
    out['truncated_by_max_steps'] = bool(truncated_by_max_steps)
    out['full_horizon_completed'] = bool(terminated_by_env and out['termination_reason'] == 'horizon_reached' and not truncated_by_max_steps)
    out['operating_horizon'] = out['operating_horizon_min']
    out['average_charging_duration_min'] = float(env_episode_metrics.get('average_charging_duration_min', out.get('average_charging_duration_min', 0.0)))
    out['average_positive_charging_duration_min'] = float(env_episode_metrics.get('average_positive_charging_duration_min', out.get('average_positive_charging_duration_min', 0.0)))
    out['total_charging_duration_min'] = float(env_episode_metrics.get('total_charging_duration_min', out.get('total_charging_duration_min', 0.0)))
    out['mean_requested_charging_duration_min'] = float(env_episode_metrics.get('mean_requested_charging_duration_min', out.get('mean_requested_charging_duration_min', 0.0)))
    out['mean_executed_charging_duration_min'] = float(env_episode_metrics.get('mean_executed_charging_duration_min', out.get('mean_executed_charging_duration_min', 0.0)))
    out['mean_requested_action_index'] = float(env_episode_metrics.get('mean_requested_action_index', out.get('mean_requested_action_index', 0.0)))
    out['mean_executed_action_index'] = float(env_episode_metrics.get('mean_executed_action_index', out.get('mean_executed_action_index', 0.0)))
    out['charging_event_count'] = float(env_episode_metrics.get('charging_event_count', out.get('charging_event_count', 0.0)))
    out['zero_charging_event_count'] = float(env_episode_metrics.get('zero_charging_event_count', out.get('zero_charging_event_count', 0.0)))
    out['valid_charging_opportunity_count'] = float(env_episode_metrics.get('valid_charging_opportunity_count', out.get('steps', 0.0)))
    out['episode_length_decisions'] = float(env_episode_metrics.get('episode_length_decisions', out.get('steps', 0.0)))
    out = add_legacy_metric_aliases(out)
    missing = [k for k in REQUIRED_PAPER_METRICS if k not in out]
    if missing:
        raise KeyError(f"Missing required paper metrics in evaluator output: {missing}")
    return validate_episode_result(out, allow_debug_truncation=allow_debug_truncation)

def save_eval_metrics(rows: list[dict], out_csv: str):
    p = Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Evaluation produced no rows.')
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys() if k not in LEGACY_METRIC_ALIASES))
    for metric in REQUIRED_PAPER_METRICS:
        if metric not in fieldnames:
            fieldnames.append(metric)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows([{k: v for k, v in r.items() if k in fieldnames} for r in rows])
