from __future__ import annotations
import csv, json, time
from copy import deepcopy
from pathlib import Path
from src.harness.benchmark_runner import build_policy
from src.harness.methods import normalize_method_name
from src.harness.evaluator import evaluate_policy
from src.harness.result_aggregator import aggregate

FACTOR_PATHS={
    'passenger_intensity':('passenger','demand_intensity_factor'),
    'base_load_perturbation_intensity':('power','disturbance_std_kw'),
    'num_customers':('generation','num_customers'),
    'customer_number':('generation','num_customers'),
    'parcel_intensity':('parcel','demand_intensity_factor'),
    'chargers_per_station':('charging','chargers_per_station'),
    'drones_per_station':('drone','drones_per_station'),
    'locker_capacity':('parcel','locker_capacity_kg'),
    'bus_freight_capacity':('bus','freight_capacity_kg'),
    'station_unloading_capacity':('parcel','unloading_capacity_kg_per_stop'),
    'station_power_capacity':('power','station_capacity_kw'),
    'charging_power':('charging','pantograph_power_kw'),
    'initial_full_batteries':('battery','initial_fully_charged_per_station'),
    'max_charging_duration':('charging','max_single_stop_seconds'),
    'freight_trip_availability':('generation','freight_trip_availability'),
}

ONLINE_SCENARIO_ONLY={'passenger_intensity','base_load_perturbation_intensity','chargers_per_station','station_power_capacity','charging_power'}
ONLINE_ENV_ONLY={'station_power_capacity','initial_full_batteries'}
REQUIRES_OFFLINE_RESOLVE={'num_customers','customer_number','parcel_intensity','locker_capacity','drones_per_station','bus_freight_capacity','station_unloading_capacity','integrated_station_set','freight_trip_availability'}
ENV_REBUILD_POSSIBLY_REEVAL={'chargers_per_station','max_charging_duration','action_set'}


def _factor_flags(factor:str)->dict[str,bool]:
    regen = factor in REQUIRES_OFFLINE_RESOLVE
    offline = factor in REQUIRES_OFFLINE_RESOLVE
    rebuild = factor in (ONLINE_ENV_ONLY | REQUIRES_OFFLINE_RESOLVE | ENV_REBUILD_POSSIBLY_REEVAL | ONLINE_SCENARIO_ONLY)
    return {'regenerate_instance':regen,'resolve_offline':offline,'rebuild_env':rebuild}


def run_sensitivity(methods, out_csv:str, env_builder, instance_name:str, test_seeds:list[int], cfg:dict, factor:str, values:list[float], smoke_test: bool=False, train_if_missing: bool=False):
    if factor not in FACTOR_PATHS:
        raise ValueError(f'Unsupported sensitivity factor: {factor}')
    methods = [normalize_method_name(m) for m in methods]
    rows=[]
    eval_episodes = int(cfg.get('rl', {}).get('benchmark_eval_episodes', cfg.get('rl', {}).get('evaluation_episodes', 1)))
    k1,k2=FACTOR_PATHS[factor]
    flags=_factor_flags(factor)
    hooks=cfg.get('_sensitivity_hooks',{})
    regen_cb=hooks.get('regenerate_instance')
    resolve_cb=hooks.get('resolve_offline')
    retrain_cb=hooks.get('retrain_policy')
    for v in values:
        cfg_mod=deepcopy(cfg); cfg_mod.setdefault(k1,{})[k2]=v
        print(f"[sensitivity] factor={factor} value={v} flags={flags}", flush=True)
        for seed in test_seeds:
            scenario_token={'seed':seed,'factor':factor,'value':v,'instance':instance_name}
            offline_status='not_required'
            term_reason='completed'
            regenerated=False
            offline_resolved=False
            if flags['regenerate_instance']:
                regenerated=True
                if regen_cb is not None:
                    regen_cb(cfg_mod, instance_name, seed, factor, v)
            if flags['resolve_offline']:
                offline_resolved=True
                if resolve_cb is not None:
                    offline_status=resolve_cb(cfg_mod, instance_name, seed, factor, v)
                else:
                    offline_status='resolved'
                if offline_status in {'infeasible','failed'}:
                    term_reason='offline_infeasible'
                    for m in methods:
                        rows.append({'sensitivity_name':factor,'sensitivity_value':v,'instance':instance_name,'instance_size':instance_name,'method':m,'seed':seed,'offline_status':offline_status,'whether_instance_regenerated':regenerated,'whether_offline_resolved':offline_resolved,'whether_policy_retrained':False,'policy_source':'not_run','full_horizon_completed':False,'termination_reason':term_reason,'runtime_sec':0.0,'scenario_token':json.dumps(scenario_token, sort_keys=True),'smoke_mode':bool(smoke_test)})
                    continue
            for m in methods:
                t0=time.time()
                env=env_builder(seed, cfg_mod)
                retrained = bool(retrain_cb(cfg_mod, instance_name, seed, factor, v, m)) if retrain_cb is not None else False
                pol=build_policy(m, env, out_root=cfg['paths']['outputs'], train_if_missing=train_if_missing, smoke_test=smoke_test, cfg=cfg_mod, seed=seed, instance_name=instance_name)
                met=evaluate_policy(env, pol, episodes=eval_episodes, max_steps=10 if smoke_test else None)
                met.update({'method':m,'instance':instance_name,'instance_size':instance_name,'seed':seed,'runtime_sec':time.time()-t0,'smoke_mode':bool(smoke_test),
                            'sensitivity_name':factor,'sensitivity_value':v,'whether_instance_regenerated':regenerated,'whether_offline_resolved':offline_resolved,
                            'offline_status':offline_status,'whether_policy_retrained':retrained,'policy_source':'retrained' if retrained else 'reused_default',
                            'full_horizon_completed':True,'termination_reason':term_reason,'scenario_token':json.dumps(scenario_token, sort_keys=True)})
                rows.append(met)
    if not rows: raise ValueError('Sensitivity produced no rows.')
    p=Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    grouped={}
    for m in methods:
        for v in values:
            sub=[r for r in rows if r['method']==m and r['sensitivity_value']==v and r.get('offline_status') not in {'infeasible','failed'}]
            if sub:
                grouped[f'{m}:{v}']=aggregate(sub)
    Path(out_csv).with_suffix('.json').write_text(json.dumps({'aggregated':grouped}, indent=2), encoding='utf-8')
    related = {
        "passenger_intensity": "onboard_passenger_delay_passenger_min",
        "station_power_capacity": "station_power_overload_amount",
        "chargers_per_station": "charger_utilization",
    }.get(factor)
    if related:
        vals = [float(r.get(related, 0.0)) for r in rows if r.get("offline_status") not in {"infeasible", "failed"}]
        if vals and len(set(round(v, 10) for v in vals)) == 1:
            print(f"[sensitivity][warning] factor={factor} produced identical related metric '{related}' across tested values.", flush=True)
    return rows
