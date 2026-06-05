from __future__ import annotations
import csv, json, time
from copy import deepcopy
from pathlib import Path
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.evaluator import evaluate_policy
from src.harness.methods import normalize_method_name
from src.harness.trainer import train_agent
from src.harness.result_aggregator import aggregate
from src.harness.curve_export import export_eval_curves
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.policies import BatteryThresholdPolicy, DwellGreedyPolicy, LearnedPolicy, MaxFeasiblePolicy, NoChargingPolicy, UniformPolicy
from src.utils.metrics import REQUIRED_PAPER_METRICS, LEGACY_METRIC_ALIASES

AGENT_MAP={"dqn_dr":DQNDRAgent,"ddqn_dr":DDQNDRAgent,"am_ddqn_dr":AMDDQNDRAgent,"am_dueling_ddqn_dr":AMDuelingDDQNDRAgent}


def _checkpoint_agent_config_path(ckpt: Path) -> Path:
    return ckpt.with_suffix('.agent_config.json')


def _load_agent_config(ckpt: Path, cfg: dict | None) -> dict:
    cfg_path = _checkpoint_agent_config_path(ckpt)
    if cfg_path.exists():
        payload = json.loads(cfg_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid saved agent config format: {cfg_path}")
        return dict(payload)
    return dict((cfg or {}).get('rl', {}))


def _resolve_agent_config(method: str, obs_dim: int, action_dim: int, cfg: dict | None, ckpt: Path) -> dict:
    base = _load_agent_config(ckpt, cfg)
    if not base:
        base = {'device': 'auto'}
    resolved = deepcopy(base)
    resolved.setdefault('device', 'auto')
    resolved['method'] = method
    resolved['obs_dim'] = int(base.get('obs_dim', obs_dim))
    resolved['action_dim'] = int(base.get('action_dim', action_dim))
    if 'dueling' not in resolved:
        resolved['dueling'] = method in {'am_dueling_ddqn_dr'}
    if 'use_action_mask' not in resolved:
        resolved['use_action_mask'] = method in {'am_ddqn_dr', 'am_dueling_ddqn_dr'}
    return resolved


def _validate_architecture_or_raise(agent_cfg: dict, obs_dim: int, action_dim: int, method: str, ckpt: Path):
    mismatches = []
    if int(agent_cfg.get('obs_dim', obs_dim)) != int(obs_dim):
        mismatches.append(f"obs_dim saved={agent_cfg.get('obs_dim')} current={obs_dim}")
    if int(agent_cfg.get('action_dim', action_dim)) != int(action_dim):
        mismatches.append(f"action_dim saved={agent_cfg.get('action_dim')} current={action_dim}")
    expected_dueling = method in {'am_dueling_ddqn_dr'}
    if bool(agent_cfg.get('dueling', expected_dueling)) != expected_dueling:
        mismatches.append(f"dueling saved={agent_cfg.get('dueling')} expected_for_method={expected_dueling}")
    if mismatches:
        raise ValueError(f"Checkpoint architecture mismatch for {ckpt}: " + "; ".join(mismatches))


def uniform_seconds_from_method(method: str, cfg: dict | None = None) -> int:
    raw = str(method).strip().lower()
    if raw == 'uniform':
        return int((cfg or {}).get('uniform_duration_sec', 45))
    if raw.startswith('uniform_'):
        suffix = raw.removeprefix('uniform_')
        if suffix.isdigit() and int(suffix) > 0:
            return int(suffix)
    raise ValueError(f"Method is not a uniform charging policy: {method}")

def build_policy(method: str, env: EBusDroneEnv, out_root='outputs', checkpoint: str|None=None, train_if_missing: bool=False, smoke_test: bool=False, cfg:dict|None=None, seed:int=0, instance_name:str='unknown', log_interval: int | None = None):
    uniform_duration_sec = None
    if str(method).strip().lower().startswith('uniform'):
        uniform_duration_sec = uniform_seconds_from_method(method, cfg=cfg)
    method=normalize_method_name(method)
    if method == 'no_charging': return NoChargingPolicy()
    if method == 'uniform': return UniformPolicy(uniform_duration_sec if uniform_duration_sec is not None else uniform_seconds_from_method(method, cfg=cfg))
    if method == 'max_feasible': return MaxFeasiblePolicy()
    if method == 'dwell_greedy': return DwellGreedyPolicy()
    if method == 'battery_threshold': return BatteryThresholdPolicy()
    if method not in AGENT_MAP: raise ValueError(f'Unknown method: {method}')
    ckpt = Path(checkpoint) if checkpoint else Path(out_root)/'checkpoints'/f'checkpoint_{method}_{instance_name}_seed_{seed}.pt'
    if not ckpt.exists():
        if not train_if_missing:
            raise FileNotFoundError(f"Missing checkpoint for learning method '{method}': {ckpt}. Re-run with --train-if-missing.")
        rl_cfg = (cfg or {}).get('rl', {})
        if smoke_test:
            train_episodes = 1
            train_max_steps = 10
        else:
            train_episodes = rl_cfg.get('train_if_missing_episodes')
            if train_episodes is None:
                train_episodes = rl_cfg.get('episodes', 5000)
            train_max_steps = rl_cfg.get('train_if_missing_max_steps')
        _, path = train_agent(
            env,
            method=method,
            episodes=int(train_episodes),
            max_steps=10 if smoke_test else (int(train_max_steps) if train_max_steps is not None else None),
            smoke_test=smoke_test,
            out_root=out_root,
            cfg=cfg,
            seed=seed,
            instance_name=instance_name,
            log_interval=log_interval,
        )
        ckpt = Path(path)
    obs,_=env.reset(seed=seed)
    obs_dim = len(obs)
    action_dim = len(env.get_action_mask())
    agent_cfg = _resolve_agent_config(method, obs_dim, action_dim, cfg, ckpt)
    _validate_architecture_or_raise(agent_cfg, obs_dim, action_dim, method, ckpt)
    agent=AGENT_MAP[method](obs_dim, action_dim, agent_cfg)
    try:
        agent.load_checkpoint(str(ckpt))
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint due to network architecture mismatch for {ckpt}: {exc}") from exc
    return LearnedPolicy(agent)

def run_benchmark(methods, out_csv: str, env_builder, instance_name:str, test_seeds:list[int], cfg:dict, smoke_test: bool = False, train_if_missing:bool=False, log_interval: int | None = None):
    methods=[normalize_method_name(m) for m in methods]
    rows=[]
    eval_episodes = int(cfg.get('rl', {}).get('benchmark_eval_episodes', cfg.get('rl', {}).get('evaluation_episodes', 1)))
    for seed in test_seeds:
        for m in methods:
            env = env_builder(seed)
            t0=time.time()
            print(f"[benchmark] instance={instance_name} seed={seed} method={m} started", flush=True)
            pol = build_policy(m, env, out_root=cfg['paths']['outputs'], train_if_missing=train_if_missing, smoke_test=smoke_test, cfg=cfg, seed=seed, instance_name=instance_name, log_interval=log_interval)
            per_ep=[]
            for ep_i in range(eval_episodes):
                em = evaluate_policy(env, pol, episodes=1, max_steps=10 if smoke_test else None, allow_debug_truncation=bool(smoke_test))
                em.update({'eval_episode': ep_i+1, 'method':m,'instance':instance_name,'seed':seed,'runtime_sec':time.time()-t0,'smoke':bool(smoke_test),'smoke_mode':bool(smoke_test)})
                per_ep.append(em)
            export_eval_curves(cfg['paths']['outputs'], instance_name, m, int(seed), per_ep, smoke=bool(smoke_test))
            met=evaluate_policy(env, pol, episodes=eval_episodes, max_steps=10 if smoke_test else None, allow_debug_truncation=bool(smoke_test))
            met.update({'method':m,'instance':instance_name,'seed':seed,'runtime_sec':time.time()-t0,'smoke':bool(smoke_test),'smoke_mode':bool(smoke_test)})
            if m == 'uniform':
                met['uniform_duration_sec'] = uniform_seconds_from_method(m, cfg=cfg)
            rows.append(met)
            print(f"[benchmark] instance={instance_name} seed={seed} method={m} completed total_cost={met.get('total_cost', 'NA')} runtime_sec={met.get('runtime_sec', 'NA')}", flush=True)
    if not rows: raise ValueError('Benchmark produced no rows.')
    for m in methods:
        if not any(r['method']==m for r in rows): raise ValueError(f'No results for method: {m}')
    p=Path(out_csv); p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys() if k not in LEGACY_METRIC_ALIASES))
    for metric in REQUIRED_PAPER_METRICS:
        if metric not in fieldnames:
            fieldnames.append(metric)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows([{k: v for k, v in r.items() if k in fieldnames} for r in rows])
    agg={m:aggregate([r for r in rows if r['method']==m]) for m in methods}
    Path(out_csv).with_suffix('.json').write_text(json.dumps({'metadata':{'instance':instance_name,'seeds':test_seeds,'methods':methods},'aggregated':agg}, indent=2), encoding='utf-8')
    return rows
