import pytest
from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.trainer import train_agent
from src.harness.benchmark_runner import build_policy, run_benchmark, normalize_method_name, uniform_seconds_from_method
from src.harness.sensitivity_runner import run_sensitivity
from src.harness.evaluator import save_eval_metrics, evaluate_policy
from src.policies.dwell_greedy_policy import DwellGreedyPolicy
import csv



class _NoopAgent:
    def __init__(self, *_args, **_kwargs):
        pass

    def load_checkpoint(self, _path):
        return None

    def select_action(self, _obs, _mask):
        return 0

class _LongEpisodeEnv:
    def __init__(self, done_after: int = 5):
        self.done_after = done_after
        self.horizon_sec = 600.0
        self.state = {"time": 0.0}
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        self.state = {"time": 0.0}
        return [0.0, 0.0], {}

    def get_action_mask(self):
        return [1, 1]

    def step(self, _action):
        self._steps += 1
        self.state["time"] = float(self._steps * 60.0)
        done = self._steps >= self.done_after
        reason = "horizon_reached" if done else None
        return [0.0, 0.0], 0.0, done, False, {"termination_reason": reason, "reward_components": {}}


def test_training_smoke_writes_checkpoint_and_log(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(env, method='proposed', episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}}, seed=1, instance_name='small')
    assert (tmp_path / 'checkpoints' / 'checkpoint_am_dueling_ddqn_dr_small_seed_1.pt').exists()
    assert (tmp_path / 'metrics' / 'train_log_am_dueling_ddqn_dr_small_seed_1.csv').exists()


def test_formal_training_ignores_max_steps_and_runs_to_done(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(
        env,
        method='proposed',
        episodes=1,
        max_steps=2,
        smoke_test=False,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_am_dueling_ddqn_dr_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert int(row['episode_steps']) > 2
    assert row['truncated_by_max_steps'].lower() == 'false'


def test_smoke_training_allows_max_step_truncation_and_labels_non_paper_ready(tmp_path):
    env = _LongEpisodeEnv(done_after=5)
    train_agent(
        env,
        method='proposed',
        episodes=1,
        max_steps=2,
        smoke_test=True,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 1, 'max_steps_per_episode': 2}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_am_dueling_ddqn_dr_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        row = next(csv.DictReader(f))
    assert row['truncated_by_max_steps'].lower() == 'true'
    assert row['termination_reason'] == 'max_steps_truncated'
    assert row['paper_ready_episode'].lower() == 'false'


def test_train_agent_cli_episodes_override_config(tmp_path):
    env = _LongEpisodeEnv(done_after=1)
    train_agent(
        env,
        method='proposed',
        episodes=3,
        max_steps=1,
        smoke_test=True,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 20, 'max_steps_per_episode': 1}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_am_dueling_ddqn_dr_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_train_agent_none_episodes_falls_back_to_config(tmp_path):
    env = _LongEpisodeEnv(done_after=1)
    train_agent(
        env,
        method='proposed',
        episodes=None,
        max_steps=1,
        smoke_test=True,
        out_root=str(tmp_path),
        cfg={'rl': {'episodes': 20, 'max_steps_per_episode': 1}},
        seed=1,
        instance_name='small',
    )
    with (tmp_path / 'metrics' / 'train_log_am_dueling_ddqn_dr_small_seed_1.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20


def test_missing_checkpoint_error(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    with pytest.raises(FileNotFoundError, match='Missing checkpoint'):
        build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=False, seed=1, instance_name='small')


def test_benchmark_generates_summary_csv(tmp_path):
    rows = run_benchmark(['no_charging'], str(tmp_path / 'benchmark.csv'), env_builder=lambda s: EBusDroneEnv(smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}}, smoke_test=True)
    assert rows and (tmp_path / 'benchmark.csv').exists() and (tmp_path / 'benchmark.json').exists()


def test_method_name_normalization():
    assert normalize_method_name('dwell_based_greedy') == 'dwell_greedy'
    assert normalize_method_name('proposed') == 'am_dueling_ddqn_dr'
    assert normalize_method_name('uniform') == 'uniform'
    assert normalize_method_name('uniform_45') == 'uniform'
    assert normalize_method_name('am_ddqn_dr') == 'am_ddqn_dr'
    assert normalize_method_name('am_dueling_ddqn_dr') == 'am_dueling_ddqn_dr'


def test_uniform_method_parsing_supports_paper_durations():
    assert uniform_seconds_from_method('uniform') == 45
    assert uniform_seconds_from_method('uniform_15') == 15
    assert uniform_seconds_from_method('uniform_30') == 30
    assert uniform_seconds_from_method('uniform_45') == 45
    assert uniform_seconds_from_method('uniform_60') == 60
    assert uniform_seconds_from_method('uniform_120') == 120


def test_build_policy_supports_all_learning_ablation_methods(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    for method in ["dqn_dr", "ddqn_dr", "am_ddqn_dr", "am_dueling_ddqn_dr"]:
        policy = build_policy(method, env, out_root=str(tmp_path), train_if_missing=True, smoke_test=True, seed=7, instance_name="small")
        assert policy is not None


def test_evaluation_full_horizon_default_not_50():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics['steps'] != 50


def test_max_steps_explicit_truncates():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=3)
    assert metrics['steps'] <= 3
    assert metrics['max_steps'] == 3
    assert metrics['full_horizon_completed'] is False


def test_full_horizon_only_true_when_operating_horizon_reached():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics['full_horizon_completed'] == (metrics['termination_reason'] == 'horizon_reached')


def test_episode_end_time_and_operating_horizon_are_minutes():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    assert metrics["operating_horizon_min"] == pytest.approx(float(env.horizon))
    assert metrics["episode_end_time"] <= metrics["operating_horizon_min"] + 1e-6


def test_completed_episode_end_time_near_delivery_horizon_minutes():
    env = EBusDroneEnv(smoke_test=True)
    metrics = evaluate_policy(env, DwellGreedyPolicy(), episodes=1, max_steps=None)
    if metrics["termination_reason"] == "horizon_reached":
        assert metrics["episode_end_time"] == pytest.approx(float(env.horizon), abs=1e-6)


def test_sensitivity_changes_factor(tmp_path):
    out = tmp_path / 'sens.csv'
    rows = run_sensitivity(['no_charging'], str(out), env_builder=lambda s, c: EBusDroneEnv(config=c, smoke_test=True), instance_name='small', test_seeds=[1], cfg={'paths': {'outputs': str(tmp_path)}, '_sensitivity_hooks': {'regenerate_instance': lambda *_args: None}}, factor='passenger_intensity', values=[0.75, 1.25], smoke_test=True)
    assert {r['sensitivity_value'] for r in rows} == {0.75, 1.25}
    assert all('whether_instance_regenerated' in r for r in rows)


def test_sensitivity_regen_and_offline_rules_and_scenario_consistency(tmp_path):
    calls = {'regen': 0, 'offline': 0, 'env': []}

    def regen_cb(cfg_mod, *_args):
        calls['regen'] += 1

    def offline_cb(_cfg_mod, _instance, _seed, factor, value):
        calls['offline'] += 1
        if factor == 'locker_capacity' and value == 0.0:
            return 'infeasible'
        return 'resolved'

    def env_builder(seed, cfg_mod):
        calls['env'].append((seed, cfg_mod.get('power', {}).get('station_capacity_kw')))
        return EBusDroneEnv(config=cfg_mod, smoke_test=True)

    cfg = {'paths': {'outputs': str(tmp_path)}, '_sensitivity_hooks': {'regenerate_instance': regen_cb, 'resolve_offline': offline_cb}}

    rows_p = run_sensitivity(['no_charging', 'max_feasible'], str(tmp_path / 'p.csv'), env_builder, 'small', [3], cfg, 'passenger_intensity', [0.9], smoke_test=True)
    assert calls['regen'] == 1 and calls['offline'] == 0
    assert all(r['whether_instance_regenerated'] and not r['whether_offline_resolved'] for r in rows_p)
    assert len({r['scenario_token'] for r in rows_p}) == 1

    rows_n = run_sensitivity(['no_charging'], str(tmp_path / 'n.csv'), env_builder, 'small', [3], cfg, 'num_customers', [1.2], smoke_test=True)
    assert calls['regen'] == 2 and calls['offline'] == 1
    assert rows_n[0]['whether_instance_regenerated'] and rows_n[0]['whether_offline_resolved']

    rows_d = run_sensitivity(['no_charging'], str(tmp_path / 'd.csv'), env_builder, 'small', [3], cfg, 'drones_per_station', [1.1], smoke_test=True)
    assert calls['regen'] == 3 and calls['offline'] == 2 and rows_d[0]['whether_offline_resolved']

    rows_pow = run_sensitivity(['no_charging'], str(tmp_path / 'pow.csv'), env_builder, 'small', [4], cfg, 'station_power_capacity', [1600.0], smoke_test=True)
    assert calls['regen'] == 4 and calls['offline'] == 2
    assert not rows_pow[0]['whether_offline_resolved'] and rows_pow[0]['whether_instance_regenerated']
    assert any(v == 1600.0 for _s, v in calls['env'])

    rows_inf = run_sensitivity(['no_charging'], str(tmp_path / 'inf.csv'), env_builder, 'small', [5], cfg, 'locker_capacity', [0.0], smoke_test=True)
    assert rows_inf[0]['offline_status'] == 'infeasible'
    assert rows_inf[0]['full_horizon_completed'] is False
    assert rows_inf[0]['termination_reason'] == 'offline_infeasible'
    assert 'whether_policy_retrained' in rows_inf[0]


def test_sensitivity_freight_trip_availability_forces_regen_and_resolve(tmp_path):
    calls = {'regen': 0, 'offline': 0}

    def regen_cb(*_args):
        calls['regen'] += 1

    def offline_cb(*_args):
        calls['offline'] += 1
        return 'resolved'

    cfg = {'paths': {'outputs': str(tmp_path)}, '_sensitivity_hooks': {'regenerate_instance': regen_cb, 'resolve_offline': offline_cb}}
    rows = run_sensitivity(['no_charging'], str(tmp_path / 'freight.csv'), lambda s, c: EBusDroneEnv(config=c, smoke_test=True), 'small', [1], cfg, 'freight_trip_availability', [4], smoke_test=True)
    assert calls['regen'] == 1 and calls['offline'] == 1
    assert rows[0]['whether_instance_regenerated'] is True
    assert rows[0]['whether_offline_resolved'] is True


def test_eval_metrics_saved_per_seed(tmp_path):
    rows=[{'method':'no_charging','instance':'small','seed':1,'total_reward':0.0},{'method':'no_charging','instance':'small','seed':2,'total_reward':0.0}]
    save_eval_metrics(rows, str(tmp_path/'eval.csv'))
    txt=(tmp_path/'eval.csv').read_text()
    assert 'seed' in txt and '2' in txt


def test_empty_rows_raise_error(tmp_path):
    with pytest.raises(ValueError):
        save_eval_metrics([], str(tmp_path/'empty.csv'))


def test_checkpoint_config_preserves_hidden_layers_for_eval(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    cfg = {'rl': {'hidden_layers': [16, 8], 'device': 'cpu', 'episodes': 1}}
    train_agent(env, method='am_dueling_ddqn_dr', episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg=cfg, seed=11, instance_name='small')
    policy = build_policy('am_dueling_ddqn_dr', env, out_root=str(tmp_path), train_if_missing=False, cfg=cfg, seed=11, instance_name='small')
    first_linear = policy.agent.online.feature[0]
    assert first_linear.out_features == 16


def test_checkpoint_architecture_mismatch_raises_clear_error(tmp_path):
    env = EBusDroneEnv(smoke_test=True)
    cfg = {'rl': {'hidden_layers': [16, 8], 'device': 'cpu', 'episodes': 1}}
    train_agent(env, method='am_dueling_ddqn_dr', episodes=1, max_steps=2, smoke_test=True, out_root=str(tmp_path), cfg=cfg, seed=13, instance_name='small')
    ckpt_cfg = tmp_path / 'checkpoints' / 'checkpoint_am_dueling_ddqn_dr_small_seed_13.agent_config.json'
    payload = __import__('json').loads(ckpt_cfg.read_text(encoding='utf-8'))
    payload['hidden_layers'] = [32, 32]
    ckpt_cfg.write_text(__import__('json').dumps(payload), encoding='utf-8')
    with pytest.raises(RuntimeError, match='architecture mismatch'):
        build_policy('am_dueling_ddqn_dr', env, out_root=str(tmp_path), train_if_missing=False, cfg=cfg, seed=13, instance_name='small')


class _CapturePolicy:
    def __init__(self):
        self.last_info = None

    def select_action(self, _obs, action_mask, info=None):
        self.last_info = info or {}
        infeasible = next((i for i, v in enumerate(action_mask) if v == 0), None)
        return int(infeasible if infeasible is not None else 0)


class _SingleStepInfoEnv:
    def __init__(self, *, unloading_volume_kg: float = 0.0, safety_battery_kwh: float = 40.0):
        self.state = {"time": 0.0}
        self.config = {"bus": {"safety_battery_kwh": safety_battery_kwh, "battery_capacity_kwh": 160.0}}
        self.instance = {"parcel": {"unloading_time_sec_per_kg": 3.0}}
        self.bus_states = {0: {"battery_kwh": 80.0, "battery_capacity_kwh": 160.0}}
        self.current_event = {"trip_id": 0, "unloading_volume_kg": unloading_volume_kg, "passenger_service_preview": {"passenger_dwell_min": 0.5}}

    def reset(self, seed=None):
        self.state = {"time": 0.0}
        return [0.0], {}

    def get_action_mask(self):
        return [1, 0, 1]

    def step(self, action):
        _ = action
        return [0.0], 0.0, True, False, {"termination_reason": "horizon_reached", "reward_components": {}}


def test_evaluator_passes_nonzero_freight_dwell_estimate_when_unloading_exists():
    env = _SingleStepInfoEnv(unloading_volume_kg=10.0)
    pol = _CapturePolicy()
    evaluate_policy(env, pol, episodes=1, max_steps=1, allow_debug_truncation=True)
    assert pol.last_info["T_F"] > 0.0


def test_evaluator_passes_configured_battery_safety_level():
    env = _SingleStepInfoEnv(safety_battery_kwh=47.0)
    pol = _CapturePolicy()
    evaluate_policy(env, pol, episodes=1, max_steps=1, allow_debug_truncation=True)
    assert pol.last_info["E_min"] == 47.0


def test_rule_policies_only_select_feasible_actions():
    env = EBusDroneEnv(smoke_test=True)
    mask = env.get_action_mask()
    obs, _ = env.reset(seed=1)
    pol_dwell = DwellGreedyPolicy()
    a_dwell = pol_dwell.select_action(obs, mask, {"T_P_est": 0.0, "T_F": 0.0})
    assert mask[a_dwell] == 1

    from src.policies.battery_threshold_policy import BatteryThresholdPolicy
    pol_bat = BatteryThresholdPolicy()
    a_bat = pol_bat.select_action(obs, mask, {"E_current": 0.0, "E_min": 40.0, "E_max": 160.0})
    assert mask[a_bat] == 1


def test_build_policy_train_if_missing_formal_uses_rl_episodes(tmp_path, monkeypatch):
    env = EBusDroneEnv(smoke_test=True)
    monkeypatch.setitem(__import__('src.harness.benchmark_runner', fromlist=['AGENT_MAP']).AGENT_MAP, "am_dueling_ddqn_dr", _NoopAgent)
    calls = {}

    def fake_train_agent(*args, **kwargs):
        calls.update(kwargs)
        ckpt = tmp_path / 'checkpoints' / 'checkpoint_am_dueling_ddqn_dr_small_seed_1.pt'
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text('x', encoding='utf-8')
        return None, str(ckpt)

    monkeypatch.setattr('src.harness.benchmark_runner.train_agent', fake_train_agent)
    cfg = {'paths': {'outputs': str(tmp_path)}, 'rl': {'episodes': 123}}
    build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=True, smoke_test=False, cfg=cfg, seed=1, instance_name='small')
    assert calls['episodes'] == 123
    assert calls['max_steps'] is None


def test_build_policy_train_if_missing_smoke_is_fast(tmp_path, monkeypatch):
    env = EBusDroneEnv(smoke_test=True)
    monkeypatch.setitem(__import__('src.harness.benchmark_runner', fromlist=['AGENT_MAP']).AGENT_MAP, "am_dueling_ddqn_dr", _NoopAgent)
    calls = {}

    def fake_train_agent(*args, **kwargs):
        calls.update(kwargs)
        ckpt = tmp_path / 'checkpoints' / 'checkpoint_am_dueling_ddqn_dr_small_seed_1.pt'
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text('x', encoding='utf-8')
        return None, str(ckpt)

    monkeypatch.setattr('src.harness.benchmark_runner.train_agent', fake_train_agent)
    cfg = {'paths': {'outputs': str(tmp_path)}, 'rl': {'episodes': 999}}
    build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=True, smoke_test=True, cfg=cfg, seed=1, instance_name='small')
    assert calls['episodes'] == 1
    assert calls['max_steps'] == 10


def test_build_policy_train_if_missing_episode_override(tmp_path, monkeypatch):
    env = EBusDroneEnv(smoke_test=True)
    monkeypatch.setitem(__import__('src.harness.benchmark_runner', fromlist=['AGENT_MAP']).AGENT_MAP, "am_dueling_ddqn_dr", _NoopAgent)
    calls = {}

    def fake_train_agent(*args, **kwargs):
        calls.update(kwargs)
        ckpt = tmp_path / 'checkpoints' / 'checkpoint_am_dueling_ddqn_dr_small_seed_1.pt'
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text('x', encoding='utf-8')
        return None, str(ckpt)

    monkeypatch.setattr('src.harness.benchmark_runner.train_agent', fake_train_agent)
    cfg = {'paths': {'outputs': str(tmp_path)}, 'rl': {'episodes': 123, 'train_if_missing_episodes': 77, 'train_if_missing_max_steps': 42}}
    build_policy('proposed', env, out_root=str(tmp_path), train_if_missing=True, smoke_test=False, cfg=cfg, seed=1, instance_name='small')
    assert calls['episodes'] == 77
    assert calls['max_steps'] == 42
