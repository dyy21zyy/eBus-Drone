import csv

import numpy as np

from src.env.ebus_drone_env import EBusDroneEnv
from src.harness.benchmark_runner import run_benchmark
from src.policies.base_policy import BasePolicy
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.utils.config import load_yaml
from src.data_generation.scenario_generator import generate_instance
from src.offline.assignment_data_builder import build_assignment_data
from src.offline.assignment_solver import solve_assignment


class AlwaysLargeActionPolicy(BasePolicy):
    def select_action(self, obs, mask, info=None):
        _ = obs, info
        return len(mask) - 1


def _build_env(seed=1):
    cfg = load_yaml('configs/default.yaml')
    inst_cfg = load_yaml('configs/instances/small.yaml')
    instance = generate_instance(cfg, inst_cfg, seed=seed)
    scenario = {"passenger": {"passenger_arrivals": {}}, "power": {"station_base_load_kw": {}}}
    assignment = solve_assignment(build_assignment_data(instance)).to_dict()
    return EBusDroneEnv(config=cfg, instance=instance, scenario=scenario, assignment=assignment)


def test_infeasible_action_repaired_to_zero_when_no_charger_available():
    env = _build_env(seed=10)
    obs, _ = env.reset(seed=10)
    _ = obs
    # Force no available charger at decision time so only action index 0 is feasible.
    st = env.station_states[int(env.current_decision_event.station_id)]
    st['charger_release_times_min'] = [env.state['time'] + 1000.0 for _ in st['charger_release_times_min']]
    mask = env.get_action_mask()
    assert mask[0] == 1 and int(np.sum(mask)) == 1
    _, _, _, _, info = env.step(len(mask) - 1)
    assert info['requested_action'] == len(mask) - 1
    assert info['executed_action'] == 0
    assert info['was_action_repaired'] is True
    assert info['action_repair_count'] >= 1


def test_masked_policy_never_selects_infeasible_action_under_known_mask():
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)
    agent = AMDDQNDRAgent(4, len(mask), {"epsilon_start": 1.0, "epsilon_end": 1.0})
    for _ in range(200):
        a = agent.select_action(obs, mask, training=True)
        assert mask[a] == 1


def test_summary_csv_contains_action_repair_statistics(tmp_path):
    out_csv = tmp_path / 'summary.csv'
    cfg = load_yaml('configs/default.yaml')
    cfg['paths']['outputs'] = str(tmp_path)

    def env_builder(seed):
        return _build_env(seed=seed)

    rows = run_benchmark(
        methods=['no_charging'],
        out_csv=str(out_csv),
        env_builder=env_builder,
        instance_name='small',
        test_seeds=[1],
        cfg=cfg,
        smoke_test=True,
        train_if_missing=False,
    )
    assert rows
    with out_csv.open('r', encoding='utf-8') as f:
        header = next(csv.reader(f))
    for key in [
        'invalid_action_count',
        'action_repair_count',
        'action_repair_rate',
        'mean_requested_action_index',
        'mean_executed_action_index',
        'mean_action_gap_after_repair',
    ]:
        assert key in header
