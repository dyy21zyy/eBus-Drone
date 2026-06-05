from __future__ import annotations

from pathlib import Path
import copy
import pytest

from src.utils.config import load_yaml


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    cfg = load_yaml('configs/default.yaml')
    cfg = copy.deepcopy(cfg)
    cfg['paths']['data_generated'] = str(tmp_path / 'data' / 'generated')
    cfg['paths']['outputs'] = str(tmp_path / 'outputs')
    cfg.setdefault('rl', {})['train_if_missing_episodes'] = 1
    cfg.setdefault('rl', {})['benchmark_eval_episodes'] = 1
    cfg.setdefault('rl', {})['evaluation_episodes'] = 1
    p = tmp_path / 'test_config.yaml'
    try:
        import yaml  # type: ignore
        p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    except Exception:
        # PyYAML is a project dependency; fallback kept for robustness.
        lines = ['paths:', f"  data_generated: {cfg['paths']['data_generated']}", f"  outputs: {cfg['paths']['outputs']}"]
        p.write_text('\n'.join(lines), encoding='utf-8')
    return p


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.nodeid
        if 'test_main_experiments.py::test_pipeline_smoke_creates_nonempty_results' in name:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.smoke)
            continue
        if any(k in name for k in ('test_main_experiments.py', 'test_data_generation.py', 'test_offline_assignment.py')):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
        if 'smoke' in item.name:
            item.add_marker(pytest.mark.smoke)
