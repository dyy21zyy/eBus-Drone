import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.smoke
def test_paper_cli_smoke_train_eval_benchmark(temp_config):
    """Regression smoke for the paper CLI flow using canonical method names."""
    root = Path(temp_config).parent
    output_dir = root / 'outputs_test'

    cmds = [
        [sys.executable, '-m', 'src.main', '--mode', 'generate', '--config', str(temp_config), '--instance', 'small', '--seed', '1', '--output-dir', str(output_dir), '--overwrite'],
        [sys.executable, '-m', 'src.main', '--mode', 'offline', '--config', str(temp_config), '--instance', 'small', '--seed', '1', '--output-dir', str(output_dir), '--overwrite'],
        [sys.executable, '-m', 'src.main', '--mode', 'train', '--config', str(temp_config), '--instance', 'small', '--seed', '1', '--method', 'am_dueling_ddqn_dr', '--output-dir', str(output_dir), '--smoke', '--overwrite'],
        [sys.executable, '-m', 'src.main', '--mode', 'eval', '--config', str(temp_config), '--instance', 'small', '--seed', '1', '--method', 'am_dueling_ddqn_dr', '--output-dir', str(output_dir)],
        [
            sys.executable,
            '-m',
            'src.main',
            '--mode',
            'benchmark',
            '--config',
            str(temp_config),
            '--instance',
            'small',
            '--seeds',
            '1',
            '--methods',
            'uniform',
            'dwell_greedy',
            'battery_threshold',
            'dqn_dr',
            'ddqn_dr',
            'am_ddqn_dr',
            'am_dueling_ddqn_dr',
            '--output-dir',
            str(output_dir),
            '--overwrite',
            '--train-if-missing',
        ],
    ]

    for cmd in cmds:
        subprocess.run(cmd, check=True)

    metrics_dir = output_dir / 'metrics'
    benchmark_dir = output_dir / 'results' / 'benchmark' / 'small'

    assert any(metrics_dir.glob('train_log_am_dueling_ddqn_dr_small_seed_1.csv'))
    assert any(metrics_dir.glob('eval_am_dueling_ddqn_dr_small_seed_1.csv'))
    summary_files = list(benchmark_dir.glob('summary*.csv'))
    assert summary_files, 'benchmark summary should not be empty'

    # Guard against legacy "proposed" naming in output artifacts.
    artifact_names = [p.name for p in output_dir.rglob('*') if p.is_file()]
    assert all('proposed' not in name.lower() for name in artifact_names)
