from __future__ import annotations

from pathlib import Path
import csv
import json
import time
import torch

from src.rl.agents.dqn_dr_agent import DQNDRAgent
from src.rl.agents.ddqn_dr_agent import DDQNDRAgent
from src.rl.agents.am_ddqn_dr_agent import AMDDQNDRAgent
from src.rl.agents.am_dueling_ddqn_dr_agent import AMDuelingDDQNDRAgent
from src.rl.sketch_buffer import SketchBuffer
from src.harness.methods import normalize_method_name
from src.harness.curve_export import export_training_curve

LEARNING_METHODS = {"dqn_dr", "ddqn_dr", "am_ddqn_dr", "am_dueling_ddqn_dr"}


def _buffer_total_added(buffer) -> int:
    if hasattr(buffer, "total_added"):
        return int(getattr(buffer, "total_added"))
    print("[train_agent][warning] ReplayBuffer missing total_added; falling back to len(buffer) for insertion accounting.", flush=True)
    return int(len(buffer))


def _agent_cls(method: str):
    method = normalize_method_name(method)
    return {
        "dqn_dr": DQNDRAgent,
        "ddqn_dr": DDQNDRAgent,
        "am_ddqn_dr": AMDDQNDRAgent,
        "am_dueling_ddqn_dr": AMDuelingDDQNDRAgent,
    }.get(method, AMDuelingDDQNDRAgent)


def train_agent(env, method: str = "am_dueling_ddqn_dr", episodes: int | None = 5, max_steps: int | None = 100, smoke_test: bool = False, out_root: str = "outputs", cfg: dict | None = None, seed: int | None = 0, instance_name: str = "unknown", resume: bool = False, checkpoint: str | None = None, log_interval: int | None = None):
    obs, _ = env.reset(seed=seed)
    cfg = dict(cfg or {})
    rl_cfg = dict(cfg.get("rl", {}))
    rl_cfg["method"] = normalize_method_name(method) if method is not None else normalize_method_name(rl_cfg.get("method", "am_dueling_ddqn_dr"))
    if "epsilon_decay_steps" not in rl_cfg:
        frac = float(rl_cfg.get("epsilon_decay_fraction", 0.8))
        ep_for_decay = int(episodes) if episodes is not None else int(rl_cfg.get("episodes", 5000))
        rl_cfg["epsilon_decay_steps"] = max(1, int(frac * ep_for_decay))
    rl_cfg.setdefault("batch_size", 8 if smoke_test else 32)
    rl_cfg.setdefault("warmup_steps", 4 if smoke_test else 20)
    rl_cfg.setdefault("target_update_interval", 10)
    rl_cfg.setdefault("replay_buffer_size", 2000)
    rl_cfg.setdefault("action_set_seconds", cfg.get("charging", {}).get("action_set_seconds", []))
    cfg_method = rl_cfg.get("method")
    if method is None:
        method = cfg_method or "am_dueling_ddqn_dr"
    cfg_seed = rl_cfg.get("seed")
    if seed is None:
        seed = int(cfg_seed) if cfg_seed is not None else 0
    method = normalize_method_name(method)
    cls = _agent_cls(method)
    agent = cls(len(obs), len(env.get_action_mask()), rl_cfg)
    print(f"[train_agent] selected device={agent.device} torch={torch.__version__} cuda_available={torch.cuda.is_available()} gpu_name={(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')}", flush=True)
    episodes = int(episodes) if episodes is not None else int(rl_cfg.get("episodes", 5))
    allow_train_truncation = bool(rl_cfg.get("allow_train_truncation", False))
    cfg_max_steps = rl_cfg.get("max_steps_per_episode", 100)
    resolved_max_steps = max_steps if max_steps is not None else cfg_max_steps
    if smoke_test or allow_train_truncation:
        max_steps = int(resolved_max_steps) if resolved_max_steps is not None else None
    else:
        max_steps = None
    effective_cfg = {
        "method": method,
        "instance": instance_name,
        "seed": int(seed),
        "episodes": episodes,
        "gamma": rl_cfg.get("gamma"),
        "learning_rate": rl_cfg.get("learning_rate"),
        "batch_size": rl_cfg.get("batch_size"),
        "replay_buffer_size": rl_cfg.get("replay_buffer_size"),
        "hidden_layers": rl_cfg.get("hidden_layers"),
        "device": rl_cfg.get("device", "cpu"),
        "max_steps": max_steps,
        "smoke_test": bool(smoke_test),
    }
    print(f"[train_agent] effective training config: {json.dumps(effective_cfg, sort_keys=True)}")
    rows = []
    progress_every = int(log_interval) if log_interval is not None else int(rl_cfg.get("progress_print_every", 10))
    ckpt_last_every = int(rl_cfg.get("checkpoint_last_every", 100))
    ckpt_periodic_every = int(rl_cfg.get("checkpoint_periodic_every", 1000))
    out = Path(out_root)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "raw_logs").mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "done").mkdir(parents=True, exist_ok=True)
    csv_path = out / "metrics" / f"train_log_{method}_{instance_name}_seed_{seed}.csv"
    fieldnames = ["episode", "method", "instance", "seed", "episode_reward", "episode_cost", "total_reward", "total_cost", "moving_avg_reward_10", "moving_avg_reward_50", "moving_avg_cost_10", "moving_avg_cost_50", "onboard_passenger_delay_passenger_min", "average_excess_dwell_time_min", "total_bus_operating_delay_min", "parcel_lateness_parcel_min", "late_delivery_count", "undelivered_parcel_count", "minimum_bus_battery", "battery_safety_violation_count", "total_energy_consumption", "station_power_overload_amount", "locker_overflow_amount", "average_charging_duration_min", "average_positive_charging_duration_min", "total_charging_duration_min", "mean_requested_charging_duration_min", "mean_executed_charging_duration_min", "mean_requested_action_index", "mean_executed_action_index", "epsilon", "loss", "loss_mean", "loss_last", "runtime_sec", "termination_reason", "steps", "episode_length_decisions", "episode_steps", "truncated_by_max_steps", "delayed_reward_sketch_count", "completed_transition_count", "incomplete_sketch_count", "terminal_transition_count", "replay_insertions_episode", "buffer_len", "buffer_total_added", "paper_ready_episode"]
    start_ep = 0
    best_metric = None
    if resume:
        ckpt_path = Path(checkpoint) if checkpoint else out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}_last.pt"
        if ckpt_path.exists():
            agent.load_checkpoint(str(ckpt_path))
            loaded = torch.load(str(ckpt_path), map_location=agent.device)
            start_ep = int(loaded.get("episode", 0))
            best_metric = loaded.get("best_metric")
            print(f"[train_agent] resumed from {ckpt_path} at episode={start_ep} global_step={agent.steps}", flush=True)
            print("[train_agent] replay buffer not restored; resuming with empty replay buffer.", flush=True)
    csv_exists = csv_path.exists()
    done_path = out / "done" / f"{method}_{instance_name}_seed_{seed}.done"
    t0 = time.time()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader(); csv_file.flush()
        for ep in range(start_ep, episodes):
            obs, _ = env.reset(seed=seed + ep)
            sketch = SketchBuffer()
            ep_reward = 0.0
            ep_cost = 0.0
            losses = []
            q_values = []
            action_sum = 0.0
            bat_dep = 0
            dec = 0
            step_idx = 0
            delayed_reward_sketch_count = 0
            completed_transition_count = 0
            terminal_transition_count = 0
            replay_insertions_before = _buffer_total_added(agent.buffer)
            termination_reason = None
            truncated_by_max_steps = False
            while True:
                if max_steps is not None and step_idx >= max_steps:
                    truncated_by_max_steps = True
                    termination_reason = "max_steps_truncated"
                    break
                mask = env.get_action_mask(); action = agent.select_action(obs, mask, training=True)
    
                sketch.start(
                    observation=obs,
                    action_index=action,
                    action_mask=mask,
                    info={"decision_step": step_idx, "decision_event": getattr(env, "state", {}).get("time")},
                )
                delayed_reward_sketch_count += 1
    
                nxt, reward, term, trunc, info = env.step(action)
                nxt_mask = env.get_action_mask() if not (term or trunc) else mask
                completed = sketch.finalize(reward, nxt, term or trunc, nxt_mask, info)
                if completed is not None:
                    agent.observe(
                        completed["observation"],
                        completed["action_index"],
                        completed["reward"],
                        completed["next_observation"],
                        completed["done"],
                        completed["action_mask"],
                        completed["next_action_mask"],
                        completed["info"],
                    )
                    completed_transition_count += 1
                    if term or trunc:
                        terminal_transition_count += 1
                loss = agent.update(); obs = nxt; ep_reward += reward; dec += 1
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    try:
                        q_t = agent.online(obs_t)
                    except TypeError:
                        q_t = agent.online(obs_t, torch.tensor(mask, dtype=torch.float32, device=agent.device).unsqueeze(0))
                    q_values.append(float(q_t.max().item()))
                action_sum += float(action)
                step_idx += 1
                if loss is not None: losses.append(float(loss))
                rc = info.get("reward_components", {})
                ep_cost += rc.get("total_cost", -reward)
                bat_dep += int(rc.get("battery_safety", 0) > 0)
                if term or trunc:
                    termination_reason = info.get("termination_reason") or ("truncated" if trunc else None)
                    break
            incomplete_sketch_count = 1 if sketch.has_pending() else 0
            replay_insertions_after = _buffer_total_added(agent.buffer)
            replay_insertions_episode = replay_insertions_after - replay_insertions_before
            buffer_len = len(agent.buffer)
            buffer_capacity = int(getattr(agent.buffer, "capacity", buffer_len))
            buffer_total_added = replay_insertions_after
            if not truncated_by_max_steps:
                assert incomplete_sketch_count == 0, "incomplete_sketch_count must be zero at formal episode end"
            if completed_transition_count != replay_insertions_episode:
                raise RuntimeError(
                    "completed transitions must match replay insertions | "
                    f"method={method} instance={instance_name} seed={seed} episode={ep + 1} "
                    f"completed_transition_count={completed_transition_count} replay_insertions_episode={replay_insertions_episode} "
                    f"buffer_len={buffer_len} buffer_capacity={buffer_capacity} buffer_total_added={buffer_total_added} "
                    f"incomplete_sketch_count={incomplete_sketch_count} terminal_transition_count={terminal_transition_count} "
                    f"termination_reason={termination_reason} truncated_by_max_steps={truncated_by_max_steps}"
                )
            if termination_reason != "max_steps_truncated" and dec > 0:
                assert terminal_transition_count == 1, "terminal transition must be stored exactly once"
            operating_horizon = float(getattr(env, "horizon", getattr(env, "delivery_evaluation_horizon", getattr(env, "state", {}).get("horizon", 0.0))))
            episode_end_time = float(getattr(env, "state", {}).get("time", 0.0))
            ep_runtime = float(time.time() - t0)
            episode_metrics = env.get_episode_metrics() if hasattr(env, "get_episode_metrics") else {}
            loss_mean = (sum(losses)/len(losses)) if losses else float("nan")
            last10 = rows[-9:] + [{"episode_reward": ep_reward, "episode_cost": float(episode_metrics.get("total_cost", ep_cost))}]
            last50 = rows[-49:] + [{"episode_reward": ep_reward, "episode_cost": float(episode_metrics.get("total_cost", ep_cost))}]
            row = {
            "episode": ep + 1,
            "method": method,
            "instance": instance_name,
            "seed": int(seed),
            "episode_reward": ep_reward,
            "episode_cost": float(episode_metrics.get("total_cost", ep_cost)),
            "total_reward": ep_reward,
            "total_cost": float(episode_metrics.get("total_cost", ep_cost)),
            "moving_avg_reward_10": sum(float(r.get("episode_reward", float("nan"))) for r in last10)/len(last10),
            "moving_avg_reward_50": sum(float(r.get("episode_reward", float("nan"))) for r in last50)/len(last50),
            "moving_avg_cost_10": sum(float(r.get("episode_cost", float("nan"))) for r in last10)/len(last10),
            "moving_avg_cost_50": sum(float(r.get("episode_cost", float("nan"))) for r in last50)/len(last50),
            "onboard_passenger_delay_passenger_min": float(episode_metrics.get("onboard_passenger_delay_passenger_min", episode_metrics.get("onboard_passenger_delay", 0.0))),
            "average_excess_dwell_time_min": float(episode_metrics.get("average_excess_dwell_time_min", episode_metrics.get("average_excess_dwell_time", 0.0))),
            "total_bus_operating_delay_min": float(episode_metrics.get("total_bus_operating_delay_min", episode_metrics.get("total_bus_operating_delay", 0.0))),
            "parcel_lateness_parcel_min": float(episode_metrics.get("parcel_lateness_parcel_min", episode_metrics.get("parcel_lateness", 0.0))),
            "late_delivery_count": float(episode_metrics.get("late_delivery_count", 0.0)),
            "undelivered_parcel_count": float(episode_metrics.get("undelivered_parcel_count", 0.0)),
            "minimum_bus_battery": float(episode_metrics.get("minimum_bus_battery", env.state.get("battery", 0.0))) if hasattr(env, "state") else 0.0,
            "battery_safety_violation_count": float(episode_metrics.get("battery_safety_violation_count", bat_dep)),
            "total_energy_consumption": float(episode_metrics.get("total_energy_consumption", 0.0)),
            "station_power_overload_amount": float(episode_metrics.get("station_power_overload_amount", 0.0)),
            "locker_overflow_amount": float(episode_metrics.get("locker_overflow_amount", 0.0)),
            "average_charging_duration_min": float(episode_metrics.get("average_charging_duration_min", 0.0)),
            "average_positive_charging_duration_min": float(episode_metrics.get("average_positive_charging_duration_min", 0.0)),
            "total_charging_duration_min": float(episode_metrics.get("total_charging_duration_min", 0.0)),
            "mean_requested_charging_duration_min": float(episode_metrics.get("mean_requested_charging_duration_min", 0.0)),
            "mean_executed_charging_duration_min": float(episode_metrics.get("mean_executed_charging_duration_min", 0.0)),
            "mean_requested_action_index": float(episode_metrics.get("mean_requested_action_index", (action_sum / dec) if dec else 0.0)),
            "mean_executed_action_index": float(episode_metrics.get("mean_executed_action_index", (action_sum / dec) if dec else 0.0)),
            "epsilon": float(agent._eps()),
            "loss": loss_mean,
            "loss_mean": loss_mean,
            "loss_last": (float(losses[-1]) if losses else float("nan")),
            "runtime_sec": ep_runtime,
            "termination_reason": termination_reason or ("horizon_reached" if episode_end_time >= operating_horizon else "unknown"),
            "steps": dec,
            "episode_length_decisions": dec,
            "episode_steps": dec,
            "truncated_by_max_steps": bool(truncated_by_max_steps),
            "delayed_reward_sketch_count": delayed_reward_sketch_count,
            "completed_transition_count": completed_transition_count,
            "incomplete_sketch_count": incomplete_sketch_count,
            "terminal_transition_count": terminal_transition_count,
            "replay_insertions_episode": replay_insertions_episode,
            "buffer_len": buffer_len,
            "buffer_total_added": buffer_total_added,
            "paper_ready_episode": bool(not truncated_by_max_steps and (termination_reason or "") == "horizon_reached"),
        }
            rows.append(row)
            writer.writerow(row); csv_file.flush()
            should_log = progress_every > 0 and (((ep + 1) % progress_every == 0) or (ep + 1 == episodes))
            if should_log:
                def _fmt(v, p=3):
                    if v is None or (isinstance(v, float) and (v != v)):
                        return "NA"
                    try:
                        return f"{float(v):.{p}f}"
                    except Exception:
                        return "NA"
                msg = (
                    f"[train][episode] method={method} instance={instance_name} seed={seed} ep={ep+1}/{episodes} "
                    f"reward={_fmt(row.get('episode_reward'), 2)} cost={_fmt(row.get('episode_cost'), 2)} "
                    f"ma10={_fmt(row.get('moving_avg_reward_10'), 2)} ma50={_fmt(row.get('moving_avg_reward_50'), 2)} "
                    f"epsilon={_fmt(row.get('epsilon'), 3)} loss={_fmt(row.get('loss_mean'), 3)} "
                    f"steps={int(row.get('steps', 0)) if str(row.get('steps', '')).strip() else 'NA'} "
                    f"min_battery={_fmt(row.get('minimum_bus_battery'), 2)} "
                    f"battery_viol={_fmt(row.get('battery_safety_violation_count'), 0)} "
                    f"term={row.get('termination_reason') or 'NA'} "
                    f"replay_insertions={int(row.get('replay_insertions_episode', 0))} "
                    f"completed_transitions={int(row.get('completed_transition_count', 0))} "
                    f"buffer_len={int(row.get('buffer_len', 0))} "
                    f"buffer_total_added={int(row.get('buffer_total_added', 0))} "
                    f"elapsed={_fmt(row.get('runtime_sec'), 1)}s"
                )
                print(msg, flush=True)
            if progress_every > 0 and (((ep + 1) % progress_every == 0) or (ep + 1 == episodes)):
                export_training_curve(str(out), instance_name, method, int(seed), rows, smoke=bool(smoke_test))
            ckpt_payload = agent.checkpoint_dict() | {"episode": ep + 1, "global_step": agent.steps, "epsilon": agent._eps(), "training_config": effective_cfg, "best_metric": best_metric, "buffer_total_added": buffer_total_added}
            if (ep + 1) % ckpt_last_every == 0:
                torch.save(ckpt_payload, out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}_last.pt")
            if (ep + 1) % ckpt_periodic_every == 0:
                torch.save(ckpt_payload, out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}_ep{ep+1}.pt")
            if best_metric is None or ep_cost < best_metric:
                best_metric = ep_cost
                ckpt_payload["best_metric"] = best_metric
                torch.save(ckpt_payload, out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}_best.pt")
    ckpt = out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}.pt"
    agent.save_checkpoint(str(ckpt))
    best_ckpt = out / "checkpoints" / f"checkpoint_{method}_{instance_name}_seed_{seed}_best.pt"
    agent.save_checkpoint(str(best_ckpt))
    agent_effective_config = dict(rl_cfg)
    agent_effective_config.update({
        "method": method,
        "obs_dim": int(len(obs)),
        "action_dim": int(len(env.get_action_mask())),
        "hidden_layers": rl_cfg.get("hidden_layers", [128, 128]),
        "learning_rate": rl_cfg.get("learning_rate", rl_cfg.get("lr", 1e-3)),
        "gamma": rl_cfg.get("gamma", 0.99),
        "dueling": method in {"am_dueling_ddqn_dr"},
        "use_action_mask": method in {"am_ddqn_dr", "am_dueling_ddqn_dr"},
        "device": rl_cfg.get("device", "auto"),
    })
    cfg_path = ckpt.with_suffix('.agent_config.json')
    cfg_path.write_text(json.dumps(agent_effective_config, indent=2), encoding='utf-8')
    (out / "checkpoints" / f"{method}.pt").write_bytes(ckpt.read_bytes())
    (out / "checkpoints" / f"{method}.agent_config.json").write_bytes(cfg_path.read_bytes())
    done_path.write_text(json.dumps({"method": method, "instance": instance_name, "seed": seed, "episodes": episodes, "start_time": t0, "end_time": time.time(), "runtime_sec": time.time() - t0, "final_checkpoint": str(ckpt), "status": "success"}, indent=2), encoding="utf-8")
    with (out / "raw_logs" / f"training_curve_{method}_{instance_name}_seed_{seed}.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "method": method, "instance": instance_name, "runtime_sec": time.time()-t0, "effective_training_config": effective_cfg, "rows": rows}, f, indent=2)
    with (out / "configs" / f"train_{method}_{instance_name}_seed_{seed}.json").open("w", encoding="utf-8") as f:
        json.dump({"seed": seed, "method": method, "instance": instance_name, "config": cfg, "effective_training_config": effective_cfg}, f, indent=2)
    export_training_curve(str(out), instance_name, method, int(seed), rows, smoke=bool(smoke_test))
    return agent, str(ckpt)
