

from __future__ import annotations
import os, sys, json, math, logging, argparse
from pathlib import Path
from typing import List, Dict

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    os.environ.setdefault("JAVA_HOME", _conda_prefix)
    _jvm = Path(_conda_prefix) / "lib" / "jvm" / "lib" / "server" / "libjvm.so"
    if _jvm.exists():
        os.environ.setdefault("JVM_PATH", str(_jvm))

import torch
from training.trajectory import Trajectory, Turn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("case_study")


def load_config(path="configs/skillflow.yaml"):
    from run_training import load_config as _load_config
    return _load_config(path)


def pick_question(data: list, task_type: str, idx: int = 0) -> dict:

    candidates = [q for q in data if q.get("task_type") == task_type]
    if not candidates:
        raise ValueError(f"No questions of type {task_type}")
    return candidates[idx % len(candidates)]


def run_episodes(config: dict, question: dict, n_trajs: int) -> List[Trajectory]:

    from training.gflownet_trainer import GFlowNetTrainer, TASK_TYPE_TO_ID


    trainer = GFlowNetTrainer(config=config)
    trainer.setup(train_data=[], val_data=[])

    trajectories = []
    for i in range(n_trajs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Trajectory {i+1}/{n_trajs}")
        logger.info(f"{'='*60}")
        traj = trainer._run_episode(question)
        trajectories.append(traj)
        logger.info(f"  → {len(traj.turns)} turns, reward={traj.reward:.3f}, r_tilde={traj.r_tilde:.3f}")

    return trajectories


def compute_flow_analysis(trainer, trajectories: List[Trajectory]) -> dict:

    from training.flow_metrics import (
        fill_turn_flows,
        compute_skill_marginal_flows,
        compute_state_flows,
        compute_step_importance,
        compute_forward_trajectory_log_flow,
    )


    logger.info("\nComputing logprobs...")
    trainer._fill_turn_logprobs_no_grad(trajectories)


    logger.info("Computing Z_θ(q)...")
    log_z_tensor = trainer._compute_partition_function(trajectories)
    log_z_list = log_z_tensor.detach().tolist()


    for i, traj in enumerate(trajectories):
        fill_turn_flows(traj, log_z_list[i])


    all_skill_ids = trainer.workspace.get_all_ids() if trainer.workspace else []
    skill_flows = compute_skill_marginal_flows(trajectories, all_skill_ids) if all_skill_ids else {}

    return {
        "log_z_list": log_z_list,
        "skill_flows": skill_flows,
    }


def format_case_study(trajectories: List[Trajectory], flow_data: dict, question: dict) -> str:

    lines = []

    lines.append("=" * 80)
    lines.append("SKILLFLOW CASE STUDY")
    lines.append("=" * 80)
    lines.append("")


    lines.append(f"Task Type: {question.get('task_type', '?')}")
    lines.append(f"Question:  {question.get('question', '?')[:120]}")
    lines.append(f"Answer:    {question.get('answer', '?')[:80]}")
    lines.append("")


    for ti, traj in enumerate(trajectories):
        log_z = flow_data["log_z_list"][ti]
        lines.append(f"{'─'*70}")
        lines.append(f"Trajectory τ_{ti+1}: {len(traj.turns)} steps, "
                      f"R={traj.reward:.3f}, R̃={traj.r_tilde:.3f}, "
                      f"log Z_θ(q)={log_z:.3f}")
        lines.append(f"{'─'*70}")
        lines.append("")


        fwd_sum = sum(getattr(t, 'forward_logprob', 0.0) for t in traj.turns)
        bwd_sum = sum(getattr(t, 'backward_logprob', 0.0) for t in traj.turns)
        log_r = math.log(max(traj.r_tilde, 0.01))
        K_total = max(sum(getattr(t, 'action_token_count', 0) for t in traj.turns), 1)
        delta = log_z + fwd_sum - 1.0 * log_r - bwd_sum
        balance = delta / K_total

        lines.append(f"  TTB Balance: Δ = log Z + Σ log π_θ − β·log R̃ − Σ log P_φ")
        lines.append(f"             = {log_z:.2f} + ({fwd_sum:.2f}) − 1.0·({log_r:.2f}) − ({bwd_sum:.2f})")
        lines.append(f"             = {delta:.2f}  (K_total={K_total})")
        lines.append(f"             Δ/K = {balance:.4f}")
        lines.append("")


        lines.append(f"  {'Step':>4} {'Action':>25} {'K_t':>4} {'log π_θ':>8} {'log P_φ':>8} "
                      f"{'I(t)':>8} {'F(s_t)':>8} │ Observation (truncated)")
        lines.append(f"  {'─'*4} {'─'*25} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8} │ {'─'*30}")

        cum_log_flow = log_z  

        for si, turn in enumerate(traj.turns):
            action_type = getattr(turn, 'action_type', '?')
            if action_type == 'skill_invoke':
                action_label = f"[SKILL] {getattr(turn, 'skill_id', '?')[:18]}"
            else:
                action_label = action_type[:25]

            fwd_lp = getattr(turn, 'forward_logprob', 0.0)
            bwd_lp = getattr(turn, 'backward_logprob', 0.0)
            K_t = getattr(turn, 'action_token_count', 0)
            I_t = getattr(turn, 'step_importance', 0.0)
            state_flow = getattr(turn, 'state_flow', 0.0)


            obs = getattr(turn, 'observation', '') or ''
            obs_preview = obs[:30].replace('\n', ' ')


            if I_t > 1.5:
                it_marker = " ★★"  
            elif I_t < 0.3:
                it_marker = " ◆"   
            else:
                it_marker = ""

            lines.append(
                f"  {si:4d} {action_label:>25} {K_t:4d} {fwd_lp:8.2f} {bwd_lp:8.2f} "
                f"{I_t:8.3f}{it_marker:3s} {state_flow:8.2f} │ {obs_preview}"
            )

        lines.append("")


        lines.append("  I(t) 解读:")
        for si, turn in enumerate(traj.turns):
            I_t = getattr(turn, 'step_importance', 0.0)
            action_type = getattr(turn, 'action_type', '?')
            if action_type == 'skill_invoke':
                continue
            if I_t > 1.5:
                lines.append(f"    Step {si}: I(t)={I_t:.3f} ★★ CRITICAL — "
                              f"前向策略的探索性决策，后向策略（看了结果后）不确定")
            elif I_t < 0.3:
                lines.append(f"    Step {si}: I(t)={I_t:.3f} ◆ CONFIRMED — "
                              f"后向策略高度认同（看了执行结果后觉得这步很好）")
        lines.append("")


    if flow_data["skill_flows"]:
        lines.append(f"{'='*70}")
        lines.append("SKILL MARGINAL FLOW F̂(s) (论文 Eq.12)")
        lines.append(f"{'='*70}")
        for sid, flow in sorted(flow_data["skill_flows"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * max(1, int((flow + 20) / 2))
            lines.append(f"  {sid:40s}: log F̂(s)={flow:7.2f}  {bar}")
        lines.append("")


    lines.append(f"{'='*70}")
    lines.append("DAG COMPARISON (同问题多轨迹对比)")
    lines.append(f"{'='*70}")
    rewards = [(i, t.r_tilde) for i, t in enumerate(trajectories)]
    rewards.sort(key=lambda x: x[1], reverse=True)
    best_i, best_r = rewards[0]
    worst_i, worst_r = rewards[-1]
    lines.append(f"  Best:  τ_{best_i+1} R̃={best_r:.3f}")
    lines.append(f"  Worst: τ_{worst_i+1} R̃={worst_r:.3f}")
    lines.append(f"  Gap:   {best_r - worst_r:.3f}")
    lines.append("")

    if best_r - worst_r > 0.3:
        best_traj = trajectories[best_i]
        worst_traj = trajectories[worst_i]
        lines.append("  Success path (τ_best):")
        for si, turn in enumerate(best_traj.turns):
            at = getattr(turn, 'action_type', '?')
            if at == 'skill_invoke': continue
            I_t = getattr(turn, 'step_importance', 0.0)
            lines.append(f"    Step {si}: {at:20s} I(t)={I_t:.3f}")
        lines.append("")
        lines.append("  Failure path (τ_worst):")
        for si, turn in enumerate(worst_traj.turns):
            at = getattr(turn, 'action_type', '?')
            if at == 'skill_invoke': continue
            I_t = getattr(turn, 'step_importance', 0.0)
            lines.append(f"    Step {si}: {at:20s} I(t)={I_t:.3f}")
        lines.append("")


    lines.append(f"{'='*70}")
    lines.append("EVOLUTION SIGNAL (论文 §4.4)")
    lines.append(f"{'='*70}")

    all_rewards = [t.r_tilde for t in trajectories]
    acc = sum(1 for r in all_rewards if r > 0.5) / len(all_rewards)
    avg_balance = sum(
        abs(getattr(t, 'log_z', 0) +
            sum(getattr(turn, 'forward_logprob', 0) for turn in t.turns) -
            math.log(max(t.r_tilde, 0.01)) -
            sum(getattr(turn, 'backward_logprob', 0) for turn in t.turns))
        / max(sum(getattr(turn, 'action_token_count', 0) for turn in t.turns), 1)
        for t in trajectories
    ) / len(trajectories)

    lines.append(f"  Task accuracy:     {acc:.2f}")
    lines.append(f"  Avg |Δ/K|:         {avg_balance:.4f}")
    lines.append(f"  Reward gap:        {best_r - worst_r:.3f}")
    lines.append("")


    critical_steps = []
    for ti, traj in enumerate(trajectories):
        if traj.r_tilde < 0.5:
            continue
        for si, turn in enumerate(traj.turns):
            I_t = getattr(turn, 'step_importance', 0.0)
            if I_t > 1.5:
                critical_steps.append({
                    'traj': ti, 'step': si,
                    'action': getattr(turn, 'action_type', '?'),
                    'I_t': I_t,
                    'instruction': (getattr(turn, 'instruction', '') or '')[:60],
                })

    if critical_steps:
        lines.append("  Critical decision points (I(t) > 1.5, from successful trajectories):")
        for cs in critical_steps[:5]:
            lines.append(f"    τ_{cs['traj']+1} Step {cs['step']}: {cs['action']} "
                          f"I(t)={cs['I_t']:.3f} — {cs['instruction']}")
    else:
        lines.append("  No critical steps found (I(t) > 1.5)")

    lines.append("")


    lines.append(f"{'='*70}")
    lines.append("SKILL EVOLUTION INPUT (传给 Skill Creator 的信息)")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append("Skill Creator 收到以下 evidence 来决定 ADD/UPDATE/DELETE/SKIP:")
    lines.append("")
    lines.append(f"  1. Success trajectories ({sum(1 for t in trajectories if t.r_tilde > 0.5)}):")
    for ti, traj in enumerate(trajectories):
        if traj.r_tilde > 0.5:
            steps = [getattr(t, 'action_type', '?') for t in traj.turns if getattr(t, 'action_type', '') != 'skill_invoke']
            lines.append(f"     τ_{ti+1}: {' → '.join(steps)} (R̃={traj.r_tilde:.3f})")
    lines.append("")
    lines.append(f"  2. Failed trajectories ({sum(1 for t in trajectories if t.r_tilde <= 0.5)}):")
    for ti, traj in enumerate(trajectories):
        if traj.r_tilde <= 0.5:
            steps = [getattr(t, 'action_type', '?') for t in traj.turns if getattr(t, 'action_type', '') != 'skill_invoke']
            lines.append(f"     τ_{ti+1}: {' → '.join(steps)} (R̃={traj.r_tilde:.3f})")
    lines.append("")
    lines.append(f"  3. Critical steps (I(t) > 1.5): {len(critical_steps)} found")
    lines.append(f"  4. DAG reward gap: {best_r - worst_r:.3f}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SkillFlow Case Study")
    parser.add_argument("--task-type", type=str, default="multi_hop_qa")
    parser.add_argument("--n-trajs", type=int, default=4)
    parser.add_argument("--question-idx", type=int, default=0)
    parser.add_argument("--config", type=str, default="configs/skillflow.yaml")
    parser.add_argument("--output", type=str, default="case_study_output.txt")
    args = parser.parse_args()

    config = load_config(args.config)


    train_path = config.get("train_data", "data/train_v3.json")
    with open(train_path) as f:
        data = json.load(f)

    question = pick_question(data, args.task_type, args.question_idx)
    logger.info(f"Selected question: {question.get('question', '?')[:100]}")


    from training.gflownet_trainer import GFlowNetTrainer
    trainer = GFlowNetTrainer(config=config)
    trainer.setup(train_data=data[:100], val_data=[])


    logger.info(f"\nRunning {args.n_trajs} episodes on same question...")
    trajectories = []
    for i in range(args.n_trajs):
        logger.info(f"\n{'='*50} Trajectory {i+1}/{args.n_trajs} {'='*50}")
        traj = trainer._run_episode(question)
        trajectories.append(traj)
        n_real_turns = len([t for t in traj.turns if getattr(t, 'action_type', '') != 'skill_invoke'])
        logger.info(f"  → {n_real_turns} action steps, reward={traj.reward:.3f}")


    logger.info("\n" + "=" * 50 + " Flow Analysis " + "=" * 50)
    flow_data = compute_flow_analysis(trainer, trajectories)


    report = format_case_study(trajectories, flow_data, question)


    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"\nCase study saved to {output_path}")


    print("\n" + report)


if __name__ == "__main__":
    main()
