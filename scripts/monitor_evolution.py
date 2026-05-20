#!/usr/bin/env python

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def compute_phase_stats(entries, phase_start, phase_end):

    phase = [e for e in entries if phase_start <= e["step"] <= phase_end]
    if not phase:
        return None

    stats = {
        "steps": f"{phase_start}-{phase_end}",
        "n_steps": len(phase),
        "avg_reward": sum(e["avg_reward"] for e in phase) / len(phase),
        "avg_answer": sum(e["avg_answer"] for e in phase) / len(phase),
        "avg_loss": sum(e["loss"] for e in phase) / len(phase),
        "avg_flow_entropy": sum(e.get("flow_entropy", 0) for e in phase) / len(phase),
        "avg_steps_per_ep": sum(e["avg_steps"] for e in phase) / len(phase),
        "workspace_size": phase[-1].get("workspace_size", "?"),
    }


    task_rewards = defaultdict(list)
    for e in phase:
        for task, reward in e.get("task_rewards", {}).items():
            task_rewards[task].append(reward)
    stats["task_rewards"] = {
        task: round(sum(rs) / len(rs), 4) for task, rs in sorted(task_rewards.items())
    }


    if len(phase) >= 4:
        mid = len(phase) // 2
        first_half = phase[:mid]
        second_half = phase[mid:]
        stats["trend_reward"] = round(
            sum(e["avg_reward"] for e in second_half) / len(second_half)
            - sum(e["avg_reward"] for e in first_half) / len(first_half),
            4,
        )
        stats["trend_answer"] = round(
            sum(e["avg_answer"] for e in second_half) / len(second_half)
            - sum(e["avg_answer"] for e in first_half) / len(first_half),
            4,
        )
    else:
        stats["trend_reward"] = 0
        stats["trend_answer"] = 0

    return stats


def extract_evolution_info(verbose_log: Path, step: int):

    info = {
        "triggered": False,
        "h_ratio": None,
        "avg_reward_at_evolve": None,
        "pruned": 0,
        "distilled": 0,
        "refined": 0,
        "new_skills": 0,
        "skills_after": None,
    }

    if not verbose_log or not verbose_log.exists():
        return info


    try:
        with open(verbose_log, "r") as f:
            lines = f.readlines()
    except Exception:
        return info


    for line in lines:
        if "[Evolution]" not in line:
            continue


        if "H_ratio=" in line and f"Step {step}" in line:
            info["triggered"] = True
            m = re.search(r"H_ratio=([\d.]+)", line)
            if m:
                info["h_ratio"] = float(m.group(1))
            m = re.search(r"avg_reward=([\d.]+)", line)
            if m:
                info["avg_reward_at_evolve"] = float(m.group(1))


        if f"Step {step} complete" in line:
            m = re.search(r"\+(\d+) new skills", line)
            if m:
                info["new_skills"] = int(m.group(1))
            m = re.search(r"-(\d+) pruned", line)
            if m:
                info["pruned"] = int(m.group(1))
            m = re.search(r"~(\d+) distilled", line)
            if m:
                info["distilled"] = int(m.group(1))
            m = re.search(r"~(\d+) refined", line)
            if m:
                info["refined"] = int(m.group(1))
            m = re.search(r"total=(\d+)", line)
            if m:
                info["skills_after"] = int(m.group(1))

    return info


def print_report(entries, verbose_log=None):

    if not entries:
        print("⚠ 无训练数据")
        return

    max_step = max(e["step"] for e in entries)
    evolution_phase_steps = 10  

    print("=" * 72)
    print(f"  SkillFlow Run 14 训练监控 — 当前 Step {max_step}")
    print("=" * 72)


    print(f"\n📊 全局概览 (Step 0 → {max_step}):")
    first = entries[0]
    last = entries[-1]
    print(f"  avg_reward:  {first['avg_reward']:.3f} → {last['avg_reward']:.3f}")
    print(f"  avg_answer:  {first['avg_answer']:.3f} → {last['avg_answer']:.3f}")
    print(f"  loss:        {first['loss']:.1f} → {last['loss']:.1f}")
    print(f"  H_flow:      {first.get('flow_entropy', 0):.4f} → {last.get('flow_entropy', 0):.4f}")
    print(f"  skills:      {first.get('workspace_size', '?')} → {last.get('workspace_size', '?')}")


    phases = []
    phase_start = 0
    while phase_start <= max_step:
        phase_end = min(phase_start + evolution_phase_steps - 1, max_step)
        stats = compute_phase_stats(entries, phase_start, phase_end)
        if stats:
            phases.append((phase_start, phase_end, stats))
        phase_start += evolution_phase_steps


    for i, (p_start, p_end, stats) in enumerate(phases):
        is_evolution_point = (p_end + 1) % evolution_phase_steps == 0 or p_end == max_step
        evol_step = p_end + 1 if is_evolution_point and p_end < max_step else None

        header = f"Phase {i} (Step {stats['steps']})"
        if evol_step and evol_step <= max_step:
            header += f" → 进化 @ Step {evol_step}"
        print(f"\n{'─' * 60}")
        print(f"  {header}")
        print(f"{'─' * 60}")

        print(f"  avg_reward:    {stats['avg_reward']:.4f}  (趋势: {stats['trend_reward']:+.4f})")
        print(f"  avg_answer:    {stats['avg_answer']:.4f}  (趋势: {stats['trend_answer']:+.4f})")
        print(f"  avg_loss:      {stats['avg_loss']:.1f}")
        print(f"  H_flow:        {stats['avg_flow_entropy']:.4f}")
        print(f"  avg_ep_steps:  {stats['avg_steps_per_ep']:.1f}")
        print(f"  workspace:     {stats['workspace_size']} skills")


        print(f"  任务明细:")
        for task, reward in stats["task_rewards"].items():
            bar = "█" * int(reward * 10) + "░" * (10 - int(reward * 10))
            print(f"    {task:20s} {reward:.4f} |{bar}|")


        if evol_step and evol_step <= max_step and verbose_log:
            evol_info = extract_evolution_info(verbose_log, evol_step)
            if evol_info["triggered"]:
                print(f"  🔄 进化详情 (Step {evol_step}):")
                if evol_info["h_ratio"] is not None:
                    print(f"    H_ratio:   {evol_info['h_ratio']:.3f} (< 0.85 触发)")
                if evol_info["avg_reward_at_evolve"] is not None:
                    print(f"    avg_reward: {evol_info['avg_reward_at_evolve']:.3f} (< 1.0 gate)")
                print(f"    +{evol_info['new_skills']} new, -{evol_info['pruned']} pruned, "
                      f"~{evol_info['distilled']} distilled, ~{evol_info['refined']} refined")
                if evol_info["skills_after"]:
                    print(f"    → skills: {evol_info['skills_after']}")


    evolutions_detected = []
    for step_idx in range(evolution_phase_steps, max_step + 1, evolution_phase_steps):
        if step_idx <= max_step:
            evolutions_detected.append(step_idx)

    if evolutions_detected:
        print(f"\n{'=' * 60}")
        print(f"  📈 进化前后对比")
        print(f"{'=' * 60}")

        for evol_step in evolutions_detected:

            pre_start = max(0, evol_step - 5)
            pre_end = evol_step - 1
            post_start = evol_step
            post_end = min(max_step, evol_step + 4)

            pre = compute_phase_stats(entries, pre_start, pre_end)
            post = compute_phase_stats(entries, post_start, post_end)

            if pre and post:
                print(f"\n  Evolution @ Step {evol_step}:")
                dr = post["avg_reward"] - pre["avg_reward"]
                da = post["avg_answer"] - pre["avg_answer"]
                dl = post["avg_loss"] - pre["avg_loss"]
                print(f"    reward: {pre['avg_reward']:.4f} → {post['avg_reward']:.4f}  ({dr:+.4f})")
                print(f"    answer: {pre['avg_answer']:.4f} → {post['avg_answer']:.4f}  ({da:+.4f})")
                print(f"    loss:   {pre['avg_loss']:.1f} → {post['avg_loss']:.1f}  ({dl:+.1f})")
                print(f"    skills: {pre['workspace_size']} → {post['workspace_size']}")


                all_tasks = set(list(pre["task_rewards"].keys()) + list(post["task_rewards"].keys()))
                for task in sorted(all_tasks):
                    r_pre = pre["task_rewards"].get(task, 0)
                    r_post = post["task_rewards"].get(task, 0)
                    delta = r_post - r_pre
                    arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
                    print(f"      {task:20s} {r_pre:.3f} → {r_post:.3f} {arrow} ({delta:+.3f})")


    print(f"\n{'─' * 60}")
    print(f"  最近 5 步逐步明细:")
    print(f"{'─' * 60}")
    recent = entries[-5:]
    print(f"  {'Step':>5s} {'Reward':>8s} {'Answer':>8s} {'Loss':>10s} {'H_flow':>8s} {'Steps':>6s} {'Skills':>6s}")
    for e in recent:
        print(
            f"  {e['step']:5d} {e['avg_reward']:8.4f} {e['avg_answer']:8.4f} "
            f"{e['loss']:10.1f} {e.get('flow_entropy', 0):8.4f} "
            f"{e['avg_steps']:6.1f} {e.get('workspace_size', '?'):>6}"
        )

    print(f"\n{'=' * 72}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        default="outputs/skillflow_general/training_log.jsonl",
    )
    parser.add_argument(
        "--verbose-log",
        default="logs/training_run14_fresh.log",
    )
    args = parser.parse_args()

    base = Path(os.environ.get("SKILLFLOW_REPO", Path(__file__).resolve().parents[1]))
    log_file = base / args.log_file
    verbose_log = base / args.verbose_log

    if not log_file.exists():
        print(f"日志文件不存在: {log_file}")
        sys.exit(1)

    entries = load_jsonl(log_file)
    print_report(entries, verbose_log if verbose_log.exists() else None)


if __name__ == "__main__":
    main()
