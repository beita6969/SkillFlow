

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from training.trajectory import Trajectory


def edge_logprob_tilde(turn, which: str = "forward") -> float:

    attr = "forward_logprob" if which == "forward" else "backward_logprob"
    raw = float(getattr(turn, attr, 0.0) or 0.0)
    k = int(getattr(turn, "action_token_count", 0) or 0)
    if k <= 0:
        return 0.0
    return raw / max(k, 1)


def edge_log_i(turn) -> float:

    return edge_logprob_tilde(turn, "forward") - edge_logprob_tilde(turn, "backward")


def effective_paper_steps(traj: Trajectory) -> int:

    return max(
        sum(1 for t in traj.turns if int(getattr(t, "action_token_count", 0) or 0) > 0),
        1,
    )


def compute_state_flows(
    traj: Trajectory,
    log_z: float,
) -> List[float]:

    log_flow = log_z
    state_flows: List[float] = [log_flow]  

    for turn in traj.turns:
        log_flow = log_flow + edge_log_i(turn)
        state_flows.append(log_flow)

    return state_flows  


def compute_edge_flows(
    state_flows: List[float],
    forward_logprobs: List[float],
) -> List[float]:

    edge_flows = []
    for t, fwd in enumerate(forward_logprobs):
        edge_flows.append(state_flows[t] + fwd)
    return edge_flows


def compute_step_importance(
    forward_logprob: float,
    backward_logprob: float,
    action_token_count: int = 0,
) -> float:

    if action_token_count and action_token_count > 0:
        diff = (forward_logprob - backward_logprob) / max(action_token_count, 1)
    else:
        diff = forward_logprob - backward_logprob

    if diff > 700:
        return math.exp(700)
    if diff < -700:
        return 0.0
    return math.exp(diff)


def fill_turn_flows(traj: Trajectory, log_z: float) -> None:

    state_flows = compute_state_flows(traj, log_z)
    fwd_logprobs = [edge_logprob_tilde(t, "forward") for t in traj.turns]
    edge_flows = compute_edge_flows(state_flows, fwd_logprobs)

    for i, turn in enumerate(traj.turns):
        turn.state_flow = state_flows[i + 1]  
        turn.edge_flow = edge_flows[i]
        turn.step_importance = compute_step_importance(
            turn.forward_logprob, turn.backward_logprob, turn.action_token_count
        )

    traj.log_z = log_z


def compute_trajectory_flow(traj: Trajectory) -> float:

    if not traj.turns:
        return math.exp(traj.log_z)


    final_state_flow = traj.turns[-1].state_flow if traj.turns[-1].state_flow != 0.0 else traj.log_z
    return math.exp(final_state_flow)


def compute_forward_trajectory_log_flow(traj: Trajectory) -> float:

    log_flow = traj.log_z
    for turn in traj.turns:
        log_flow += edge_logprob_tilde(turn, "forward")
    return log_flow


def compute_forward_trajectory_flow(traj: Trajectory) -> float:

    return math.exp(compute_forward_trajectory_log_flow(traj))


def _log_sum_exp(log_values: List[float]) -> float:

    if not log_values:
        return -math.inf
    max_val = max(log_values)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def _compute_group_flow_entropy(log_flows: List[float]) -> float:

    if len(log_flows) < 2:
        return 0.0

    max_lf = max(log_flows)
    if max_lf == -math.inf:
        return 0.0

    shifted = [lf - max_lf for lf in log_flows]
    exp_shifted = [math.exp(s) if s > -500 else 0.0 for s in shifted]
    total = sum(exp_shifted)

    if total <= 0:
        return 0.0

    probs = [e / total for e in exp_shifted]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy


def compute_flow_entropy(trajectories: List[Trajectory]) -> float:

    if not trajectories:
        return 0.0

    h_max_batch = math.log(len(trajectories)) if len(trajectories) > 1 else 1.0


    norm_log_flows = [_per_token_normalized_log_flow(t) for t in trajectories]
    h = _compute_group_flow_entropy(norm_log_flows)

    return h


def compute_skill_marginal_flows(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:


    skill_traj_log_flows: Dict[str, Dict[str, List[float]]] = {sid: {} for sid in all_skill_ids}

    for traj in trajectories:
        for turn in traj.turns:
            if turn.action_type == "skill_invoke" and turn.skill_id in skill_traj_log_flows:
                sid = turn.skill_id
                if traj.traj_id not in skill_traj_log_flows[sid]:
                    skill_traj_log_flows[sid][traj.traj_id] = []

                skill_traj_log_flows[sid][traj.traj_id].append(turn.state_flow)

    marginal_flows: Dict[str, float] = {}
    for sid in all_skill_ids:
        per_traj = skill_traj_log_flows[sid]
        if per_traj:


            traj_log_sums = [_log_sum_exp(log_flows) for log_flows in per_traj.values()]
            log_marginal = _log_sum_exp(traj_log_sums) - math.log(len(per_traj))


            marginal_flows[sid] = max(min(log_marginal, 20.0), -20.0)
        else:
            marginal_flows[sid] = -20.0  

    return marginal_flows


def compute_skill_reward_variance(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:


    skill_rewards: Dict[str, List[float]] = {sid: [] for sid in all_skill_ids}

    for traj in trajectories:
        visited_in_traj: set = set()
        for turn in traj.turns:
            if (
                turn.action_type == "skill_invoke"
                and turn.skill_id in skill_rewards
                and turn.skill_id not in visited_in_traj
            ):
                skill_rewards[turn.skill_id].append(traj.reward)
                visited_in_traj.add(turn.skill_id)

    variances: Dict[str, float] = {}
    for sid in all_skill_ids:
        rewards = skill_rewards[sid]
        if len(rewards) < 2:
            variances[sid] = 0.0  
        else:
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            variances[sid] = var

    return variances


compute_skill_variances = compute_skill_reward_variance


def compute_skill_log_F_visits(
    trajectories: List[Trajectory],
    skill_id: str,
) -> List[float]:

    visits: List[float] = []
    for traj in trajectories:
        for turn in traj.turns:
            if turn.action_type == "skill_invoke" and turn.skill_id == skill_id:
                visits.append(turn.state_flow)
    return visits


def compute_skill_G(
    trajectories: List[Trajectory],
    skill_id: str,
    log_z_default: float = 0.0,
) -> float:

    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if not visits:
        return -math.inf
    log_F_mean = sum(visits) / len(visits)

    z_vals = [
        traj.log_z for traj in trajectories
        for turn in traj.turns
        if turn.action_type == "skill_invoke" and turn.skill_id == skill_id
    ]
    log_z_mean = sum(z_vals) / len(z_vals) if z_vals else log_z_default
    return log_F_mean - log_z_mean


def compute_skill_lambda_1(
    trajectories: List[Trajectory],
    skill_id: str,
    log_z_default: float = 0.0,
) -> float:

    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if not visits:
        return -math.inf
    z_vals = [
        traj.log_z for traj in trajectories
        for turn in traj.turns
        if turn.action_type == "skill_invoke" and turn.skill_id == skill_id
    ]
    log_z_mean = sum(z_vals) / len(z_vals) if z_vals else log_z_default

    log_F_hat = _log_sum_exp(visits) - math.log(len(visits))
    return log_F_hat - log_z_mean


def compute_skill_jensen_gap(
    trajectories: List[Trajectory],
    skill_id: str,
) -> float:

    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if len(visits) < 2:
        return 0.0

    mean_X = sum(visits) / len(visits)
    log_mean_exp_X = _log_sum_exp(visits) - math.log(len(visits))
    return log_mean_exp_X - mean_X


def compute_skill_lambda_tilde(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:

    lambda_1: Dict[str, float] = {
        sid: compute_skill_lambda_1(trajectories, sid)
        for sid in all_skill_ids
    }
    finite = [v for v in lambda_1.values() if v != -math.inf and not math.isnan(v)]
    if not finite:
        return {sid: 0.0 for sid in all_skill_ids}
    mean_lambda = sum(finite) / len(finite)
    return {
        sid: (v - mean_lambda) if (v != -math.inf and not math.isnan(v)) else -math.inf
        for sid, v in lambda_1.items()
    }


def compute_skill_cgf_summary(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, Dict[str, float]]:

    out: Dict[str, Dict[str, float]] = {}
    lam_tilde = compute_skill_lambda_tilde(trajectories, all_skill_ids)
    for sid in all_skill_ids:
        out[sid] = {
            "G": compute_skill_G(trajectories, sid),
            "lambda_1": compute_skill_lambda_1(trajectories, sid),
            "lambda_tilde": lam_tilde[sid],
            "jensen_gap": compute_skill_jensen_gap(trajectories, sid),
            "n_visits": len(compute_skill_log_F_visits(trajectories, sid)),
        }
    return out


def extract_trigger_steps(
    success_traj: Trajectory,
    zeta_trig: float = 1.0,
    covered_step_indices: Optional[set] = None,
) -> List[int]:

    if covered_step_indices is None:
        covered_step_indices = set()
    triggers: List[Tuple[int, float]] = []
    for t_idx, turn in enumerate(success_traj.turns):
        if t_idx in covered_step_indices:
            continue
        if turn.action_type == "skill_invoke":
            continue
        log_It = edge_log_i(turn)
        if log_It >= zeta_trig:
            triggers.append((t_idx, log_It))
    triggers.sort(key=lambda x: x[1], reverse=True)
    return [t_idx for t_idx, _ in triggers]


def identify_top_importance_steps(
    traj: Trajectory,
    top_k: int = 3,
) -> List[Tuple[int, float]]:

    if not traj.turns:
        return []

    indexed = [(i, t.step_importance) for i, t in enumerate(traj.turns)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]


def compute_ttb_balance_error(
    log_z: float,
    forward_logprobs: List[float],
    backward_logprobs: List[float],
    r_tilde: float,
    beta: float = 1.0,
    n_steps: Optional[int] = None,
    action_token_counts: Optional[List[int]] = None,
) -> float:

    if action_token_counts is not None:
        sum_fwd = sum(
            lp / max(int(k or 0), 1) if int(k or 0) > 0 else 0.0
            for lp, k in zip(forward_logprobs, action_token_counts)
        )
        sum_bwd = sum(
            lp / max(int(k or 0), 1) if int(k or 0) > 0 else 0.0
            for lp, k in zip(backward_logprobs, action_token_counts)
        )
    else:
        sum_fwd = sum(forward_logprobs)
        sum_bwd = sum(backward_logprobs)
    log_r_tilde = math.log(max(r_tilde, 1e-8))
    balance = log_z + sum_fwd - beta * log_r_tilde - sum_bwd
    if n_steps is None:
        if action_token_counts is not None:
            n_steps = max(sum(1 for k in action_token_counts if int(k or 0) > 0), 1)
        else:
            n_steps = max(len(forward_logprobs), 1)
    return (balance / max(n_steps, 1)) ** 2


def compute_batch_ttb_stats(trajectories: List[Trajectory], beta: float = 1.0) -> Dict:

    errors = []
    for traj in trajectories:
        fwd = [t.forward_logprob for t in traj.turns]
        bwd = [t.backward_logprob for t in traj.turns]
        ks = [t.action_token_count for t in traj.turns]
        err = compute_ttb_balance_error(
            traj.log_z, fwd, bwd, traj.r_tilde, beta,
            n_steps=effective_paper_steps(traj),
            action_token_counts=ks,
        )
        errors.append(err)

    if not errors:
        return {"mean_ttb_error": 0.0, "max_ttb_error": 0.0}

    return {
        "mean_ttb_error": sum(errors) / len(errors),
        "max_ttb_error": max(errors),
        "min_ttb_error": min(errors),
    }


def _per_token_normalized_log_flow(traj: Trajectory) -> float:

    log_flow = compute_forward_trajectory_log_flow(traj)
    return log_flow / effective_paper_steps(traj)


def split_by_flow_quartile(
    trajectories: List[Trajectory],
    low_frac: float = 0.25,
    high_frac: float = 0.25,
) -> Tuple[List[Trajectory], List[Trajectory]]:

    if not trajectories:
        return [], []

    sorted_trajs = sorted(trajectories, key=_per_token_normalized_log_flow)
    n = len(sorted_trajs)
    low_n = max(1, int(n * low_frac))
    high_n = max(1, int(n * high_frac))

    return sorted_trajs[:low_n], sorted_trajs[n - high_n:]


from dataclasses import dataclass, field


@dataclass
class FlowBottleneck:

    task_type: str
    step_bucket: int          
    action_type: str          
    mean_log_It: float        
    var_log_It: float         
    n_samples: int            
    example_steps: List[Dict] = field(default_factory=list)  


@dataclass
class CounterfactualPair:

    question: str
    divergence_step: int              
    context_before: List[Dict]        
    success_choice: Dict              
    failure_choice: Dict              
    success_downstream: Dict          
    failure_downstream: Dict          
    reward_gap: float


@dataclass
class BottleneckDiagnosis:

    task_type: str
    bottleneck: FlowBottleneck
    counterfactual_pairs: List[CounterfactualPair]
    current_skill_coverage: str       
    suggested_edit_type: str          


def compute_flow_bottlenecks(
    trajectories: List[Trajectory],
    bucket_size: int = 2,
    min_samples: int = 4,
    var_threshold: float = 2.0,
    mean_threshold: float = 1.0,
) -> Dict[str, List[FlowBottleneck]]:

    from collections import defaultdict


    clusters: Dict[tuple, List[Dict]] = defaultdict(list)
    for traj in trajectories:
        tt = traj.task_type
        for i, turn in enumerate(traj.turns):
            if getattr(turn, 'action_type', '') == 'skill_invoke':
                continue
            log_It = edge_log_i(turn)
            bucket = i // bucket_size
            at = getattr(turn, 'action_type', 'unknown')
            clusters[(tt, bucket, at)].append({
                'log_It': log_It,
                'action_type': at,
                'instruction': (getattr(turn, 'instruction', '') or '')[:60],
                'observation': (getattr(turn, 'observation', '') or '')[:60],
                'I_t': getattr(turn, 'step_importance', 0.0),
                'reward': traj.r_tilde,
                'step': i,
            })


    result: Dict[str, List[FlowBottleneck]] = defaultdict(list)
    for (tt, bucket, at), steps in clusters.items():
        if len(steps) < min_samples:
            continue
        values = [s['log_It'] for s in steps]
        mean_v = sum(values) / len(values)
        var_v = sum((v - mean_v) ** 2 for v in values) / len(values)

        if var_v > var_threshold and abs(mean_v) > mean_threshold:
            result[tt].append(FlowBottleneck(
                task_type=tt,
                step_bucket=bucket,
                action_type=at,
                mean_log_It=round(mean_v, 2),
                var_log_It=round(var_v, 2),
                n_samples=len(steps),
                example_steps=sorted(steps, key=lambda s: abs(s['log_It']), reverse=True)[:3],
            ))


    for tt in result:
        result[tt].sort(key=lambda b: b.var_log_It, reverse=True)

    return dict(result)


def extract_counterfactual_pairs(
    trajectories: List[Trajectory],
    min_reward_gap: float = 0.3,
) -> Dict[str, List[CounterfactualPair]]:

    from collections import defaultdict

    by_question: Dict[str, List[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_question[t.question[:200]].append(t)

    result: Dict[str, List[CounterfactualPair]] = defaultdict(list)

    for q_key, q_trajs in by_question.items():
        if len(q_trajs) < 2:
            continue
        q_trajs.sort(key=lambda t: t.r_tilde, reverse=True)
        best, worst = q_trajs[0], q_trajs[-1]
        gap = best.r_tilde - worst.r_tilde
        if gap < min_reward_gap:
            continue

        tt = best.task_type


        best_turns = [t for t in best.turns if getattr(t, 'action_type', '') != 'skill_invoke']
        worst_turns = [t for t in worst.turns if getattr(t, 'action_type', '') != 'skill_invoke']

        div_step = 0
        context = []
        for i in range(min(len(best_turns), len(worst_turns))):
            bt, wt = best_turns[i], worst_turns[i]
            if getattr(bt, 'action_type', '') != getattr(wt, 'action_type', ''):
                div_step = i
                break

            bi = (getattr(bt, 'instruction', '') or '')[:50]
            wi = (getattr(wt, 'instruction', '') or '')[:50]
            if bi != wi and i > 0:
                div_step = i
                break
            context.append({
                'action_type': getattr(bt, 'action_type', ''),
                'instruction': bi,
            })
        else:
            div_step = min(len(best_turns), len(worst_turns)) - 1

        if div_step >= len(best_turns) or div_step >= len(worst_turns):
            continue

        bt = best_turns[div_step]
        wt = worst_turns[div_step]

        result[tt].append(CounterfactualPair(
            question=q_key[:100],
            divergence_step=div_step,
            context_before=context,
            success_choice={
                'action_type': getattr(bt, 'action_type', ''),
                'instruction': (getattr(bt, 'instruction', '') or '')[:80],
                'observation': (getattr(bt, 'observation', '') or '')[:80],
                'I_t': round(getattr(bt, 'step_importance', 0.0), 2),
            },
            failure_choice={
                'action_type': getattr(wt, 'action_type', ''),
                'instruction': (getattr(wt, 'instruction', '') or '')[:80],
                'observation': (getattr(wt, 'observation', '') or '')[:80],
                'I_t': round(getattr(wt, 'step_importance', 0.0), 2),
            },
            success_downstream={
                'n_remaining_steps': len(best_turns) - div_step - 1,
                'final_reward': round(best.r_tilde, 2),
            },
            failure_downstream={
                'n_remaining_steps': len(worst_turns) - div_step - 1,
                'final_reward': round(worst.r_tilde, 2),
            },
            reward_gap=round(gap, 2),
        ))

    return dict(result)


def build_bottleneck_diagnoses(
    bottlenecks: List[FlowBottleneck],
    counterfactual_pairs: List[CounterfactualPair],
    existing_skills: list,
    task_type: str,
) -> List[BottleneckDiagnosis]:

    diagnoses = []
    cps = counterfactual_pairs or []

    for bn in bottlenecks[:3]:

        matched_cps = [
            cp for cp in cps
            if cp.divergence_step // 2 == bn.step_bucket  
        ][:2]


        coverage = "none"
        edit_type = "ADD"
        for skill in existing_skills:
            plan = getattr(skill, 'plan', '') or ''
            if bn.action_type in plan.lower():
                usage = getattr(skill.meta, 'usage_count', 0)
                success = getattr(skill.meta, 'success_count', 0)
                rate = success / max(usage, 1)
                if rate < 0.4:
                    coverage = f"{skill.meta.skill_id} (usage={usage}, success_rate={rate:.0%})"
                    edit_type = "UPDATE"
                else:
                    coverage = f"{skill.meta.skill_id} (working well, {rate:.0%})"
                    edit_type = "KEEP"
                break

        if bn.var_log_It > 5.0 and coverage != "none":
            edit_type = "SPLIT"

        diagnoses.append(BottleneckDiagnosis(
            task_type=task_type,
            bottleneck=bn,
            counterfactual_pairs=matched_cps,
            current_skill_coverage=coverage,
            suggested_edit_type=edit_type,
        ))

    return diagnoses
