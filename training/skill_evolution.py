

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from training.trajectory import Trajectory
from training.flow_metrics import (
    compute_flow_entropy,
    compute_skill_marginal_flows,
    compute_skill_reward_variance,
    compute_skill_variances,
    compute_skill_cgf_summary,
    edge_logprob_tilde,
    split_by_flow_quartile,
)
from src.skills.format import SkillEntry
from src.skills.workspace import SkillWorkspace
from src.skills.skill_creator import SkillCreator

logger = logging.getLogger(__name__)


class PlateauDetector:


    def __init__(self, window_size: int = 20, rho: float = 0.05, m_consecutive: int = 2):
        self.W = max(int(window_size), 2)
        self.rho = float(rho)
        self.M = max(int(m_consecutive), 1)
        self._delta_sq: List[float] = []   
        self._consec_count = 0
        self._last_trigger_step = -1

    def update(self, step: int, batch_delta_sq: float) -> None:

        self._delta_sq.append(float(batch_delta_sq))

    def should_trigger(self, step: int) -> bool:

        if len(self._delta_sq) < 2 * self.W:
            return False

        delta_recent = self._delta_sq[-self.W:]
        delta_prev = self._delta_sq[-2 * self.W:-self.W]
        mean_recent = sum(delta_recent) / self.W
        mean_prev = sum(delta_prev) / self.W
        if mean_prev <= 1e-12:
            return False
        rel_decrease = (mean_prev - mean_recent) / mean_prev
        if rel_decrease < self.rho:
            self._consec_count += 1
        else:
            self._consec_count = 0
        if self._consec_count >= self.M and (step - self._last_trigger_step) >= self.W:
            self._consec_count = 0
            self._last_trigger_step = step
            return True
        return False


@dataclass
class CurationClassification:

    prune: List[str] = field(default_factory=list)     
    retain: List[str] = field(default_factory=list)    
    refine: List[str] = field(default_factory=list)    
    cgf: Dict[str, Dict[str, float]] = field(default_factory=dict)  


def partition_skills_DRU(
    trajectories: List[Trajectory],
    skill_ids: List[str],
    n_minus_counter: Dict[str, int],
    G_threshold: float = 0.0,
    J_threshold: float = 1.0,
    K_minus: int = 2,
) -> CurationClassification:

    cgf = compute_skill_cgf_summary(trajectories, skill_ids)
    result = CurationClassification(cgf=cgf)
    for sid in skill_ids:

        lam_t = cgf[sid]["lambda_tilde"]
        if lam_t < 0:
            n_minus_counter[sid] = n_minus_counter.get(sid, 0) + 1
        else:
            n_minus_counter[sid] = 0  


        if n_minus_counter[sid] >= K_minus:
            result.prune.append(sid)
            continue

        G_s = cgf[sid]["G"]
        jensen = cgf[sid]["jensen_gap"]

        if G_s >= G_threshold and jensen <= J_threshold:
            result.retain.append(sid)
        else:

            result.refine.append(sid)
    return result


_ACCEPT_ACTIONS = frozenset({"accept", "answer", "submit", "finish", "final", "done"})
_SKILL_ACTIONS = frozenset({"skill_invoke", "skill"})


def action_type_to_alpha(action_type: str) -> str:

    a = (action_type or "").lower()
    if a in _SKILL_ACTIONS:
        return "skill"
    if a in _ACCEPT_ACTIONS:
        return "accept"
    return "act"


class TaskTypeAccuracyTracker:


    def __init__(self, threshold: float = 0.5, min_count: int = 10):
        self.threshold = threshold
        self.min_count = min_count
        self._counts: Dict[str, int] = defaultdict(int)
        self._correct: Dict[str, int] = defaultdict(int)

    def track_result(self, task_type: str, correct: bool) -> None:

        self._counts[task_type] += 1
        if correct:
            self._correct[task_type] += 1

    def get_accuracy(self, task_type: str) -> float:

        total = self._counts.get(task_type, 0)
        if total == 0:
            return 0.5  
        return self._correct.get(task_type, 0) / total

    def get_struggling_types(self) -> List[str]:

        struggling = []
        for tt, count in self._counts.items():
            if count >= self.min_count:
                acc = self._correct.get(tt, 0) / count
                if acc < self.threshold:
                    struggling.append(tt)
        return struggling

    def all_accuracies(self) -> Dict[str, float]:

        return {
            tt: self._correct.get(tt, 0) / max(count, 1)
            for tt, count in self._counts.items()
        }

    def summary(self) -> str:

        accs = self.all_accuracies()
        parts = [f"{tt}={acc:.2f}({self._counts[tt]})" for tt, acc in sorted(accs.items())]
        return ", ".join(parts) if parts else "no data"


class SkillAcceptanceGate:


    def __init__(
        self,
        run_episode_fn=None,
        n_eval_episodes: int = 0,
        min_improvement: float = 0.05,
    ):
        self.run_episode_fn = run_episode_fn
        self.n_eval_episodes = n_eval_episodes if run_episode_fn is not None else 0
        self.min_improvement = min_improvement

    def evaluate_candidate(
        self,
        candidate: SkillEntry,
        task_type: str,
        baseline_trajectories: List[Trajectory],
    ) -> Tuple[bool, Dict]:


        if baseline_trajectories:
            rewards = [float(getattr(t, 'r_tilde', 0.0)) for t in baseline_trajectories]
            baseline_mean = sum(rewards) / len(rewards)
            baseline_var = sum((r - baseline_mean) ** 2 for r in rewards) / max(len(rewards), 1)
            baseline_se = (baseline_var / max(len(rewards), 1)) ** 0.5
        else:
            baseline_mean, baseline_se = 0.5, 0.1


        if self.n_eval_episodes <= 0 or self.run_episode_fn is None:

            body = (getattr(candidate, 'plan', '') or '').strip()
            desc = (getattr(candidate, 'description', '') or '').strip()
            ok = len(body) >= 10 and len(desc) >= 5
            return ok, {
                "candidate_mean": baseline_mean,
                "baseline_mean": baseline_mean,
                "improvement": 0.0,
                "threshold": 0.0,
                "mode": "passthrough",
                "sanity_ok": ok,
            }


        candidate_rewards: List[float] = []
        for _ in range(self.n_eval_episodes):
            try:
                r = self.run_episode_fn(task_type, candidate)
                candidate_rewards.append(float(r))
            except Exception as e:
                logger.warning(f"[Gate] eval episode failed: {e}")

        if not candidate_rewards:
            return False, {"error": "all eval episodes failed", "mode": "eval"}

        candidate_mean = sum(candidate_rewards) / len(candidate_rewards)
        improvement = candidate_mean - baseline_mean
        threshold = max(self.min_improvement, 2.0 * baseline_se)
        accepted = improvement > threshold

        return accepted, {
            "candidate_mean": round(candidate_mean, 3),
            "baseline_mean": round(baseline_mean, 3),
            "improvement": round(improvement, 3),
            "threshold": round(threshold, 3),
            "baseline_se": round(baseline_se, 3),
            "n_candidate_eval": len(candidate_rewards),
            "mode": "eval",
        }


class FlowSkillEvolutionManager:


    def __init__(
        self,
        workspace: SkillWorkspace,
        skill_creator: SkillCreator,
        delta_h: float = -1.0,           
        delta_prune: float = -10.0,      
        min_usage_before_prune: int = 20,
        max_skills_total: int = 60,
        evolution_phase_steps: int = 20,
        min_trajs_for_evolution: int = 16,
        experience_store=None,
        beta: float = 1.0,              
        accuracy_tracker: Optional[TaskTypeAccuracyTracker] = None,
        accuracy_threshold: float = 0.5,
    ):
        self.workspace = workspace
        self.skill_creator = skill_creator
        self.experience_store = experience_store
        self.delta_h = delta_h
        self.delta_prune = delta_prune
        self.min_usage_before_prune = min_usage_before_prune
        self.max_skills_total = max_skills_total
        self.evolution_phase_steps = evolution_phase_steps
        self.min_trajs_for_evolution = min_trajs_for_evolution
        self.beta = beta
        self.accuracy_tracker = accuracy_tracker
        self.accuracy_threshold = accuracy_threshold


        self._evolution_history: List[Dict] = []
        self._total_evolutions = 0
        self._skills_created = 0
        self._skills_pruned = 0


        self._low_flow_invocations: Dict[str, List[Dict]] = {}


    def evolve_for_type(
        self,
        task_type: str,
        observations: List[Dict],
        failed_trajectories: List[Trajectory],
        step: int,

        high_flow_trajs: Optional[List[Trajectory]] = None,
        critical_steps: Optional[List[Dict]] = None,
        dag_comparisons: Optional[List[Dict]] = None,

        bottleneck_diagnoses: Optional[List] = None,
        counterfactual_pairs: Optional[List] = None,
        acceptance_gate: Optional["SkillAcceptanceGate"] = None,
        curation: Optional[CurationClassification] = None,
    ) -> List[SkillEntry]:

        n_critical = len(critical_steps) if critical_steps else 0
        n_dag = len(dag_comparisons) if dag_comparisons else 0
        n_diag = len(bottleneck_diagnoses) if bottleneck_diagnoses else 0
        n_cf = len(counterfactual_pairs) if counterfactual_pairs else 0
        logger.info(
            f"[Evolution v5] {task_type}: {len(observations)} obs, "
            f"{len(failed_trajectories)} low-flow, "
            f"{len(high_flow_trajs or [])} high-flow, "
            f"{n_critical} I(t) steps, {n_dag} DAG pairs, "
            f"{n_diag} bottleneck diagnoses, {n_cf} counterfactual pairs"
        )


        result = self.skill_creator.curate_and_evolve_tips(
            task_type=task_type,
            observations=observations,
            failed_trajectories=failed_trajectories,
            workspace=self.workspace,
            high_flow_trajs=high_flow_trajs,
            critical_steps=critical_steps,
            dag_comparisons=dag_comparisons,
            bottleneck_diagnoses=bottleneck_diagnoses,
            counterfactual_pairs=counterfactual_pairs,
        )

        if curation is not None:
            prune_set = set(curation.prune)
            retain_set = set(curation.retain)
            refine_set = set(curation.refine)


            result["deleted"] = sorted(prune_set | (set(result.get("deleted", [])) & prune_set))
            result["updated"] = [
                (old_id, entry)
                for old_id, entry in result.get("updated", [])
                if old_id in refine_set and old_id not in retain_set
            ]
            logger.info(
                f"[Evolution CGF] Applied D/R/U guard: prune={len(prune_set)}, "
                f"retain={len(retain_set)}, refine={len(refine_set)}, "
                f"allowed_updates={len(result['updated'])}"
            )


        for sid in result["deleted"]:
            self.workspace.remove(sid)

        updated_entries: List[SkillEntry] = []
        for old_id, new_entry in result["updated"]:
            new_entry.meta.creation_step = step
            self.workspace.add(new_entry)  
            updated_entries.append(new_entry)

        added = []
        rejected_by_gate: List[SkillEntry] = []
        for tip in result["added"]:
            tip.meta.creation_step = step

            if acceptance_gate is not None:
                try:
                    accepted, stats = acceptance_gate.evaluate_candidate(
                        candidate=tip,
                        task_type=task_type,
                        baseline_trajectories=failed_trajectories + (high_flow_trajs or []),
                    )
                    if not accepted:
                        rejected_by_gate.append(tip)
                        logger.info(
                            f"[Evolution v5] Gate REJECTED {tip.meta.skill_id}: "
                            f"candidate_mean={stats.get('candidate_mean'):.3f} vs "
                            f"baseline_mean={stats.get('baseline_mean'):.3f}"
                        )
                        continue
                    else:
                        logger.info(
                            f"[Evolution v5] Gate ACCEPTED {tip.meta.skill_id}: "
                            f"improvement={stats.get('improvement'):.3f}"
                        )
                except Exception as e:
                    logger.warning(f"[Evolution v5] Gate error (admitting by default): {e}")

            if self.workspace.add(tip):
                added.append(tip)

        self._total_evolutions += 1
        self._skills_created += len(added)

        summary = f"+{len(added)} add, {len(result['updated'])} update, {len(result['deleted'])} delete"
        if rejected_by_gate:
            summary += f", {len(rejected_by_gate)} gate-rejected"
        if result["skipped"]:
            summary += " (no new tip needed)"
        logger.info(f"[Evolution v5] {task_type}: {summary}, workspace={self.workspace.size}")

        return added + updated_entries


    def maybe_evolve(
        self,
        trajectories: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
        step: int,
    ) -> Tuple[bool, List[SkillEntry]]:

        if len(trajectories) < self.min_trajs_for_evolution:
            return False, []

        if not self.should_evolve(skill_marginal_flows, trajectories):
            return False, []

        delta_h = self._adaptive_delta_h()
        logger.info(
            f"[Evolution] Step {step}: triggering skill evolution "
            f"(H_skill < δ_H={delta_h:.3f}, S_size={self.workspace.size})"
        )

        new_skills = self.evolve(trajectories, skill_marginal_flows, step)
        return True, new_skills

    def _adaptive_delta_h(self) -> float:

        if self.delta_h > 0:
            return self.delta_h
        s_size = max(self.workspace.size, 2)
        return 1.0 - math.log(2) / math.log(s_size)

    def _compute_skill_flow_entropy(
        self, skill_marginal_flows: Dict[str, float],
    ) -> Tuple[float, float, float]:

        if not skill_marginal_flows:
            return 0.0, 0.0, 0.0

        log_flows = list(skill_marginal_flows.values())
        n_skills = len(log_flows)
        h_max = math.log(n_skills) if n_skills > 1 else 1.0


        max_lf = max(log_flows)
        exp_shifted = [math.exp(lf - max_lf) if (lf - max_lf) > -500 else 0.0
                       for lf in log_flows]
        total = sum(exp_shifted)

        if total <= 0:
            return 0.0, h_max, 0.0

        probs = [e / total for e in exp_shifted]
        h_skill = -sum(p * math.log(p) for p in probs if p > 0)
        h_ratio = h_skill / h_max if h_max > 0 else 0.0

        return h_skill, h_max, h_ratio

    def should_evolve(
        self,
        skill_marginal_flows: Dict[str, float],
        phase_trajectories: Optional[List[Trajectory]] = None,
    ) -> bool:

        s_size = self.workspace.size

        if s_size >= self.max_skills_total:
            logger.info(f"[Evolution] Skill library full ({s_size}/{self.max_skills_total}), "
                        f"trying to prune before evolution")
            self._try_prune()
            s_size = self.workspace.size

        capacity_ok = s_size < self.max_skills_total


        loss_stagnation = False
        if phase_trajectories and len(phase_trajectories) >= 24:
            mid = len(phase_trajectories) // 2
            first_half = phase_trajectories[:mid]
            second_half = phase_trajectories[mid:]

            def _balance_error(traj):
                log_F = self._trajectory_log_flow(traj)
                log_R_beta = self.beta * math.log(max(traj.r_tilde, 1e-10))
                return abs(log_F - log_R_beta)
            avg_first = sum(_balance_error(t) for t in first_half) / len(first_half)
            avg_second = sum(_balance_error(t) for t in second_half) / len(second_half)

            loss_stagnation = avg_second >= avg_first * 0.95
            logger.info(
                f"[Evolution] TTB balance error: first_half={avg_first:.1f}, "
                f"second_half={avg_second:.1f}, stagnation={loss_stagnation}"
            )


        n_active = sum(1 for f in skill_marginal_flows.values() if f > -15.0)
        n_total = max(len(skill_marginal_flows), 1)
        utilization_trigger = n_active < n_total * 0.4


        accuracy_trigger = False
        struggling_types = []
        if self.accuracy_tracker:
            struggling_types = self.accuracy_tracker.get_struggling_types()
            accuracy_trigger = len(struggling_types) > 0
            if accuracy_trigger:
                logger.info(
                    f"[Evolution] Struggling task_types: {struggling_types} "
                    f"(accuracies: {self.accuracy_tracker.summary()})"
                )


        flow_entropy_trigger = False
        if skill_marginal_flows and s_size >= 3:
            _, _, h_ratio = self._compute_skill_flow_entropy(skill_marginal_flows)
            delta_h = self._adaptive_delta_h()
            flow_entropy_trigger = h_ratio < delta_h
            if flow_entropy_trigger:
                logger.info(
                    f"[Evolution] Flow entropy trigger: H_ratio={h_ratio:.3f} < δ_H={delta_h:.3f}"
                )

        triggered = (loss_stagnation or utilization_trigger or accuracy_trigger or flow_entropy_trigger) and capacity_ok
        logger.info(
            f"[Evolution] active_skills={n_active}/{s_size}, "
            f"loss_stagnation={loss_stagnation}, "
            f"util_trigger={utilization_trigger}, "
            f"accuracy_trigger={accuracy_trigger} (struggling={struggling_types}), "
            f"flow_entropy_trigger={flow_entropy_trigger}, "
            f"|S|={s_size}/{self.max_skills_total} → "
            f"triggered={triggered}"
        )
        return triggered

    def evolve(
        self,
        trajectories: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
        step: int,
    ) -> List[SkillEntry]:

        self._total_evolutions += 1


        self._update_flow_scores(skill_marginal_flows)


        pruned = self._try_prune(current_step=step)
        if pruned:
            logger.info(f"[Evolution] Pruned {len(pruned)} skills: {pruned}")
            self._skills_pruned += len(pruned)


        sorted_trajs = sorted(trajectories, key=lambda t: self._trajectory_log_flow(t))
        n = len(sorted_trajs)
        low_flow_trajs = sorted_trajs[:max(1, n // 4)]
        high_flow_trajs = sorted_trajs[max(0, n - n // 4):]


        by_type_all = defaultdict(list)
        for t in trajectories:
            by_type_all[t.task_type].append(t)

        high_by_type = defaultdict(list)
        low_by_type = defaultdict(list)
        for tt, type_trajs in by_type_all.items():
            sorted_type = sorted(type_trajs, key=lambda t: self._trajectory_log_flow(t))
            n_tt = len(sorted_type)
            q = max(1, n_tt // 4)
            low_by_type[tt] = sorted_type[:q]
            high_by_type[tt] = sorted_type[n_tt - q:]

        balanced_high = []
        balanced_low = []
        for tt in set(list(high_by_type.keys()) + list(low_by_type.keys())):
            h = sorted(high_by_type.get(tt, []), key=lambda t: -self._trajectory_log_flow(t))[:3]
            l = sorted(low_by_type.get(tt, []), key=lambda t: self._trajectory_log_flow(t))[:3]
            balanced_high.extend(h)
            balanced_low.extend(l)

        critique_exps = self.skill_creator.cross_trajectory_critique(
            high_flow_trajs=balanced_high or high_flow_trajs,
            low_flow_trajs=balanced_low or low_flow_trajs,
        )
        if critique_exps and self.experience_store is not None:
            from training.experience_store import Experience
            for exp_dict in critique_exps:

                avg_reward = sum(t.reward for t in high_flow_trajs) / max(len(high_flow_trajs), 1)
                exp = Experience(
                    condition=exp_dict["condition"],
                    action=exp_dict["action"],
                    task_types=exp_dict.get("task_types", []),
                    reward_signal=avg_reward,
                    source_step=step,
                )
                self.experience_store.add(exp)
            logger.info(f"[Evolution] Extracted {len(critique_exps)} experiences via cross-trajectory critique")


        mined_patterns = []
        for traj in high_flow_trajs[:8]:
            patterns = self.skill_creator.backward_pattern_mine(traj, top_k=2)
            mined_patterns.extend(patterns)


        distilled_count = self._distill_high_flow_skills(
            high_flow_trajs=high_flow_trajs,
            skill_marginal_flows=skill_marginal_flows,
            step=step,
        )
        if distilled_count:
            logger.info(f"[Evolution] Distilled {distilled_count} high-flow skills")


        self._record_low_flow_invocations(low_flow_trajs, skill_marginal_flows)


        refined_count = self._refine_low_flow_skills(skill_marginal_flows, step)
        if refined_count:
            logger.info(f"[Evolution] Refined {refined_count} low-flow skills")


        logger.info("[Evolution] v2: extracting atomic tips per task_type (ADD, not MERGE)")
        by_type = defaultdict(list)
        for traj in trajectories:
            by_type[traj.task_type].append(traj)

        for tt, type_trajs in by_type.items():
            sorted_t = sorted(type_trajs, key=lambda t: t.reward, reverse=True)
            high_t = [t for t in sorted_t if t.reward >= 0.5][:5]
            low_t = [t for t in sorted_t if t.reward <= 0.0][:5]
            if len(high_t) < 2:
                high_t = sorted_t[:3]

            logger.info(
                f"[Evolution] Atomic tips for {tt}: "
                f"high={len(high_t)}, low={len(low_t)}"
            )


            new_tips = self.skill_creator.extract_atomic_tips(
                high_trajs=high_t,
                low_trajs=low_t,
                task_type=tt,
                max_tips=2,  
            )


            for tip in new_tips:
                tip.meta.creation_step = getattr(self, '_current_step', 0)
                self.workspace.add(tip)
                self._skills_created += 1
                logger.info(
                    f"[Evolution] Added tip for {tt}: "
                    f"\"{tip.plan[:80]}\" ({len(tip.plan.split())} words)"
                )


        _legacy = getattr(self, '_legacy_evolution_enabled', False)
        if _legacy:
            logger.info("[Evolution Legacy] Running deprecated Living Doc + flow_guided_evolution")
            current_skills = self.workspace.get_all()
            struggling_types = self.accuracy_tracker.get_struggling_types() if self.accuracy_tracker else []
            new_skills = self.skill_creator.flow_guided_evolution(
                low_flow_trajs=balanced_low or low_flow_trajs,
                high_flow_trajs=balanced_high or high_flow_trajs,
                current_skills=current_skills,
                skill_marginal_flows=skill_marginal_flows,
                mined_patterns=mined_patterns,
                creation_step=step,
                struggling_types=struggling_types,
            )
            added = self.workspace.add_batch(new_skills)
            self._skills_created += added


        self._evolution_history.append({
            "step": step,
            "h_flow_before": compute_flow_entropy(trajectories),
            "skills_before": len(current_skills),
            "pruned": len(pruned),
            "distilled": distilled_count,
            "refined": refined_count,
            "new_skills": added,
            "skills_after": self.workspace.size,
        })

        logger.info(
            f"[Evolution] Step {step} complete: "
            f"+{added} new skills, -{len(pruned)} pruned, "
            f"~{distilled_count} distilled, ~{refined_count} refined → total={self.workspace.size}"
        )

        return new_skills

    def _consolidate_all_skills(
        self,
        trajectories: List[Trajectory],
        high_flow_trajs: List[Trajectory],
        step: int,
        max_consolidate: int = 3,
    ) -> int:

        if not high_flow_trajs:
            return 0


        from src.skills.skill_prompts import format_trajectory_for_skill_generation, clean_skill_output
        from src.skills.format import TaskTypeSkillDocument

        type_trajs = defaultdict(list)
        for traj in high_flow_trajs:
            type_trajs[traj.task_type].append(traj)


        sorted_trajs = sorted(trajectories, key=lambda t: t.reward)
        low_trajs_by_type = defaultdict(list)
        for traj in sorted_trajs[:len(sorted_trajs)//4]:
            low_trajs_by_type[traj.task_type].append(traj)

        consolidated = 0

        all_skills = self.workspace.get_all()
        all_skill_ids = [s.meta.skill_id for s in all_skills]
        skill_variances = compute_skill_variances(trajectories, all_skill_ids)


        type_variance = {}
        for tt, tt_trajs in type_trajs.items():
            type_skills = [s for s in all_skills if tt in (s.meta.task_types or [])]
            if type_skills:
                avg_var = sum(skill_variances.get(s.meta.skill_id, 0.0) for s in type_skills) / len(type_skills)
                type_variance[tt] = avg_var
            else:
                type_variance[tt] = 1.0  

        sorted_types = sorted(type_variance.keys(), key=lambda t: type_variance[t], reverse=True)

        for tt in sorted_types[:max_consolidate]:
            ht = type_trajs.get(tt, [])[:3]
            lt = low_trajs_by_type.get(tt, [])[:2]
            if not ht:
                continue


            s_traj = ht[0]
            f_traj = lt[0] if lt else ht[-1]
            s_text = format_trajectory_for_skill_generation(s_traj)
            f_text = format_trajectory_for_skill_generation(f_traj)

            outcome_summary = (
                f"success: reward={getattr(s_traj, 'reward', 0.0):.3f}, "
                f"status={'success' if getattr(s_traj, 'reward', 0.0) >= 0.5 else 'failure'}\n"
                f"contrast: reward={getattr(f_traj, 'reward', 0.0):.3f}, "
                f"status={'success' if getattr(f_traj, 'reward', 0.0) >= 0.5 else 'failure'}"
            )

            from src.skills.skill_prompts import GENERATE_SKILL_PROMPT
            prompt = GENERATE_SKILL_PROMPT.format(
                task_type=tt,
                success_trajectory=s_text,
                failure_trajectory=f_text,
                outcome_summary=outcome_summary,
            )

            try:
                new_raw = self.skill_creator.m_exec.execute(prompt, max_tokens=2000, temperature=0.4)
                new_raw = clean_skill_output(new_raw)
            except Exception as e:
                logger.warning(f"[Consolidation] Generate failed for {tt}: {e}")
                continue


            existing_doc = self.workspace.get_type_document(tt)
            existing_content = existing_doc.consolidated_strategy if existing_doc else ""


            merged = self.skill_creator.merge_into_living_document(
                task_type=tt,
                existing_document=existing_content,
                new_skill_contents=[new_raw],
            )

            if not merged:
                continue


            refined = self.skill_creator.refine_living_document(
                task_type=tt,
                document=merged,
                word_threshold=1000,
            )


            accuracy = 0.5
            sample_count = 0
            if self.accuracy_tracker:
                accuracy = self.accuracy_tracker.get_accuracy(tt)
                sample_count = self.accuracy_tracker._counts.get(tt, 0)

            type_skills = self.workspace.get_skills_by_task_type(tt)
            doc = TaskTypeSkillDocument(
                task_type=tt,
                consolidated_strategy=refined,
                variant_skill_ids=[s.meta.skill_id for s in type_skills],
                proven_patterns=[],
                accuracy=accuracy,
                sample_count=sample_count,
            )
            self.workspace.update_type_document(doc)
            consolidated += 1
            logger.info(
                f"[Consolidation] Living document for {tt}: "
                f"{len(refined.split())} words (acc={accuracy:.2f})"
            )

        return consolidated

    def _update_type_documents(
        self,
        trajectories: List[Trajectory],
        step: int,
    ) -> None:

        from src.skills.format import TaskTypeSkillDocument

        affected_types = set(t.task_type for t in trajectories)

        for tt in affected_types:
            type_skills = self.workspace.get_skills_by_task_type(tt)


            proven_patterns = []
            if self.experience_store:
                exps = self.experience_store.retrieve(query=tt, task_type=tt, top_k=5)
                for exp in exps:
                    proven_patterns.append(f"When: {exp.condition} → Do: {exp.action}")


            accuracy = 0.5
            sample_count = 0
            if self.accuracy_tracker:
                accuracy = self.accuracy_tracker.get_accuracy(tt)
                sample_count = self.accuracy_tracker._counts.get(tt, 0)


            existing_doc = self.workspace.get_type_document(tt)
            existing_strategy = existing_doc.consolidated_strategy if existing_doc else ""

            doc = TaskTypeSkillDocument(
                task_type=tt,
                consolidated_strategy=existing_strategy,
                variant_skill_ids=[s.meta.skill_id for s in type_skills],
                proven_patterns=proven_patterns,
                accuracy=accuracy,
                sample_count=sample_count,
            )
            self.workspace.update_type_document(doc)
            logger.debug(
                f"[TypeDoc] Updated {tt}: acc={accuracy:.2f}, "
                f"{len(proven_patterns)} patterns, "
                f"doc={'YES' if existing_strategy else 'NO'} "
                f"({len(existing_strategy.split())}w)" if existing_strategy else ""
            )

    @staticmethod
    def _trajectory_log_flow(traj: Trajectory) -> float:

        return traj.log_z + sum(edge_logprob_tilde(t, "forward") for t in traj.turns)

    def _update_flow_scores(self, skill_marginal_flows: Dict[str, float]) -> None:

        for sid, flow in skill_marginal_flows.items():
            if flow > -20.0:  
                self.workspace.update_flow_score(sid, flow, alpha=0.3)

    def _try_prune(self, current_step: int = -1) -> List[str]:

        if self.workspace.size <= self.max_skills_total:
            return []

        merged_ids = []

        while self.workspace.size > self.max_skills_total:
            pair = self._find_most_similar_pair()
            if pair is None:
                logger.info("[Evolution] No similar pair found (sim > 0.70) — cannot merge further")
                break

            sid_a, sid_b, sim = pair
            skill_a = self.workspace.get_by_id(sid_a)
            skill_b = self.workspace.get_by_id(sid_b)
            if skill_a is None or skill_b is None:
                break


            merge_prompt = f"""Merge these two similar skills into ONE unified skill that combines their strengths.

Skill A:
Name: {skill_a.name}
Trigger: {skill_a.trigger}
Plan: {skill_a.plan}
Pitfall: {skill_a.pitfall}

Skill B:
Name: {skill_b.name}
Trigger: {skill_b.trigger}
Plan: {skill_b.plan}
Pitfall: {skill_b.pitfall}

Similarity: {sim:.2f}

Output ONE merged skill in YAML format:
---
name: "Short Name (max 7 words)"
description: "Combined description"
trigger: "When to use: merged trigger conditions"
plan: |
  1. [Merged step plan]
  ...
pitfall: "Combined pitfalls from both skills"
constraint: "Combined constraints"
---"""

            try:
                response = self.skill_creator.m_exec.execute(merge_prompt, max_tokens=1500, temperature=0.3)
                from src.skills.format import parse_skill_blocks
                parsed = parse_skill_blocks(response)
                if parsed:
                    merged_skill = parsed[0]

                    keep_id = sid_a if (skill_a.meta.flow_score >= skill_b.meta.flow_score) else sid_b
                    remove_id = sid_b if keep_id == sid_a else sid_a
                    merged_skill.meta.skill_id = keep_id
                    merged_skill.meta.flow_score = max(skill_a.meta.flow_score, skill_b.meta.flow_score)
                    merged_skill.meta.creation_step = current_step
                    merged_skill.meta.source = "consolidation_merge"

                    types_a = set(skill_a.meta.task_types) if skill_a.meta.task_types else set()
                    types_b = set(skill_b.meta.task_types) if skill_b.meta.task_types else set()
                    merged_skill.meta.task_types = list(types_a | types_b)

                    self.workspace.add(merged_skill)  
                    self.workspace.remove(remove_id)  
                    merged_ids.append(remove_id)
                    logger.info(
                        f"[Evolution] Merged {sid_a}+{sid_b} (sim={sim:.2f}) → {keep_id}: {merged_skill.name}"
                    )
                else:
                    logger.warning(f"[Evolution] Merge parse failed for {sid_a}+{sid_b}")
                    break
            except Exception as e:
                logger.warning(f"[Evolution] Merge failed for {sid_a}+{sid_b}: {e}")
                break

        return merged_ids

    def _find_most_similar_pair(self, threshold: float = 0.70) -> Optional[Tuple[str, str, float]]:

        cache = self.workspace._compute_skill_embeddings()
        if cache is None:
            return None

        skill_ids = cache["skill_ids"]
        embeddings = cache["embeddings"]  
        n = len(skill_ids)
        if n < 2:
            return None


        import numpy as np
        avg_embs = embeddings.mean(axis=1)  

        norms = np.linalg.norm(avg_embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        avg_embs = avg_embs / norms


        sim_matrix = avg_embs @ avg_embs.T  

        np.fill_diagonal(sim_matrix, -1.0)

        max_idx = int(np.argmax(sim_matrix))
        i, j = divmod(max_idx, n)
        max_sim = float(sim_matrix[i, j])

        if max_sim < threshold:
            return None

        return skill_ids[i], skill_ids[j], max_sim

    def _record_low_flow_invocations(
        self,
        low_flow_trajs: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
    ) -> None:

        for traj in low_flow_trajs:
            for t_idx, turn in enumerate(traj.turns):
                if turn.action_type == "skill_invoke" and turn.skill_id:
                    sid = turn.skill_id
                    if sid not in self._low_flow_invocations:
                        self._low_flow_invocations[sid] = []


                    backward_context = traj.to_backward_text_per_turn(t_idx)

                    self._low_flow_invocations[sid].append({

                        "instruction": turn.instruction or "",
                        "observation": turn.observation[:200],
                        "reward": traj.reward,

                        "backward_context": backward_context,
                        "action_text": turn.supervisor_output,
                        "forward_logprob": turn.forward_logprob,
                        "backward_logprob": turn.backward_logprob,
                        "step_importance": turn.step_importance,
                        "traj_id": traj.traj_id,
                    })

                    if len(self._low_flow_invocations[sid]) > 20:
                        self._low_flow_invocations[sid] = (
                            self._low_flow_invocations[sid][-20:]
                        )

    def _refine_low_flow_skills(
        self,
        skill_marginal_flows: Dict[str, float],
        step: int,
        max_refine: int = 2,
    ) -> int:

        refined = 0


        refine_threshold = self.delta_prune + math.log(3)
        candidates = [
            sid for sid, flow in sorted(skill_marginal_flows.items(), key=lambda x: x[1])
            if flow < refine_threshold
            and sid in self._low_flow_invocations
            and len(self._low_flow_invocations[sid]) >= 5
        ]

        for sid in candidates[:max_refine]:
            skill = self.workspace.get_by_id(sid)
            if skill is None:
                continue

            invocations = self._low_flow_invocations[sid]
            refined_skill = self.skill_creator.refine_by_counterfactual(
                skill, invocations, creation_step=step
            )

            if refined_skill is not None:

                valid, reason = refined_skill.validate()
                if not valid:
                    logger.warning(
                        f"[Evolution] Refined skill {sid} failed validation: {reason} — skipping"
                    )
                    continue

                self.workspace.add(refined_skill)
                refined += 1

                self._low_flow_invocations[sid] = []
                logger.info(f"[Evolution] Refined skill {sid}: {skill.name}")

        return refined

    def _distill_high_flow_skills(
        self,
        high_flow_trajs: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
        step: int,
        max_distill: int = 2,
    ) -> int:

        distilled_count = 0

        if not skill_marginal_flows or not high_flow_trajs:
            return 0


        candidates = sorted(
            [
                (sid, flow)
                for sid, flow in skill_marginal_flows.items()
                if flow > -20.0  
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for sid, flow in candidates[:max_distill * 2]:  
            skill = self.workspace.get_by_id(sid)
            if skill is None:
                continue

            distilled_skill = self.skill_creator.distill_high_flow_skill(
                skill=skill,
                high_flow_trajectories=high_flow_trajs,
                skill_marginal_flows=skill_marginal_flows,
                creation_step=step,
            )

            if distilled_skill is not None:

                valid, reason = distilled_skill.validate()
                if not valid:
                    logger.warning(
                        f"[Evolution] Distilled skill {sid} failed validation: {reason} — skipping"
                    )
                    continue

                self.workspace.add(distilled_skill)
                distilled_count += 1
                logger.info(
                    f"[Evolution] Distilled high-flow skill {sid}: {skill.name} "
                    f"(F̂={flow:.4f})"
                )

            if distilled_count >= max_distill:
                break

        return distilled_count

    @property
    def stats(self) -> Dict:

        return {
            "total_evolutions": self._total_evolutions,
            "skills_created": self._skills_created,
            "skills_pruned": self._skills_pruned,
            "current_workspace_size": self.workspace.size,
            "recent_evolutions": self._evolution_history[-3:],
        }
