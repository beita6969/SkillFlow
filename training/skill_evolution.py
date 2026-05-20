"""
FlowSkillEvolutionManager — Flow 触发的技能进化管理器。

论文映射：
  Eq. 15 / §4.3       Φ(S^(k); {(G,Λ̃)}, {log I(t)})  —— 进化算子
  Definition F.1      Plateau trigger: 滑动窗 W 上 (Δ²_{w-W}-Δ²_w)/Δ²_{w-W} < ρ 持续 M 个窗口
  Definition F.2      三类划分:
                        𝒟⁻  prune  : n^-(s) ≥ K^-（持续多 phase Λ̃<0）
                        ℛ   retain : G(s) ≥ Φ^G_thr ∧ Λ_1-G ≤ Φ^J_thr
                        𝒰   refine : 其余（高 Jensen gap 或低 G）
  Definition F.4      Trigger steps：高 |log I(t)| ∧ 不被现存 skill 覆盖
  Definition F.5      Skill Creator Ψ: 从 (q,τ⁺,τ⁻,t) 合成 atomic tip
  Lemma F.6 (Φ 保持 atomic composability) → trainer 内部不变量
"""

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


# ──────────────────────────────────────────────────────
# Plateau detector — 论文 Definition F.1
# ──────────────────────────────────────────────────────

class PlateauDetector:
    """
    平台触发器（论文 Definition F.1）：

        plateau ⟺ (Δ̄²_{w-W} - Δ̄²_w) / Δ̄²_{w-W} < ρ
                  连续 M 个 non-overlapping window 满足

    其中 Δ̄²_w = 滑动窗 W 内 batch 平均 squared TTB residual。

    Args:
        window_size: W
        rho: 相对下降容忍度（典型 0.05 = 5%）
        m_consecutive: M 连续窗口数（典型 2-3）
    """

    def __init__(self, window_size: int = 20, rho: float = 0.05, m_consecutive: int = 2):
        self.W = max(int(window_size), 2)
        self.rho = float(rho)
        self.M = max(int(m_consecutive), 1)
        self._delta_sq: List[float] = []   # 每个 step 的 batch-mean Δ²
        self._consec_count = 0
        self._last_trigger_step = -1

    def update(self, step: int, batch_delta_sq: float) -> None:
        """每个 training step 调用一次，传入该 step 的 batch-mean Δ²。"""
        self._delta_sq.append(float(batch_delta_sq))

    def should_trigger(self, step: int) -> bool:
        """
        判断当前 step 是否触发 phase 边界。
        消费内部状态 — 触发后复位 consecutive counter，避免连发。
        """
        if len(self._delta_sq) < 2 * self.W:
            return False
        # 当前窗口与前一窗口的均值
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


# ──────────────────────────────────────────────────────
# 𝒟⁻ / ℛ / 𝒰 三类划分 — 论文 Definition F.2
# ──────────────────────────────────────────────────────

@dataclass
class CurationClassification:
    """每个 phase 边界的 𝒟⁻/ℛ/𝒰 三类划分结果。"""
    prune: List[str] = field(default_factory=list)     # 𝒟⁻
    retain: List[str] = field(default_factory=list)    # ℛ
    refine: List[str] = field(default_factory=list)    # 𝒰
    cgf: Dict[str, Dict[str, float]] = field(default_factory=dict)  # per-skill (G, Λ_1, Λ̃, jensen_gap)


def partition_skills_DRU(
    trajectories: List[Trajectory],
    skill_ids: List[str],
    n_minus_counter: Dict[str, int],
    G_threshold: float = 0.0,
    J_threshold: float = 1.0,
    K_minus: int = 2,
) -> CurationClassification:
    """
    论文 Definition F.2 三类划分：

        𝒟⁻_k = { s : n^-(s) ≥ K^- }                     prune
        ℛ_k  = { s ∉ 𝒟⁻ : G(s) ≥ Φ^G_thr ∧
                            Λ^(s)_1 - G(s) ≤ Φ^J_thr }  retain
        𝒰_k  = S^(k) ∖ (𝒟⁻ ∪ ℛ)                         refine

    Args:
        trajectories: 当前 phase 的所有轨迹
        skill_ids: 当前 library 的所有 skill id
        n_minus_counter: 每个 skill 历史上 Λ̃<0 的连续 phase 计数（持久状态）
        G_threshold: Φ^G_thr — mean log-flow 阈值
        J_threshold: Φ^J_thr — Jensen gap 阈值
        K_minus: K^- — 进入 𝒟⁻ 所需的连续负 share 次数

    Returns:
        CurationClassification 含三类 skill_id 列表 + per-skill CGF 摘要
    """
    cgf = compute_skill_cgf_summary(trajectories, skill_ids)
    result = CurationClassification(cgf=cgf)
    for sid in skill_ids:
        # 更新 n^-(s)
        lam_t = cgf[sid]["lambda_tilde"]
        if lam_t < 0:
            n_minus_counter[sid] = n_minus_counter.get(sid, 0) + 1
        else:
            n_minus_counter[sid] = 0  # 重置：连续要求

        # 1. 𝒟⁻: n^-(s) ≥ K^-
        if n_minus_counter[sid] >= K_minus:
            result.prune.append(sid)
            continue

        G_s = cgf[sid]["G"]
        jensen = cgf[sid]["jensen_gap"]
        # 2. ℛ: 高 G + 小 Jensen gap
        if G_s >= G_threshold and jensen <= J_threshold:
            result.retain.append(sid)
        else:
            # 3. 𝒰: 其余
            result.refine.append(sid)
    return result


# ──────────────────────────────────────────────────────
# α_t 映射 {skill, act, accept} — 论文主文 Eq. 1
# ──────────────────────────────────────────────────────

_ACCEPT_ACTIONS = frozenset({"accept", "answer", "submit", "finish", "final", "done"})
_SKILL_ACTIONS = frozenset({"skill_invoke", "skill"})


def action_type_to_alpha(action_type: str) -> str:
    """
    将 trainer 中粒度更细的 tool_name 映射到论文主文的三类 α_t ∈ {skill, act, accept}：

        skill   ← skill_invoke（调用一个 atomic tip）
        accept  ← 终止动作 (answer / accept / submit / finish ...)
        act     ← 其他所有 tool 调用（search / think / edit / click / ...）

    论文 Definition 1：a_t = (α_t, o_t)，α_t 决定动作类型，o_t 为参数。
    """
    a = (action_type or "").lower()
    if a in _SKILL_ACTIONS:
        return "skill"
    if a in _ACCEPT_ACTIONS:
        return "accept"
    return "act"


class TaskTypeAccuracyTracker:
    """
    Per-task-type 准确率追踪（参考 OpenClaw-RL 的 track_result）。

    追踪每个 task_type 的 correct/total 统计，
    识别 struggling types（acc < threshold 且 count >= min_count）。
    """

    def __init__(self, threshold: float = 0.5, min_count: int = 10):
        self.threshold = threshold
        self.min_count = min_count
        self._counts: Dict[str, int] = defaultdict(int)
        self._correct: Dict[str, int] = defaultdict(int)

    def track_result(self, task_type: str, correct: bool) -> None:
        """记录一个 episode 的结果"""
        self._counts[task_type] += 1
        if correct:
            self._correct[task_type] += 1

    def get_accuracy(self, task_type: str) -> float:
        """获取某 task_type 的准确率"""
        total = self._counts.get(task_type, 0)
        if total == 0:
            return 0.5  # 无数据时返回中性值
        return self._correct.get(task_type, 0) / total

    def get_struggling_types(self) -> List[str]:
        """返回 acc < threshold 且样本量 >= min_count 的 task_types"""
        struggling = []
        for tt, count in self._counts.items():
            if count >= self.min_count:
                acc = self._correct.get(tt, 0) / count
                if acc < self.threshold:
                    struggling.append(tt)
        return struggling

    def all_accuracies(self) -> Dict[str, float]:
        """返回所有 task_type 的准确率"""
        return {
            tt: self._correct.get(tt, 0) / max(count, 1)
            for tt, count in self._counts.items()
        }

    def summary(self) -> str:
        """日志友好的摘要"""
        accs = self.all_accuracies()
        parts = [f"{tt}={acc:.2f}({self._counts[tt]})" for tt, acc in sorted(accs.items())]
        return ", ".join(parts) if parts else "no data"


class SkillAcceptanceGate:
    """
    Skill acceptance gate — validates candidate skills before committing to workspace.

    论文 §4.4 拓展：防止 curator 产生的低质量 skill 污染库。

    两种模式：
    - Passthrough (n_eval_episodes=0 or run_episode_fn=None): 仅做结构性 sanity check，
      admit by default。
    - Eval mode: 调用 run_episode_fn 跑 n_eval_episodes 轮，对比 baseline 分布，
      要求 improvement > max(min_improvement, 2*baseline_se) 才接受。

    baseline_trajectories: 最近的 low-flow + high-flow 轨迹, reward 作为无候选 skill 的基线。
    """

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
        """
        Returns (accepted, stats)。

        stats 包含: candidate_mean, baseline_mean, improvement, threshold, mode。
        """
        # ── baseline 统计 ──
        if baseline_trajectories:
            rewards = [float(getattr(t, 'r_tilde', 0.0)) for t in baseline_trajectories]
            baseline_mean = sum(rewards) / len(rewards)
            baseline_var = sum((r - baseline_mean) ** 2 for r in rewards) / max(len(rewards), 1)
            baseline_se = (baseline_var / max(len(rewards), 1)) ** 0.5
        else:
            baseline_mean, baseline_se = 0.5, 0.1

        # ── Passthrough 模式（默认禁用实际 eval）──
        if self.n_eval_episodes <= 0 or self.run_episode_fn is None:
            # 轻量 sanity: skill body 不能为空、description 不能为空
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

        # ── Eval 模式：跑候选 skill ──
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
    """
    管理技能库的 flow-guided 进化生命周期。

    与 GFlowNetTrainer 协作：
      - Trainer 在每个 phase_steps 调用 maybe_evolve()
      - EvolutionManager 决定是否触发，并执行进化
      - 进化后通知 Trainer 进行 Z_θ re-warm
    """

    def __init__(
        self,
        workspace: SkillWorkspace,
        skill_creator: SkillCreator,
        delta_h: float = -1.0,           # <0 表示使用自适应阈值 1-ln2/ln|S|；>0 为手动覆盖
        delta_prune: float = -10.0,      # log F̂(s) 剪枝阈值（log 空间）
        min_usage_before_prune: int = 20,
        max_skills_total: int = 60,
        evolution_phase_steps: int = 20,
        min_trajs_for_evolution: int = 16,
        experience_store=None,
        beta: float = 1.0,              # GFlowNet β — R̃^β 作为 flow proxy（Eq.3 收敛时 F(τ)=R̃^β）
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

        # 历史记录
        self._evolution_history: List[Dict] = []
        self._total_evolutions = 0
        self._skills_created = 0
        self._skills_pruned = 0

        # 低 flow invocation 记录（供反事实细化用）
        self._low_flow_invocations: Dict[str, List[Dict]] = {}

    # ──────────────────────────────────────────────
    # v2: Per-type flow-driven tip evolution
    # ──────────────────────────────────────────────

    def evolve_for_type(
        self,
        task_type: str,
        observations: List[Dict],
        failed_trajectories: List[Trajectory],
        step: int,
        # ── v3: flow 信号（legacy, fallback 路径）──
        high_flow_trajs: Optional[List[Trajectory]] = None,
        critical_steps: Optional[List[Dict]] = None,
        dag_comparisons: Optional[List[Dict]] = None,
        # ── v5: 3+2+1 结构化诊断信号 ──
        bottleneck_diagnoses: Optional[List] = None,
        counterfactual_pairs: Optional[List] = None,
        acceptance_gate: Optional["SkillAcceptanceGate"] = None,
        curation: Optional[CurationClassification] = None,
    ) -> List[SkillEntry]:
        """论文 §4.4：flow/CGF-guided skill evolution (v5 — 3+2+1 融合版)。

        两阶段：
        Phase 1: CGF/Λ̃/Jensen gap 给出 D⁻/R/U 类别，决定 prune/protect/refine 范围
        Phase 2: Skill Creator 只在 flow 允许的范围内改写/新增 tip

        当 bottleneck_diagnoses 存在时，走结构化诊断路径（DIAGNOSE_AND_CURATE_PROMPT）；
        否则 fallback 到传统完整轨迹 prompt。
        """
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

        # Skill Creator proposes edits/additions from evidence; CGF D/R/U guard
        # below decides which existing skills may actually be pruned/refined.
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
            # 论文 Definition F.2: D⁻ 由持久负 centered share 决定，直接 prune；
            # R 由高 G + 小 Jensen gap 保护；U 才允许被 Skill Creator 改写。
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

        # Execute workspace mutations
        for sid in result["deleted"]:
            self.workspace.remove(sid)

        updated_entries: List[SkillEntry] = []
        for old_id, new_entry in result["updated"]:
            new_entry.meta.creation_step = step
            self.workspace.add(new_entry)  # overwrites by same ID
            updated_entries.append(new_entry)

        added = []
        rejected_by_gate: List[SkillEntry] = []
        for tip in result["added"]:
            tip.meta.creation_step = step
            # Acceptance gate: evaluate candidate before committing
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

    # ──────────────────────────────────────────────
    # Legacy: maybe_evolve (保留向后兼容，不再从训练主循环调用)
    # ──────────────────────────────────────────────

    def maybe_evolve(
        self,
        trajectories: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
        step: int,
    ) -> Tuple[bool, List[SkillEntry]]:
        """
        [Legacy] 检查是否应触发进化，如果是则执行。
        v2 中由 gflownet_trainer._try_evolve() 直接调用 evolve_for_type()，
        不再经过此方法。保留以供调试和向后兼容。
        """
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
        """
        自适应进化阈值：半利用率原则。

        δ_H = 1 - ln(2) / ln(|S|)

        当有效技能数 < |S|/2 时触发进化。
        阈值随技能库增长自动升高（大库不应有太多死技能）。
        如果 self.delta_h > 0，使用手动覆盖值。
        """
        if self.delta_h > 0:
            return self.delta_h
        s_size = max(self.workspace.size, 2)
        return 1.0 - math.log(2) / math.log(s_size)

    def _compute_skill_flow_entropy(
        self, skill_marginal_flows: Dict[str, float],
    ) -> Tuple[float, float, float]:
        """
        计算技能流量熵 H_skill（基于论文 Eq.7 的 F̂(s)）。

        用 F̂(s) 的 softmax 分布计算熵，直接衡量技能库的流量集中度。
        相比轨迹级 flow entropy（Eq.10），技能级熵不受 LLM per-token
        归一化问题影响，因为 F̂(s) 聚合的是边流量而非全轨迹 token 序列。

        Returns:
            (h_skill, h_max, h_ratio)
        """
        if not skill_marginal_flows:
            return 0.0, 0.0, 0.0

        log_flows = list(skill_marginal_flows.values())
        n_skills = len(log_flows)
        h_max = math.log(n_skills) if n_skills > 1 else 1.0

        # softmax in log space
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
        """
        判断是否应触发进化 — 基于 GFlowNet TTB loss 停滞检测。

        GFlowNet 的训练目标是最小化 TTB balance error（论文 Eq.6）。
        当 loss 在一个 phase 内不再改善，说明当前 skill 配置下的 DAG 结构
        已无法让 flow distribution 进一步逼近 reward-proportional 分布
        → 需要改变 DAG（进化 skills）让 TTB 优化有新的自由度。

        触发条件（任一满足 + 容量允许）：
          (1) TTB loss 停滞 — 后半 phase 的平均 loss 不低于前半
          (2) 低利用率 — 有效技能数 < 总数 40%（F̂(s) 全低的兜底检测）
        """
        s_size = self.workspace.size

        if s_size >= self.max_skills_total:
            logger.info(f"[Evolution] Skill library full ({s_size}/{self.max_skills_total}), "
                        f"trying to prune before evolution")
            self._try_prune()
            s_size = self.workspace.size

        capacity_ok = s_size < self.max_skills_total

        # ── 条件 1：TTB loss 停滞检测 ──
        # 将 phase 轨迹按时间分前后半，比较平均 TTB balance error
        loss_stagnation = False
        if phase_trajectories and len(phase_trajectories) >= 24:
            mid = len(phase_trajectories) // 2
            first_half = phase_trajectories[:mid]
            second_half = phase_trajectories[mid:]
            # 用 |log F(τ) - β·log R̃| 作为 per-trajectory balance error
            def _balance_error(traj):
                log_F = self._trajectory_log_flow(traj)
                log_R_beta = self.beta * math.log(max(traj.r_tilde, 1e-10))
                return abs(log_F - log_R_beta)
            avg_first = sum(_balance_error(t) for t in first_half) / len(first_half)
            avg_second = sum(_balance_error(t) for t in second_half) / len(second_half)
            # 后半 balance error 没比前半低 → 停滞
            loss_stagnation = avg_second >= avg_first * 0.95
            logger.info(
                f"[Evolution] TTB balance error: first_half={avg_first:.1f}, "
                f"second_half={avg_second:.1f}, stagnation={loss_stagnation}"
            )

        # ── 条件 2：低利用率兜底 ──
        n_active = sum(1 for f in skill_marginal_flows.values() if f > -15.0)
        n_total = max(len(skill_marginal_flows), 1)
        utilization_trigger = n_active < n_total * 0.4

        # ── 条件 3：per-task-type 准确率低于阈值（OpenClaw-RL 风格） ──
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

        # ── 条件 4：flow entropy（论文 §3.4 H_skill ratio） ──
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
        """
        执行进化流程。

        Returns:
            新添加到 workspace 的技能列表
        """
        self._total_evolutions += 1

        # 1. 更新 workspace 中所有技能的 F̂(s)
        self._update_flow_scores(skill_marginal_flows)

        # 2. 剪枝低效技能（传入 step 保护最近使用的技能）
        pruned = self._try_prune(current_step=step)
        if pruned:
            logger.info(f"[Evolution] Pruned {len(pruned)} skills: {pruned}")
            self._skills_pruned += len(pruned)

        # 3. 分割高/低 flow 轨迹（论文 §3.4：用实际轨迹 flow F(τ) 做 quartile）
        # F(τ) = Z_θ · ∏_t π_θ(a_t)，直接可算。训练中 F(τ) ≠ R̃^β。
        # 论文："moderate reward but high flow = reliable; high reward but low flow = lucky"
        sorted_trajs = sorted(trajectories, key=lambda t: self._trajectory_log_flow(t))
        n = len(sorted_trajs)
        low_flow_trajs = sorted_trajs[:max(1, n // 4)]
        high_flow_trajs = sorted_trajs[max(0, n - n // 4):]

        # 3b. Cross-Trajectory Critique（XSkill 风格）：提取 action-level experiences
        # 按 task_type 均衡取 F(τ) top/bottom quartile，确保各 task_type 公平分析
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
                # 计算 reward_signal: 高 flow 轨迹的平均 reward
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

        # 4. backward_pattern_mine：提取关键步骤
        mined_patterns = []
        for traj in high_flow_trajs[:8]:
            patterns = self.skill_creator.backward_pattern_mine(traj, top_k=2)
            mined_patterns.extend(patterns)

        # 4b. 成功蒸馏（论文 §3.4 Distill 操作）：对高 flow 技能强化 plan
        distilled_count = self._distill_high_flow_skills(
            high_flow_trajs=high_flow_trajs,
            skill_marginal_flows=skill_marginal_flows,
            step=step,
        )
        if distilled_count:
            logger.info(f"[Evolution] Distilled {distilled_count} high-flow skills")

        # 5. 记录低 flow invocation（供反事实细化）
        self._record_low_flow_invocations(low_flow_trajs, skill_marginal_flows)

        # 6. 反事实细化现有低效技能
        refined_count = self._refine_low_flow_skills(skill_marginal_flows, step)
        if refined_count:
            logger.info(f"[Evolution] Refined {refined_count} low-flow skills")

        # 7. 生成新技能 — R2 模式：每次从当前轨迹新鲜生成 living document
        #
        # R2 实验证明：新鲜生成比增量 merge 效果好（+28% vs +11%）
        # 原因：RL 训练中 policy 每步变化，旧策略与新 policy 失配
        # 每次进化：4 个 SOPs → merge 为 1 个新鲜文档 → 匹配当前 policy 能力
        #
        # TTB 控制触发频率（不是每步），min_trajs=48 保证样本充分

        # XSkill Living Document 模式：
        # 1. 从轨迹生成 SOPs
        # 2. Merge 到 per-type 单一文档（不是独立 skill files）
        # 3. 超 800 字自动 Refine
        # 这样模型 prompt 中只有 1 个策略文档，不需要选择

        # v2: Atomic tips（ADD not MERGE）
        # 不再生成 Living Document，改为提取独立的原子 tip
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

            # v2: 提取原子化 tips（每条 < 60 words，独立存储）
            new_tips = self.skill_creator.extract_atomic_tips(
                high_trajs=high_t,
                low_trajs=low_t,
                task_type=tt,
                max_tips=2,  # 每次进化最多 2 条新 tip per type
            )

            # ADD: 每条 tip 独立添加到 workspace（不 merge）
            for tip in new_tips:
                tip.meta.creation_step = getattr(self, '_current_step', 0)
                self.workspace.add(tip)
                self._skills_created += 1
                logger.info(
                    f"[Evolution] Added tip for {tt}: "
                    f"\"{tip.plan[:80]}\" ({len(tip.plan.split())} words)"
                )

        # v2 模式下 atomic tips 已在 evolve_for_type() 中处理
        # 以下为 legacy 代码（Living Doc + flow_guided_evolution），默认禁用
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

        # 9. 记录进化历史
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
        """
        XSkill Living Document Consolidation — 用新轨迹证据更新 per-type living documents。

        采用 XSkill 的 MERGE + REFINE 流程：
        1. 从高分轨迹生成新的 raw skill（GENERATE_SKILL_PROMPT）
        2. Merge 进已有的 per-type living document（MERGE_SKILL_PROMPT）
        3. 如果文档过长，Refine（SKILL_REFINE_PROMPT）

        与旧实现的区别：
        - 旧：直接用 M_exec 改写 individual skill.plan → 容易过拟合
        - 新：生成 raw skill → merge 进 living document → refine → 自然泛化
        """
        if not high_flow_trajs:
            return 0

        # 按 task_type 分组高分轨迹
        from src.skills.skill_prompts import format_trajectory_for_skill_generation, clean_skill_output
        from src.skills.format import TaskTypeSkillDocument

        type_trajs = defaultdict(list)
        for traj in high_flow_trajs:
            type_trajs[traj.task_type].append(traj)

        # 低分轨迹（用于对比学习）
        sorted_trajs = sorted(trajectories, key=lambda t: t.reward)
        low_trajs_by_type = defaultdict(list)
        for traj in sorted_trajs[:len(sorted_trajs)//4]:
            low_trajs_by_type[traj.task_type].append(traj)

        consolidated = 0
        # 优先处理有高方差的 task_types
        all_skills = self.workspace.get_all()
        all_skill_ids = [s.meta.skill_id for s in all_skills]
        skill_variances = compute_skill_variances(trajectories, all_skill_ids)

        # 按 type 的平均 variance 排序
        type_variance = {}
        for tt, tt_trajs in type_trajs.items():
            type_skills = [s for s in all_skills if tt in (s.meta.task_types or [])]
            if type_skills:
                avg_var = sum(skill_variances.get(s.meta.skill_id, 0.0) for s in type_skills) / len(type_skills)
                type_variance[tt] = avg_var
            else:
                type_variance[tt] = 1.0  # 无 skill 的 type 优先

        sorted_types = sorted(type_variance.keys(), key=lambda t: type_variance[t], reverse=True)

        for tt in sorted_types[:max_consolidate]:
            ht = type_trajs.get(tt, [])[:3]
            lt = low_trajs_by_type.get(tt, [])[:2]
            if not ht:
                continue

            # 从高/低分轨迹对生成新的 raw skill；不向技能生成器提供 gold answer。
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

            # 获取已有 living document
            existing_doc = self.workspace.get_type_document(tt)
            existing_content = existing_doc.consolidated_strategy if existing_doc else ""

            # Merge
            merged = self.skill_creator.merge_into_living_document(
                task_type=tt,
                existing_document=existing_content,
                new_skill_contents=[new_raw],
            )

            if not merged:
                continue

            # Refine（如果超过 1000 词）
            refined = self.skill_creator.refine_living_document(
                task_type=tt,
                document=merged,
                word_threshold=1000,
            )

            # 更新 workspace 中的 type document
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
        """
        更新 per-type Living Documents 的辅助信息（proven patterns + accuracy）。

        注意：实际的 living document 内容（consolidated_strategy）已在
        _consolidate_all_skills 中通过 MERGE + REFINE 更新。
        这里只补充 proven_patterns 和 accuracy 元数据。
        """
        from src.skills.format import TaskTypeSkillDocument

        affected_types = set(t.task_type for t in trajectories)

        for tt in affected_types:
            type_skills = self.workspace.get_skills_by_task_type(tt)

            # 收集 proven patterns（从 experience store）
            proven_patterns = []
            if self.experience_store:
                exps = self.experience_store.retrieve(query=tt, task_type=tt, top_k=5)
                for exp in exps:
                    proven_patterns.append(f"When: {exp.condition} → Do: {exp.action}")

            # 获取准确率
            accuracy = 0.5
            sample_count = 0
            if self.accuracy_tracker:
                accuracy = self.accuracy_tracker.get_accuracy(tt)
                sample_count = self.accuracy_tracker._counts.get(tt, 0)

            # 保留已有的 living document 内容（如果有）
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
        """
        计算轨迹 flow：log F(τ) = log Z_θ(q) + Σ_t log π̃_θ(a_t | H_{t-1})。

        这是 policy 分配给此轨迹的概率质量（乘以 Z），直接可算。
        训练中 F(τ) ≠ R̃^β — 差值正是 TTB loss 在优化的对象。
        论文用 F(τ) 做 quartile 来区分 "reliable"（高 F）和 "lucky"（高 R 低 F）。
        """
        return traj.log_z + sum(edge_logprob_tilde(t, "forward") for t in traj.turns)

    def _update_flow_scores(self, skill_marginal_flows: Dict[str, float]) -> None:
        """
        EMA 更新 workspace 中所有技能的 F̂(s)。

        注意：只更新本 phase 中有调用（flow > -20）的技能。
        未使用的技能保持原 score，避免被 -20 拖低（trainer 已每步更新有调用的技能）。
        """
        for sid, flow in skill_marginal_flows.items():
            if flow > -20.0:  # 只更新有实际调用的技能
                self.workspace.update_flow_score(sid, flow, alpha=0.3)

    def _try_prune(self, current_step: int = -1) -> List[str]:
        """
        XSkill-style consolidation pruning：当 skill 库超过 max_skills_total 时，
        找最相似的两个 skill（embedding sim > 0.70）→ merge 为一个。不删除，只合并。

        对比之前的 age-based pruning（误删了有效 skill），这里通过 merge 保留知识。
        """
        if self.workspace.size <= self.max_skills_total:
            return []

        merged_ids = []
        # 最多 merge 到不超限为止
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

            # 用 M_exec merge 两个 skill 为一个
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
                    # 保留较高 flow_score 的 ID
                    keep_id = sid_a if (skill_a.meta.flow_score >= skill_b.meta.flow_score) else sid_b
                    remove_id = sid_b if keep_id == sid_a else sid_a
                    merged_skill.meta.skill_id = keep_id
                    merged_skill.meta.flow_score = max(skill_a.meta.flow_score, skill_b.meta.flow_score)
                    merged_skill.meta.creation_step = current_step
                    merged_skill.meta.source = "consolidation_merge"
                    # 合并 task_types
                    types_a = set(skill_a.meta.task_types) if skill_a.meta.task_types else set()
                    types_b = set(skill_b.meta.task_types) if skill_b.meta.task_types else set()
                    merged_skill.meta.task_types = list(types_a | types_b)

                    self.workspace.add(merged_skill)  # overwrite keep_id
                    self.workspace.remove(remove_id)  # remove the other
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
        """
        找 skill 库中 embedding 相似度最高的一对（sim > threshold）。

        Returns:
            (skill_id_a, skill_id_b, similarity) or None
        """
        cache = self.workspace._compute_skill_embeddings()
        if cache is None:
            return None

        skill_ids = cache["skill_ids"]
        embeddings = cache["embeddings"]  # (N, 3, D) — multi-vector
        n = len(skill_ids)
        if n < 2:
            return None

        # 用各 skill 的 3 个向量的平均作为代表向量
        import numpy as np
        avg_embs = embeddings.mean(axis=1)  # (N, D)
        # 归一化
        norms = np.linalg.norm(avg_embs, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        avg_embs = avg_embs / norms

        # 计算所有 pairwise 余弦相似度
        sim_matrix = avg_embs @ avg_embs.T  # (N, N)
        # 屏蔽对角线
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
        """
        记录低 flow 轨迹中的技能调用（供 refine_by_counterfactual P_φ 反事实分析）。

        P1 修复：除 instruction/observation/reward 外，额外存储：
          - backward_context: P_φ 条件化的后继状态 H_{t-1} ⊕ o_t（由 to_backward_text_per_turn 构建）
          - action_text: Supervisor 的完整输出 a_t（包含 think 部分，P_φ 评分前会自动拆分）
          - forward_logprob: log π_θ(a_t|H_{t-1})（训练时已填充）
          - backward_logprob: log P_φ(a_t|H_t)（训练时已填充）
          - step_importance: I(t) = exp(fwd - bwd)（flow 计算后已填充）
          - traj_id: 轨迹 ID（用于 grounding 证据追溯）

        这些字段使 refine_by_counterfactual 能够执行论文 §3.4 所描述的完整
        P_φ(a'|H_t) 替代动作评分流程。
        """
        for traj in low_flow_trajs:
            for t_idx, turn in enumerate(traj.turns):
                if turn.action_type == "skill_invoke" and turn.skill_id:
                    sid = turn.skill_id
                    if sid not in self._low_flow_invocations:
                        self._low_flow_invocations[sid] = []

                    # to_backward_text_per_turn(t_idx) 构建 H_{t-1} ⊕ o_t
                    backward_context = traj.to_backward_text_per_turn(t_idx)

                    self._low_flow_invocations[sid].append({
                        # 原有字段
                        "instruction": turn.instruction or "",
                        "observation": turn.observation[:200],
                        "reward": traj.reward,
                        # P1 新增字段（供 P_φ 反事实分析）
                        "backward_context": backward_context,
                        "action_text": turn.supervisor_output,
                        "forward_logprob": turn.forward_logprob,
                        "backward_logprob": turn.backward_logprob,
                        "step_importance": turn.step_importance,
                        "traj_id": traj.traj_id,
                    })
                    # 最多保留 20 条（按时间顺序，保留最新）
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
        """对 F̂(s) 低的技能进行反事实细化"""
        refined = 0

        # 找最差的技能（有足够调用记录的）
        # log 空间：delta_prune + log(3) ≈ delta_prune + 1.1，即 F̂(s) < 3 × δ_prune
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
                # 质量门控：细化后的技能同样需要通过结构校验
                valid, reason = refined_skill.validate()
                if not valid:
                    logger.warning(
                        f"[Evolution] Refined skill {sid} failed validation: {reason} — skipping"
                    )
                    continue
                # 覆盖原技能
                self.workspace.add(refined_skill)
                refined += 1
                # 清空记录
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
        """
        对 F̂(s) 最高的技能执行成功蒸馏（论文 §3.4 Distill 操作）。

        s'' = Distill(s, T_high_flow^(s))

        从高 flow 轨迹中找出被高频成功调用的技能，
        提取其调用模式并蒸馏进 plan，强化已知有效策略。

        Args:
            high_flow_trajs: 当前 phase 中 flow 最高的 25% 轨迹
            skill_marginal_flows: 各技能的 F̂(s) 当前值
            step: 当前训练步（用于 creation_step 标记）
            max_distill: 本次进化最多蒸馏几个技能（避免过度修改）

        Returns:
            实际执行了蒸馏的技能数量
        """
        distilled_count = 0

        if not skill_marginal_flows or not high_flow_trajs:
            return 0

        # 找出 F̂(s) 最高且在 high_flow_trajs 中被调用过的技能
        # 按 F̂(s) 降序，优先蒸馏最有价值的技能
        candidates = sorted(
            [
                (sid, flow)
                for sid, flow in skill_marginal_flows.items()
                if flow > -20.0  # log 空间：-20 = 未使用
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        for sid, flow in candidates[:max_distill * 2]:  # 候选池 2×，确保找到足够多可蒸馏的
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
                # 质量门控：蒸馏后的技能同样需要通过结构校验
                valid, reason = distilled_skill.validate()
                if not valid:
                    logger.warning(
                        f"[Evolution] Distilled skill {sid} failed validation: {reason} — skipping"
                    )
                    continue
                # 覆盖 workspace 中的原技能（保持相同 skill_id）
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
        """返回进化统计"""
        return {
            "total_evolutions": self._total_evolutions,
            "skills_created": self._skills_created,
            "skills_pruned": self._skills_pruned,
            "current_workspace_size": self.workspace.size,
            "recent_evolutions": self._evolution_history[-3:],
        }
