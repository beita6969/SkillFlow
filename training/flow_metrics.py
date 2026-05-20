"""
SkillFlow Flow 度量计算。

论文映射（main + appendix）：
  Eq. 12 / Def C.1     I(t) = π_θ(a_t|r_t,H_{t-1}) / P_φ(a_t|H_{t-1}⊕o_t^exec)
  Eq. 14 / Def E.3     log F(s_t) = log Z_θ(q) + Σ_{t'≤t} log I(t')
  Eq.  7 / Def E.3     F̂(s) = (1/|B_s|) Σ_τ Σ_{t: a_t invokes s} F(s_t)        ← STATE flow
  Lemma F.2           G(s)  = ∂Λ^(s)_λ/∂λ|_{λ=0} = visit-mean of log F(s_t)
  Lemma F.3           Λ^(s)_1 = log F̂(s) - log Z_θ(q)
  Prop  F.5           Jensen gap Λ^(s)_1 - G(s) = ½ Var_{V_s}[log F(s_t)] + Σ_{k≥3} κ_k/k!
  Remark F.7          Λ̃(s) = Λ^(s)_1 - E_{s'}[Λ^(s')_1]                         (centered share)
  Def F.4             Trigger steps: {t : log I(t) ≥ ζ_trig ∧ t ∉ cov(R ∪ U')}

PAPER ALIGNMENT NOTE
─────────────────────
`Turn.forward_logprob` / `Turn.backward_logprob` are stored as token-sums because
that is the natural output of an autoregressive LM scorer.  All paper-facing flow
quantities below convert them to the paper's per-token-tempered edge log-prob,
log p̃(a_t) = (1/K_t) Σ_j log p(tok_j), before forming I(t), state flows, CGF
statistics, and TTB monitoring.  Virtual auto-injected skill markers have K_t=0
and therefore contribute zero policy log-prob; real policy-selected skill_invoke
tool calls have their normal JSON token count and are scored like other actions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from training.trajectory import Trajectory


def edge_logprob_tilde(turn, which: str = "forward") -> float:
    """
    Paper Definition B.2 per-token-tempered edge log-prob.

    Stored values are token-sums.  For a real action with K_t target tokens, return
    (1/K_t) Σ_j log p(tok_j).  For virtual bookkeeping turns (K_t=0), return 0.
    """
    attr = "forward_logprob" if which == "forward" else "backward_logprob"
    raw = float(getattr(turn, attr, 0.0) or 0.0)
    k = int(getattr(turn, "action_token_count", 0) or 0)
    if k <= 0:
        return 0.0
    return raw / max(k, 1)


def edge_log_i(turn) -> float:
    """log I(t) using the paper's per-token-tempered forward/backward edges."""
    return edge_logprob_tilde(turn, "forward") - edge_logprob_tilde(turn, "backward")


def effective_paper_steps(traj: Trajectory) -> int:
    """
    Number of scored policy edges for TTB normalisation.

    Auto-injected skill markers are virtual and have K_t=0, so they are excluded.
    If a trajectory somehow has no scored edge, fall back to 1 for numerical safety.
    """
    return max(
        sum(1 for t in traj.turns if int(getattr(t, "action_token_count", 0) or 0) > 0),
        1,
    )


# ──────────────────────────────────────────────────────
# 单轨迹 flow 计算
# ──────────────────────────────────────────────────────


def compute_state_flows(
    traj: Trajectory,
    log_z: float,
) -> List[float]:
    """
    计算轨迹中每个状态 s_t 的 log-space 状态流量。

    公式：log F(s_t) = log Z(q) + Σ_{k=1}^{t} (log π̃[k] - log P̃[k])

    注：s_0 是初始状态（question），log F(s_0) = log Z(q)。
    """
    log_flow = log_z
    state_flows: List[float] = [log_flow]  # s_0

    for turn in traj.turns:
        log_flow = log_flow + edge_log_i(turn)
        state_flows.append(log_flow)

    return state_flows  # 长度 = n_turns + 1


def compute_edge_flows(
    state_flows: List[float],
    forward_logprobs: List[float],
) -> List[float]:
    """
    计算每条边 (s_{t-1} → s_t) 的 log-space 边流量。

    公式：log F(s_{t-1} → s_t) = log F(s_{t-1}) + log π̃_θ(a_t | H_{t-1})
    """
    edge_flows = []
    for t, fwd in enumerate(forward_logprobs):
        edge_flows.append(state_flows[t] + fwd)
    return edge_flows


def compute_step_importance(
    forward_logprob: float,
    backward_logprob: float,
    action_token_count: int = 0,
) -> float:
    """
    步骤重要性（论文 Eq. 12 / Definition C.1）：

        I(t) = π_θ(a_t | r_t, H_{t-1}) / P_φ(a_t | H_{t-1} ⊕ o_t^exec)
             = exp(log π_θ(a_t|...) - log P_φ(a_t|...))

    分子：forward policy 在 H_{t-1} 上对 a_t 的概率。
    分母：hindsight backward policy 在 *增广状态* H_{t-1} ⊕ o_t^exec 上对 a_t 的概率。
    信息不对称：P_φ 多看到 o_t^exec → I(t) 编码"事后才看出来"的 credit。

    物理意义（Lemma I.1 / 论文 §4.2）：
      - I(t) >> 1 → forward 选了一个 hindsight 来看不太可能的动作 → 高 impact 步骤
      - I(t) << 1 → forward 选的动作在 hindsight 看来本应低概率 → sub-optimal 步骤
      - I(t) ≈ 1 → 前后向一致 → routine 步骤

    用于 state flow 的 telescoping（Eq. 14）：
        log F(s_t) = log Z_θ(q) + Σ_{t'=1}^{t} log I(t')
    """
    if action_token_count and action_token_count > 0:
        diff = (forward_logprob - backward_logprob) / max(action_token_count, 1)
    else:
        diff = forward_logprob - backward_logprob
    # 数值安全：防止 exp overflow/underflow（float64 安全范围 ≈ [-709, 709]）
    if diff > 700:
        return math.exp(700)
    if diff < -700:
        return 0.0
    return math.exp(diff)


def fill_turn_flows(traj: Trajectory, log_z: float) -> None:
    """
    原地填充 trajectory 中每个 turn 的 flow 字段。
    在 batch 训练完成、forward/backward logprobs 已填充后调用。
    """
    state_flows = compute_state_flows(traj, log_z)
    fwd_logprobs = [edge_logprob_tilde(t, "forward") for t in traj.turns]
    edge_flows = compute_edge_flows(state_flows, fwd_logprobs)

    for i, turn in enumerate(traj.turns):
        turn.state_flow = state_flows[i + 1]  # s_i+1 对应第 i 步动作
        turn.edge_flow = edge_flows[i]
        turn.step_importance = compute_step_importance(
            turn.forward_logprob, turn.backward_logprob, turn.action_token_count
        )

    traj.log_z = log_z


# ──────────────────────────────────────────────────────
# Batch level 度量
# ──────────────────────────────────────────────────────


def compute_trajectory_flow(traj: Trajectory) -> float:
    """
    计算轨迹的状态流 F(τ)（基于最终状态流，含 P_φ 项）。

    定义：F(τ) = exp(log F(s_T))
    = Z(q) · ∏_t [π_θ(a_t|H_{t-1}) / P_φ(a_t|H_{t-1}⊕o_t)]

    用途：flow 监控、adjust / debugging。
    注意：不用于 Ĥ_flow 计算，熵应使用 compute_forward_trajectory_flow()。
    """
    if not traj.turns:
        return math.exp(traj.log_z)

    # 最后一个状态的 flow（fill_turn_flows 调用后可用）
    final_state_flow = traj.turns[-1].state_flow if traj.turns[-1].state_flow != 0.0 else traj.log_z
    return math.exp(final_state_flow)


def compute_forward_trajectory_log_flow(traj: Trajectory) -> float:
    """
    计算轨迹的 forward-only log-flow（log 空间，避免下溢）。

    论文 §3.4：
      log F(τ) = log Z_θ(q) + Σ_t log π̃_θ(a_t | H_{t-1})
    """
    log_flow = traj.log_z
    for turn in traj.turns:
        log_flow += edge_logprob_tilde(turn, "forward")
    return log_flow


def compute_forward_trajectory_flow(traj: Trajectory) -> float:
    """计算轨迹的 forward-only flow（线性空间，可能下溢为 0）。"""
    return math.exp(compute_forward_trajectory_log_flow(traj))


def _log_sum_exp(log_values: List[float]) -> float:
    """数值稳定的 log-sum-exp。"""
    if not log_values:
        return -math.inf
    max_val = max(log_values)
    if max_val == -math.inf:
        return -math.inf
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def _compute_group_flow_entropy(log_flows: List[float]) -> float:
    """
    对一组 log F(τ) 计算 flow entropy（内部工具函数）。

    log-space softmax → p̂(τ) = exp(log_flow - log_sum_exp) → entropy
    """
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
    """
    计算 batch 内的流量熵 Ĥ_flow(B)。

    论文 Eq.10：
      p̂(τ) = F(τ) / Σ F(τ'),  Ĥ_flow = -Σ p̂(τ) · log p̂(τ)
      F(τ) = Z_θ(q) · ∏_t π_θ(a_t | H_{t-1})

    LLM multi-token 适配：
      LLM 的每个 action 包含 30-200+ tokens，导致 log F(τ) = log Z + Σ log π
      的绝对值非常大（-50 到 -300+），不同步数的轨迹之间差异达 50-200+。
      直接 softmax 必然坍塌到 log flow 最大的那条（通常是步数最少的）。

      修复：先按论文对每条 edge 做 per-token tempering，再用 /T 的
      normalized log flow 计算 entropy，避免输出 token 长度和轨迹步数
      主导 softmax。论文 Eq.10 的语义不变（衡量 flow 是否集中在少数轨迹上），
      只是归一化到可比较的尺度。

    值域：[0, log|B|]
    """
    if not trajectories:
        return 0.0

    h_max_batch = math.log(len(trajectories)) if len(trajectories) > 1 else 1.0

    # 使用 paper edge + per-step normalized log flow，消除长度差异导致的尺度问题
    norm_log_flows = [_per_token_normalized_log_flow(t) for t in trajectories]
    h = _compute_group_flow_entropy(norm_log_flows)

    return h


def compute_skill_marginal_flows(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:
    """
    技能边际流量 F̂(s)（论文 Eq. 7 / Definition E.3）：

        F̂(s) = (1 / |B_s|) · Σ_{τ ∈ B_s} Σ_{t: a_t invokes s} F(s_t)

    注：求和的是 *post-action state flow* F(s_t)，不是 edge flow F(s_{t-1}→s_t)。
    在 SkillFlow 的 tree-DAG 上（Lemma C.2），TB 收敛时两者相等；训练中差值
    恰为 log P_φ(a_t | H_{t-1}⊕o_t^exec)，体现未收敛残差。

    log F(s_t) 由 telescoping 还原（Eq. 14）：
        log F(s_t) = log Z_θ(q) + Σ_{t'≤t} log I(t')   = turn.state_flow

    `|B_s|` 是 *包含 skill s 的轨迹数*，不是调用总次数。
    """
    # skill_id → {traj_id → [log STATE flow values]}
    skill_traj_log_flows: Dict[str, Dict[str, List[float]]] = {sid: {} for sid in all_skill_ids}

    for traj in trajectories:
        for turn in traj.turns:
            if turn.action_type == "skill_invoke" and turn.skill_id in skill_traj_log_flows:
                sid = turn.skill_id
                if traj.traj_id not in skill_traj_log_flows[sid]:
                    skill_traj_log_flows[sid][traj.traj_id] = []
                # 论文 Eq.7: 用 state flow log F(s_t)，不是 edge flow
                skill_traj_log_flows[sid][traj.traj_id].append(turn.state_flow)

    marginal_flows: Dict[str, float] = {}
    for sid in all_skill_ids:
        per_traj = skill_traj_log_flows[sid]
        if per_traj:
            # 每条轨迹内: log-sum-exp 得到 log(Σ F(edge))
            # 再对所有轨迹取平均（在 log 空间: log-sum-exp - log|B_s|）
            traj_log_sums = [_log_sum_exp(log_flows) for log_flows in per_traj.values()]
            log_marginal = _log_sum_exp(traj_log_sums) - math.log(len(per_traj))
            # 保持 log 空间，避免 exp 爆炸（原 math.exp(65+) → 1e28+）
            # clamp 到 [-20, 20] 保持数值稳定，exp(20) ≈ 4.8e8 已足够区分
            marginal_flows[sid] = max(min(log_marginal, 20.0), -20.0)
        else:
            marginal_flows[sid] = -20.0  # 未使用的 skill → 最低 log flow

    return marginal_flows


def compute_skill_reward_variance(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:
    """
    每技能 *轨迹奖励方差* Var_{τ∈B_s}[R(τ)]（辅助诊断量，**非论文 Jensen gap**）。

    注：这与 paper Prop F.5 的 Jensen gap 是 **不同的物理量**。
      Jensen gap = ½ Var_{V_s}[log F(s_t)] + 高阶 cumulants
                 → 度量 skill 在不同上下文下的 *flow 稳定性*
      Reward variance = Var[R(τ)]
                 → 度量 skill 出现的轨迹的 reward 分布广度
    Jensen gap 实现见 `compute_skill_jensen_gap()`。
    """
    # 每个技能对应的轨迹奖励列表（去重：同一轨迹只计一次）
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
            variances[sid] = 0.0  # 样本不足，方差设为 0
        else:
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            variances[sid] = var

    return variances


# 向后兼容别名
compute_skill_variances = compute_skill_reward_variance


# ──────────────────────────────────────────────────────
# CGF helpers — 论文 §4.3 + Appendix F
# ──────────────────────────────────────────────────────

def compute_skill_log_F_visits(
    trajectories: List[Trajectory],
    skill_id: str,
) -> List[float]:
    """
    收集所有调用 skill `skill_id` 的访问点上的 log F(s_t)。

    每个 (τ, t) 满足 a_t = skill_invoke(skill_id) 的访问产生一个 log F(s_t) 样本。
    在 atomic-tip 假设下（每条轨迹至多调用一次），|V_s| = |B_s|。
    """
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
    """
    Mean log-flow G(s) = ∂Λ^(s)_λ/∂λ|_{λ=0} = visit-mean of (log F(s_t) - log Z_θ(q))
    （论文 Lemma F.2）。

    实现：取所有访问点的 log F(s_t) 平均，再减去访问点的 log_z 平均。
    """
    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if not visits:
        return -math.inf
    log_F_mean = sum(visits) / len(visits)
    # log Z 在不同 query 间不同，取访问轨迹平均
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
    """
    Λ^(s)_1 = log F̂(s) - log Z_θ(q)（论文 Lemma F.3）。

    通过 log-sum-exp 的 visit 平均稳定地计算 log F̂(s) - log Z。
    """
    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if not visits:
        return -math.inf
    z_vals = [
        traj.log_z for traj in trajectories
        for turn in traj.turns
        if turn.action_type == "skill_invoke" and turn.skill_id == skill_id
    ]
    log_z_mean = sum(z_vals) / len(z_vals) if z_vals else log_z_default
    # log mean(exp(visit)) = log-sum-exp(visits) - log|V_s|
    log_F_hat = _log_sum_exp(visits) - math.log(len(visits))
    return log_F_hat - log_z_mean


def compute_skill_jensen_gap(
    trajectories: List[Trajectory],
    skill_id: str,
) -> float:
    """
    Jensen gap Λ^(s)_1 - G(s)（论文 Prop F.5）：

        Λ^(s)_1 - G(s) = ½ Var_{V_s}[log F(s_t)] + Σ_{k≥3} κ_k(s)/k!

    主导项为半方差，是 skill 跨上下文的 *flow 稳定性诊断*：
      - gap 小 → skill 在所有调用上下文下贡献近似常量 flow → context-consistent → retain
      - gap 大 → skill flow 贡献随上下文剧烈变化 → context-inconsistent → refine

    返回 Jensen gap（≥ 0）。
    """
    visits = compute_skill_log_F_visits(trajectories, skill_id)
    if len(visits) < 2:
        return 0.0
    # Λ_1 - G = log mean(exp(X)) - mean(X), where X = log F(s_t) (Z 项相消)
    mean_X = sum(visits) / len(visits)
    log_mean_exp_X = _log_sum_exp(visits) - math.log(len(visits))
    return log_mean_exp_X - mean_X


def compute_skill_lambda_tilde(
    trajectories: List[Trajectory],
    all_skill_ids: List[str],
) -> Dict[str, float]:
    """
    Centered log-flow share Λ̃(s) = Λ^(s)_1 - E_{s'}[Λ^(s')_1]（论文 Remark F.7）。

    相对排名信号：Λ̃(s) > 0 → skill 在库内为高 flow 贡献者；
                 Λ̃(s) < 0 → 持续多个 phase 后归入 𝒟⁻ (prune class)。
    """
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
    """
    一次性返回每个 skill 的 (G, Λ_1, Λ̃, Jensen gap)，供进化算子 Φ 使用。
    """
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


# ──────────────────────────────────────────────────────
# Trigger steps — 论文 Definition F.4
# ──────────────────────────────────────────────────────

def extract_trigger_steps(
    success_traj: Trajectory,
    zeta_trig: float = 1.0,
    covered_step_indices: Optional[set] = None,
) -> List[int]:
    """
    论文 Definition F.4：从 (q, τ⁺, τ⁻) pair 的成功轨迹中抽取 trigger 步骤集合：

        T_q^trig = { t : log I(t)|τ⁺ ≥ ζ_trig  ∧  t ∉ cov(R ∪ U') }

    Args:
        success_traj: τ⁺
        zeta_trig: 高重要性阈值（log space）
        covered_step_indices: 已被现存或 refined skill 覆盖的步骤索引集合

    Returns:
        触发步骤索引列表（按 log I(t) 降序）
    """
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
    """
    找到轨迹中 I(t) 最高的 top-K 步骤。

    用于 backward_pattern_mine：提取关键步骤周围的动作子序列。

    Returns:
        [(step_idx, importance_score), ...] 按重要性降序
    """
    if not traj.turns:
        return []

    indexed = [(i, t.step_importance) for i, t in enumerate(traj.turns)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return indexed[:top_k]


# ──────────────────────────────────────────────────────
# TTB Loss 辅助
# ──────────────────────────────────────────────────────


def compute_ttb_balance_error(
    log_z: float,
    forward_logprobs: List[float],
    backward_logprobs: List[float],
    r_tilde: float,
    beta: float = 1.0,
    n_steps: Optional[int] = None,
    action_token_counts: Optional[List[int]] = None,
) -> float:
    """
    单条轨迹的 TTB balance error，用于调试/验证（非 torch）。

    论文 Eq. 13 (main):
        Δ(τ) = log Z_θ(q) + Σ_t log π̃_θ(a_t|r_t,H_{t-1}) - β log R̃(τ)
                            - Σ_t log P̃_φ(a_t|H_{t-1}⊕o_t^exec)
        L_TTB(τ) = (Δ(τ) / T)²       T = |τ|

    收敛时 Δ(τ) → 0 ⟺ 满足 reward-matching TB 条件（Theorem A.5）。

    如果传入 action_token_counts，则 forward/backward 输入会按 K_t 转成
    per-token-tempered edge log-prob；否则假设输入已经是 log π̃ / log P̃。
    """
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
    """
    计算 batch 内 TTB 统计（用于日志监控）。
    假设 forward/backward logprobs 已填充。
    """
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


# ──────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────


def _per_token_normalized_log_flow(traj: Trajectory) -> float:
    """
    Per-edge 归一化的 log flow，用于 H_flow 和 quartile 排序。

    raw log F(τ) = log Z + Σ log π̃_θ(a_t)
    normalized = raw / T,  T = scored action edges

    每条 edge 内部已按论文 Definition B.2 做 per-token tempering；
    这里再除以 T，只用于跨不同轨迹长度做 entropy/quartile 排序。
    """
    log_flow = compute_forward_trajectory_log_flow(traj)
    return log_flow / effective_paper_steps(traj)


def split_by_flow_quartile(
    trajectories: List[Trajectory],
    low_frac: float = 0.25,
    high_frac: float = 0.25,
) -> Tuple[List[Trajectory], List[Trajectory]]:
    """
    按 per-token 归一化的 trajectory flow 分割低/高 quartile。

    论文 §3.4：低 flow quartile 用于失败分析，高 flow quartile 用于成功蒸馏。

    LLM 适配：使用 per-token 归一化 flow 排序，确保跨 task_type
    的排序反映策略质量而非轨迹长度/token数量。

    Returns:
        (low_flow_trajs, high_flow_trajs)
    """
    if not trajectories:
        return [], []

    sorted_trajs = sorted(trajectories, key=_per_token_normalized_log_flow)
    n = len(sorted_trajs)
    low_n = max(1, int(n * low_frac))
    high_n = max(1, int(n * high_frac))

    return sorted_trajs[:low_n], sorted_trajs[n - high_n:]


# ──────────────────────────────────────────────────────
# 3+2+1 Skill Evolution: Flow Bottleneck + Counterfactual + Diagnosis
# ──────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class FlowBottleneck:
    """Flow residual 检测到的瓶颈 cluster。"""
    task_type: str
    step_bucket: int          # 步骤位置桶 (0=steps 0-1, 1=steps 2-3, ...)
    action_type: str          # cluster 中主要的 action type
    mean_log_It: float        # log I(t) 均值（正=探索性，负=backward 认可）
    var_log_It: float         # log I(t) 方差（高=决策不一致=瓶颈）
    n_samples: int            # cluster 中的步骤数
    example_steps: List[Dict] = field(default_factory=list)  # 最多 3 个代表步骤


@dataclass
class CounterfactualPair:
    """同问题轨迹对的局部反事实分歧点。"""
    question: str
    divergence_step: int              # 路径首次分歧的步骤 (0-based)
    context_before: List[Dict]        # 分歧前共享前缀 [{action_type, instruction}]
    success_choice: Dict              # {action_type, instruction, observation, I_t}
    failure_choice: Dict              # 同上
    success_downstream: Dict          # {n_remaining_steps, final_reward}
    failure_downstream: Dict          # 同上
    reward_gap: float


@dataclass
class BottleneckDiagnosis:
    """结构化瓶颈诊断，供 LLM curation 使用。"""
    task_type: str
    bottleneck: FlowBottleneck
    counterfactual_pairs: List[CounterfactualPair]
    current_skill_coverage: str       # 已有 skill 覆盖情况
    suggested_edit_type: str          # "ADD" | "UPDATE" | "SPLIT"


def compute_flow_bottlenecks(
    trajectories: List[Trajectory],
    bucket_size: int = 2,
    min_samples: int = 4,
    var_threshold: float = 2.0,
    mean_threshold: float = 1.0,
) -> Dict[str, List[FlowBottleneck]]:
    """按 (task_type, step_bucket, action_type) 聚类，找 log I(t) 方差高的瓶颈。

    高方差 = 模型在同一决策点的前后策略不一致 = 需要 skill 指导。

    Returns: {task_type: [FlowBottleneck, ...]} 按 var 降序
    """
    from collections import defaultdict

    # 聚类
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

    # 计算统计量
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

    # 按方差降序
    for tt in result:
        result[tt].sort(key=lambda b: b.var_log_It, reverse=True)

    return dict(result)


def extract_counterfactual_pairs(
    trajectories: List[Trajectory],
    min_reward_gap: float = 0.3,
) -> Dict[str, List[CounterfactualPair]]:
    """找同问题轨迹的分歧点，构造局部反事实 pair。

    Returns: {task_type: [CounterfactualPair, ...]}
    """
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

        # 找分歧点：首个 action_type 不同的步骤
        best_turns = [t for t in best.turns if getattr(t, 'action_type', '') != 'skill_invoke']
        worst_turns = [t for t in worst.turns if getattr(t, 'action_type', '') != 'skill_invoke']

        div_step = 0
        context = []
        for i in range(min(len(best_turns), len(worst_turns))):
            bt, wt = best_turns[i], worst_turns[i]
            if getattr(bt, 'action_type', '') != getattr(wt, 'action_type', ''):
                div_step = i
                break
            # 也检查 instruction 是否差异大
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
    """将瓶颈和反事实对匹配，生成结构化诊断。"""
    diagnoses = []
    cps = counterfactual_pairs or []

    for bn in bottlenecks[:3]:
        # 匹配反事实对到此瓶颈的 step_bucket
        matched_cps = [
            cp for cp in cps
            if cp.divergence_step // 2 == bn.step_bucket  # same bucket
        ][:2]

        # 检查已有 skill 覆盖
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
