"""
GFlowNetTrainer — SkillFlow 核心训练器。

替换 CWRPOTrainer，优化目标：
  min_{θ,φ,Z} E[L_TTB(τ)]

三个可训练组件：
  - π_θ: Supervisor θ-LoRA（前向策略，Qwen3-8B）
  - P_φ: BackwardPolicy φ-LoRA（反向策略）
  - Z_θ: PartitionFunctionHead（分区函数标量）

训练循环：
  每步 = 收集 batch_size 个 episodes + 计算 TTB loss + backward + optimizer step
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.executor.m_exec import MExec
from src.skills.workspace import SkillWorkspace
from src.skills.skill_creator import SkillCreator
from training.backward_policy import BackwardPolicy
from training.environment import GenericTaskEnvironment
from training.flow_metrics import (
    compute_flow_entropy,
    compute_skill_marginal_flows,
    compute_skill_variances,
    edge_log_i,
    edge_logprob_tilde,
    effective_paper_steps,
    fill_turn_flows,
)
from training.reward import EPSILON_MIN
from training.skill_evolution import FlowSkillEvolutionManager, TaskTypeAccuracyTracker, PlateauDetector
from training.trajectory import Trajectory, split_think_and_action

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Signal-driven evolution helpers
# ──────────────────────────────────────────────────────

def _variance(xs: list) -> float:
    """Simple variance without numpy."""
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _spearman_rank(xs: list, ys: list) -> float:
    """Simple Spearman rank correlation without scipy."""
    n = len(xs)
    if n < 4:
        return 0.0
    # Compute ranks
    def _ranks(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank, idx in enumerate(indexed):
            ranks[idx] = rank + 1.0
        return ranks
    rx = _ranks(xs)
    ry = _ranks(ys)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ──────────────────────────────────────────────────────
# ObservationBuffer — 零成本 I(t) 关键步骤收集器
# ──────────────────────────────────────────────────────

class ObservationBuffer:
    """每步自动收集 I(t)<threshold 的关键步骤，供 flow-driven tip 提取使用。

    设计原则（compact memory 风格）：
    - 只收集"非显而易见"的步骤（I(t) << 1 = backward 确信但 forward 不确信）
    - 零 M_exec 成本：只读 turn.step_importance
    - Rolling window：每个 task_type 保留最近 max_per_type 条
    """

    def __init__(self, max_per_type: int = 20, importance_threshold: float = 0.3):
        from collections import defaultdict
        self._buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._max_per_type = max_per_type
        self._threshold = importance_threshold

    def collect(self, trajectories: List[Trajectory]) -> int:
        """从高 reward 轨迹中收集 I(t) < threshold 的步骤。返回收集数量。"""
        n_collected = 0
        for traj in trajectories:
            if traj.reward < 0.0:
                continue
            for t_idx, turn in enumerate(traj.turns):
                if turn.step_importance > 0 and turn.step_importance < self._threshold:
                    self._buffer[traj.task_type].append({
                        "task_type": traj.task_type,
                        "action_type": turn.action_type,
                        "instruction": (turn.instruction or "")[:100],
                        "step_position": t_idx,
                        "n_steps": len(traj.turns),
                        "I_t": round(turn.step_importance, 4),
                        "traj_reward": round(traj.reward, 3),
                    })
                    n_collected += 1
        # Rolling window
        for tt in self._buffer:
            if len(self._buffer[tt]) > self._max_per_type:
                self._buffer[tt] = self._buffer[tt][-self._max_per_type:]
        return n_collected

    def get(self, task_type: str) -> List[Dict]:
        return self._buffer.get(task_type, [])

    def clear(self, task_type: str) -> None:
        self._buffer.pop(task_type, None)

    def all_types(self) -> List[str]:
        return list(self._buffer.keys())

    def total_count(self) -> int:
        return sum(len(v) for v in self._buffer.values())


# ──────────────────────────────────────────────────────
# Partition Function Head Z_θ(q)
# ──────────────────────────────────────────────────────


class PartitionFunctionHead(nn.Module):
    """
    Z_θ(q) = 任务条件化的标量分区函数。

    实现：在 Supervisor 的 query 表示上加一个线性层。
    使用任务类型 embedding 融合，避免不同任务奖励量级混淆。

    收敛时：log Z_θ(q) ≈ β·log Σ_τ R̃(τ)（当 L_TTB → 0 时）
    """

    def __init__(self, hidden_size: int = 3584, num_task_types: int = 10):
        super().__init__()
        self.task_embed = nn.Embedding(num_task_types, 64)
        self.head = nn.Linear(hidden_size + 64, 1, bias=True)

        # 初始化：bias = log(ε_min) ≈ -2.3（log 空间）
        # head 输出直接作为 log Z_θ(q)，初始 Z 应接近 ε_min = 0.1
        # 旧值 0.1 导致初始 Z = exp(0.1) ≈ 1.1，TTB balance error 偏大
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.3)  # log(0.1) ≈ -2.303

    def forward(
        self,
        query_hidden: torch.Tensor,  # [B, hidden_size]
        task_type_ids: torch.Tensor, # [B]
    ) -> torch.Tensor:
        """返回 log Z_θ(q) for each query in batch [B]"""
        task_vec = self.task_embed(task_type_ids)  # [B, 64]
        combined = torch.cat([query_hidden, task_vec], dim=-1)  # [B, H+64]
        log_z = self.head(combined).squeeze(-1)  # [B]
        # Clamp to physically reasonable range: Z ∈ [e^-10, e^10]
        return torch.clamp(log_z, min=-10.0, max=10.0)


# ──────────────────────────────────────────────────────
# TTB Loss
# ──────────────────────────────────────────────────────


def compute_ttb_loss(
    log_z: torch.Tensor,                        # [B]
    forward_logprob_sums: torch.Tensor,         # [B] Σ_t log π̃_θ(a_t)  (paper-tempered)
    backward_logprob_sums: torch.Tensor,        # [B] Σ_t log P̃_φ(a_t)  (paper-tempered)
    r_tilde: torch.Tensor,                      # [B] R̃ = R + ε_min
    beta: float = 1.0,
    normalizer: Optional[torch.Tensor] = None,  # [B] per-trajectory T = # scored action edges
) -> torch.Tensor:
    """
    Tempered Trajectory Balance (TTB) Loss — 论文 Eq. 13 (main):

        Δ(τ) = log Z_θ(q) + Σ_t log π̃_θ(a_t|r_t,H_{t-1})
                            - β·log R̃(τ)
                            - Σ_t log P̃_φ(a_t|H_{t-1}⊕o_t^exec)
        L_TTB(τ) = (Δ(τ) / T)²        T = |τ|

    调用者需先把 LM token-sum 转为 per-token-tempered edge log-prob：
        log π̃(a_t) = (1/K_t) Σ_j log π(tok_j)
    `normalizer` 应传论文中的 T=有效 action edges 数。

    收敛: Δ(τ)→0 ⟹ π_θ(a_{1:T}|r_{1:T},o^exec_{1:T},q) ∝ R̃(τ)^β（Theorem A.5）。
    """
    log_r_tilde = torch.log(r_tilde.clamp(min=1e-8))
    balance = log_z + forward_logprob_sums - beta * log_r_tilde - backward_logprob_sums
    if normalizer is not None:
        balance = balance / normalizer.clamp(min=1)
    loss = (balance ** 2).mean()
    return loss


# ──────────────────────────────────────────────────────
# GFlowNetTrainer
# ──────────────────────────────────────────────────────

TASK_TYPE_TO_ID = {
    "multi_hop_qa": 0,
    "factual_qa": 1,
    "fact_checking": 2,
    "math_reasoning": 3,
    "code_generation": 4,
    "strategy_qa": 5,
    "open_ended": 6,
    "interactive_agent": 7,
    "science_qa": 8,
    "unknown": 9,
    "webshop": 10,
    "alfworld": 11,
}


class GFlowNetTrainer:
    """
    GFlowNet TTB 训练主类。

    管理：
      - Episode 收集（通过 GenericTaskEnvironment）
      - Forward/backward logprob 计算
      - TTB loss 反向传播
      - Skill evolution（通过 FlowSkillEvolutionManager）
      - Checkpoint 保存/加载
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
        self.output_dir = Path(config.get("output_dir", "outputs/skillflow"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 模型路径
        self.base_model_path = config["base_model"]
        self.beta = config.get("beta", 1.0)
        self.epsilon_min = config.get("epsilon_min", EPSILON_MIN)

        # 训练超参
        self.max_steps = config.get("max_steps", 300)
        self.batch_size = config.get("batch_size", 8)
        self.max_episode_steps = config.get("max_episode_steps", 8)
        self.save_every = config.get("save_every", 50)
        self.sync_lora_every = config.get("sync_lora_every", 5)
        self.evolution_phase_steps = config.get("evolution_phase_steps", 20)
        self.lora_rank = config.get("lora_rank", 64)

        # 组件（延迟初始化）
        self.tokenizer: Optional[AutoTokenizer] = None
        self.supervisor_model: Optional[PeftModel] = None  # π_θ-LoRA
        self.partition_fn: Optional[PartitionFunctionHead] = None
        self.backward_policy: Optional[BackwardPolicy] = None
        self.m_exec: Optional[MExec] = None
        self.workspace: Optional[SkillWorkspace] = None
        self.skill_creator: Optional[SkillCreator] = None
        self.evolution_manager: Optional[FlowSkillEvolutionManager] = None
        self.env: Optional[GenericTaskEnvironment] = None

        # 优化器（θ / Z_θ / φ 三路联合）
        self._supervisor_optimizer: Optional[torch.optim.Optimizer] = None
        self._partition_optimizer: Optional[torch.optim.Optimizer] = None
        self._phi_optimizer: Optional[torch.optim.Optimizer] = None

        # 状态
        self._current_step = 0
        self._skills_just_updated = False
        self._phase_trajectories: List[Trajectory] = []

        # Flow-driven observation buffer（I(t) 关键步骤收集器）
        self._observation_buffer = ObservationBuffer(
            max_per_type=20,
            importance_threshold=self.config.get("observation_I_threshold", 0.3),
        )
        # Per-type balance error history（进化触发信号）
        from collections import defaultdict
        self._per_type_balance_history: Dict[str, List[float]] = defaultdict(list)
        self._per_type_last_evolution: Dict[str, int] = defaultdict(int)  # step of last evolution per type
        self._per_type_evolution_helped: Dict[str, bool] = defaultdict(lambda: True)  # adaptive cooldown
        self._per_type_acc_history: Dict[str, List[float]] = defaultdict(list)  # per-type accuracy history
        self._skill_negative_counter: Dict[str, int] = defaultdict(int)  # paper D⁻ persistent negative Λ̃ counter
        self._plateau_detector = PlateauDetector(
            window_size=self.config.get("plateau_window_size", self.evolution_phase_steps),
            rho=self.config.get("plateau_rho", 0.05),
            m_consecutive=self.config.get("plateau_m_consecutive", 2),
        )

        # 跨 episode 经验缓冲（M_mem 的 E 组件，论文 §3.3 E_ret）
        # Experience 功能已移除 — prompt 注入有害（膨胀上下文），reward shaping 效果极小
        self.experience_store = None
        self._experience_buffer: List[Dict] = []
        self._experience_lock = threading.Lock()

        # 日志
        self._log_file = self.output_dir / "training_log.jsonl"

        # wandb
        try:
            import wandb
            wandb.init(
                project="skillflow",
                name=config.get("exp_name", "skillflow"),
                config={k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))},
                resume="allow",
            )
            self._wandb = wandb
            logger.info("wandb initialized")
        except Exception as e:
            self._wandb = None
            logger.warning(f"wandb init failed: {e}")

    def setup(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None) -> None:
        """初始化所有组件"""
        logger.info("Setting up GFlowNetTrainer...")

        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 共享基础模型 + 两个 LoRA 适配器（节省 ~16 GB 显存）
        #    θ-LoRA (π_θ, Supervisor 前向策略) + φ-LoRA (P_φ, 后向策略)
        #    gradient_checkpointing：batched micro-batch 模式下安全启用。
        #    每个 micro-batch 内适配器固定，recompute 时 adapter 不变。
        logger.info("Loading shared base model (θ-LoRA + φ-LoRA)...")
        self.extra_device = self.config.get("extra_device", None)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        # θ-LoRA：Supervisor 前向策略，较大（覆盖 q/k/v/o 四个投影）
        theta_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.shared_model = get_peft_model(base_model, theta_config, adapter_name="theta")

        # φ-LoRA：反向策略，轻量（仅 q/v）
        # 论文 Appendix H.2 + J: rank=16, α=32, target=(q_proj, v_proj)
        phi_rank = self.config.get("backward_lora_rank", 16)
        phi_config = LoraConfig(
            r=phi_rank,
            lora_alpha=phi_rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.shared_model.add_adapter("phi", phi_config)

        # 默认激活 theta（Supervisor 前向策略）
        # NOTE: set_adapter() 会把非活跃 adapter 的 requires_grad 重置为 False，
        # 所以必须在 set_adapter 之后再手动开启 phi 参数的梯度。
        self.shared_model.set_adapter("theta")
        self.shared_model.train()

        # set_adapter("theta") 把 phi 的 requires_grad 关掉了，手动重新开启
        phi_grad_count = 0
        for name, param in self.shared_model.named_parameters():
            if ".lora_A.phi" in name or ".lora_B.phi" in name:
                param.requires_grad_(True)
                phi_grad_count += 1
        logger.info(f"Re-enabled requires_grad for {phi_grad_count} φ-LoRA params")
        self.supervisor_model = self.shared_model  # alias，保持其他代码兼容

        # 启用 gradient checkpointing（batched micro-batch 下安全）
        self.shared_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for batched micro-batch training")

        # ── Data Parallel: GPU 3 replica (backward, WITH gradient checkpointing) ──
        # Key insight: gradient checkpointing with use_reentrant=False (transformers 5.3 default)
        # is thread-safe when each thread operates on a SEPARATE model on a SEPARATE GPU.
        # The old approach (no checkpointing on replica, micro_bs=2) was a workaround for a
        # misdiagnosed issue — the real problem was threading on a SHARED model, not checkpointing.
        # With checkpointing on both GPUs: peak ~25GB each (vs 61GB without), micro_bs=4 on both.
        self.replica_model = None
        if self.extra_device:
            logger.info(f"Loading replica on {self.extra_device}...")
            _b2 = AutoModelForCausalLM.from_pretrained(
                self.base_model_path, dtype=torch.bfloat16,
                device_map=self.extra_device, trust_remote_code=True)
            _phi_r = self.config.get("backward_lora_rank", 16)
            self.replica_model = get_peft_model(_b2, LoraConfig(
                r=self.lora_rank, lora_alpha=self.lora_rank*2,
                target_modules=["q_proj","k_proj","v_proj","o_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"), adapter_name="theta")
            self.replica_model.add_adapter("phi", LoraConfig(
                r=_phi_r, lora_alpha=_phi_r*2, target_modules=["q_proj","v_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
            self.replica_model.set_adapter("theta")
            self.replica_model.train()
            for _n, _p in self.replica_model.named_parameters():
                if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                    _p.requires_grad_(True)
            self.replica_model.gradient_checkpointing_enable()
            self._sync_replica_weights()
            logger.info(f"  Replica ready on {self.extra_device} (with gradient checkpointing)")

        # 3. Partition Function Head
        hidden_size = base_model.config.hidden_size
        self.partition_fn = PartitionFunctionHead(
            hidden_size=hidden_size,
            num_task_types=len(TASK_TYPE_TO_ID),
        ).to(self.device)

        # 4. Backward Policy P_φ（共享模型，不单独加载 base）
        self.backward_policy = BackwardPolicy(
            shared_model=self.shared_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        # 5. M_exec
        self.m_exec = MExec(
            api_base=self.config["executor_api_base"],
            model_name=self.config.get("executor_model", "gpt-oss-120b"),
            api_key=self.config.get("executor_api_key") or os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY"),
        )
        logger.info(f"M_exec connectivity: {self.m_exec.test_connectivity()}")

        # 6. Skill Workspace
        skills_dir = self.output_dir / "skills"
        self.workspace = SkillWorkspace(
            skills_dir=skills_dir,
            max_skills=self.config.get("max_skills_total", 60),
        )

        # 7. Skill Creator（传入 backward_policy，供 refine_by_counterfactual P_φ 评分）
        self.skill_creator = SkillCreator(
            m_exec=self.m_exec,
            skill_workspace=self.workspace,
            backward_policy=self.backward_policy,
        )

        # 8. Environment（必须在 Genesis 之前初始化：_run_exploratory_episodes 依赖 self.env）
        self.env = GenericTaskEnvironment(
            m_exec=self.m_exec,
            max_episode_steps=self.max_episode_steps,
            epsilon_min=self.epsilon_min,
            skill_workspace=self.workspace,
            reward_mode=self.config.get("reward_mode", "outcome_only"),
            skill_mode=self.config.get("skill_mode", "policy_action"),
        )

        # 9. Genesis（如果 workspace 为空）
        # S₀ = ∅：从空技能库开始，前 N 步用 tools 探索，
        # 然后通过进化从轨迹中自动发现 skills（更符合论文设计）
        if self.workspace.size == 0:
            genesis_count = self.config.get("genesis_count", 12)
            if genesis_count > 0:
                logger.info("Workspace empty, running genesis...")
                self._run_genesis(train_data)
            else:
                logger.info("Workspace empty, S₀=∅ — skills will be discovered from trajectories via evolution")

        # 10. Per-task-type accuracy tracker（OpenClaw-RL 风格）
        self.accuracy_tracker = TaskTypeAccuracyTracker(
            threshold=self.config.get("accuracy_threshold", 0.5),
            min_count=10,
        )

        # 11. Evolution Manager
        self.evolution_manager = FlowSkillEvolutionManager(
            workspace=self.workspace,
            skill_creator=self.skill_creator,
            delta_h=self.config.get("flow_entropy_threshold", 0.85),
            delta_prune=self.config.get("prune_threshold", -10.0),
            min_usage_before_prune=self.config.get("min_usage_before_prune", 20),
            max_skills_total=self.config.get("max_skills_total", 60),
            evolution_phase_steps=self.evolution_phase_steps,
            min_trajs_for_evolution=self.config.get("min_trajs_for_evolution", 48),
            experience_store=None,
            beta=self.beta,
            accuracy_tracker=self.accuracy_tracker,
            accuracy_threshold=self.config.get("accuracy_threshold", 0.5),
        )

        # 11. 优化器（在所有模型组件初始化完成后）
        self._setup_optimizers()

        self._train_data = train_data
        self._val_data = val_data or []

        logger.info(
            f"Setup complete: workspace={self.workspace.size} skills, "
            f"train={len(train_data)}, val={len(self._val_data)}"
        )

    def _sample_batch_for_step(self, step: int) -> List[Dict]:
        """GFlowNet DAG 采样：每个问题采样 N 条轨迹，构成树形 DAG。

        论文 §4.3 要求同一个 q 的多条轨迹共享 Z(q)，使得：
        - flow conservation 在分叉点有效约束策略
        - P_φ 从成功/失败对比中学习 credit assignment
        - I(t) = π_θ/P_φ 在分叉点具有对比意义
        temperature > 0 保证同一 q 的不同轨迹走不同路径。
        """
        import random as _random
        _seed_offset = int(self.config.get("seed_offset", 0))
        rng = _random.Random(42 + step + _seed_offset * 131)

        n_traj = self.config.get("n_trajectories_per_question", 2)

        by_source: Dict[str, List[Dict]] = {}
        for q in self._train_data:
            src = q.get("extra", {}).get("source", q.get("task_type", "unknown"))
            by_source.setdefault(src, []).append(q)

        # 每个 dataset 选 1 个 unique question，每个 question 复制 n_traj 次
        n_unique = max(1, self.batch_size // n_traj)
        sources = list(by_source.keys())
        per_source = max(1, n_unique // len(sources)) if sources else 1

        unique_questions = []
        for src, qs in by_source.items():
            if qs:
                unique_questions.extend(rng.sample(qs, min(per_source, len(qs))))

        # 补齐到 n_unique
        while len(unique_questions) < n_unique:
            unique_questions.append(rng.choice(self._train_data))
        unique_questions = unique_questions[:n_unique]

        # 每个 question 复制 n_traj 次（temperature 保证不同轨迹）
        batch = []
        for q in unique_questions:
            for _ in range(n_traj):
                batch.append(q.copy())

        rng.shuffle(batch)
        return batch[:self.batch_size]

    def train(self) -> None:
        """主训练循环 — TBA 风格异步 rollout/gradient 重叠"""
        logger.info(f"Starting GFlowNet training for {self.max_steps} steps")
        start_time = time.time()
        from concurrent.futures import ThreadPoolExecutor, Future

        # v10: clean shutdown handler — kill SGLang child gracefully on SIGTERM/SIGINT
        # 否则 SIGKILL 父进程会 orphan SGLang 子进程的 CUDA context (GPU 内存不释放)
        import signal as _signal
        _shutdown_pending = {"flag": False}
        def _graceful_shutdown(signum, frame):
            if _shutdown_pending["flag"]:
                return
            _shutdown_pending["flag"] = True
            logger.warning(f"[shutdown] signal {signum} received; stopping SGLang child cleanly...")
            try:
                if hasattr(self, "sglang_mgr") and self.sglang_mgr is not None:
                    self.sglang_mgr.stop(timeout_s=30)
                logger.info("[shutdown] SGLang child stopped; exiting.")
            except Exception as _e:
                logger.warning(f"[shutdown] stop failed: {_e}")
            import sys as _sys
            _sys.exit(0)
        _signal.signal(_signal.SIGTERM, _graceful_shutdown)
        _signal.signal(_signal.SIGINT, _graceful_shutdown)

        # v10: 启动 SGLang supervisor as child mp.Process (shared authkey).
        # 必须在 first sync 之前; 如果已存在 (resume 场景) 跳过.
        if not hasattr(self, "sglang_mgr") or self.sglang_mgr is None:
            from training.sglang_manager import SGLangSupervisorManager, _set_shared_authkey
            _set_shared_authkey()  # 父进程 authkey 固定 (spawn 子进程继承)
            api_base = self.config['supervisor_api_base'].rstrip('/v1')
            port = int(api_base.split(':')[-1].split('/')[0])
            self.sglang_mgr = SGLangSupervisorManager(
                model_path=self.config.get("model_path") or self.config.get("base_model", "Qwen/Qwen3.5-9B"),
                port=port,
                api_key=self.config.get("supervisor_api_key") or os.environ.get("SUPERVISOR_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY"),
                gpu_id=self.config.get("supervisor_gpu_id", 0),
                max_lora_rank=self.config.get("lora_rank", 64),
                lora_target_modules=self.config.get("lora_target_modules", ["q_proj","k_proj","v_proj","o_proj"]),
                max_loras_per_batch=1,
                max_loaded_loras=2,
                mem_fraction_static=0.82,
                context_length=32768,
            )
            logger.info("[train] spawning SGLang supervisor child process...")
            self.sglang_mgr.start()
            logger.info(f"[train] SGLang supervisor ready (PID={self.sglang_mgr.pid()})")

        # 首步 weight sync：确保 sglang 用的是当前 LoRA 状态（而非旧 run 的 collapsed 权重）
        # --fresh 时 LoRA=0，merge 后 = base model → sglang 重置到 base
        if self._current_step == 0:
            logger.info("Initial weight sync: resetting sglang to current LoRA state")
            self._sync_lora_to_vllm(step=-1)

        # 异步 rollout 状态
        _next_future: Future = None
        _async_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async_rollout")

        for step in range(self._current_step, self.max_steps):
            self._current_step = step

            # 1. 获取 episodes（首步串行，后续从后台线程取）
            import time as _time
            _t0 = _time.time()
            if _next_future is not None:
                trajectories = _next_future.result()
                _next_future = None
            else:
                batch = self._sample_batch_for_step(step)
                trajectories = self._collect_episodes(batch)

            _t1 = _time.time()
            total_turns = sum(len(t.turns) for t in trajectories)
            logger.info(f"Episodes collected: {len(trajectories)} trajs, {total_turns} turns in {_t1-_t0:.1f}s")

            # 2.5 LoRA sync 必须在 async rollout 之前执行
            # 原因: unload adapter 需要等所有使用该 adapter 的请求完成
            # 如果 rollout 已启动 → 28 个请求阻塞 unload → 15s stall
            if step % self.sync_lora_every == 0:
                self._sync_lora_to_vllm(step)

            # 2.6 启动下一步 rollout（sync 完成后，用最新权重）
            if step + 1 < self.max_steps and _next_future is None:
                _next_batch = self._sample_batch_for_step(step + 1)
                _next_future = _async_pool.submit(self._collect_episodes, _next_batch)
                logger.info(f"  Async: step {step+1} rollout launched (overlaps with GPU compute)")

            # 3. Logprobs (GPU 1) + KL ref (GPU 3) 并行
            logger.info("Computing logprobs (GPU 1) + KL ref (GPU 3) in parallel...")
            self._kl_ref_logprobs = None
            _kl_ref_future = None
            kl_coeff = self.config.get("kl_coeff", 0.01)
            if kl_coeff > 0 and self.replica_model:
                from concurrent.futures import ThreadPoolExecutor as _TP
                _kl_ref_future = _TP(max_workers=1).submit(
                    self._compute_kl_ref_on_replica, trajectories)

            self._fill_turn_logprobs_no_grad(trajectories)

            if _kl_ref_future:
                self._kl_ref_logprobs = _kl_ref_future.result()

            _t2 = _time.time()
            logger.info(f"Logprobs + KL ref done in {_t2-_t1:.1f}s")

            # 4. Z_θ（no_grad，用于 flow 填充和日志，不保留计算图）
            with torch.no_grad():
                log_z = self._compute_partition_function(trajectories)
            log_z_floats = log_z.detach().tolist()

            # 5. 填充 flow 字段
            for i, traj in enumerate(trajectories):
                fill_turn_flows(traj, log_z_floats[i])

            # 6. Per-trajectory gradient accumulation
            #    每条轨迹独立计算 TTB loss 并立即 backward（释放计算图），
            #    避免同时保留 batch_size × max_turns 个前向传播图导致 OOM。
            logger.info("Gradient accumulation...")
            loss_val = self._gradient_accumulation_step(trajectories)
            self._plateau_detector.update(step, loss_val)
            _t3 = _time.time()
            logger.info(f"Gradient done in {_t3-_t2:.1f}s")
            loss = torch.tensor(loss_val, device=self.device, dtype=torch.float32)

            # 8. 更新 flow 度量
            self._update_flow_metrics(trajectories, log_z)

            # 8b. Flow-driven observation collection（零成本）
            n_obs = self._observation_buffer.collect(trajectories)
            if n_obs > 0:
                logger.debug(f"[ObsBuffer] Collected {n_obs} I(t)<{self._observation_buffer._threshold} observations, total={self._observation_buffer.total_count()}")

            # 8c. Per-type balance tracking（进化触发信号）
            self._update_per_type_balance(trajectories)

            # 9. 记录统计
            stats = self._collect_stats(step, loss, trajectories, log_z)
            self._log_step(stats)

            # 9a. DAG 质量监控：同问题多轨迹的奖励对比
            from collections import defaultdict as _ddict
            _q_groups = _ddict(list)
            for _t in trajectories:
                _q_groups[_t.question[:80]].append(_t.r_tilde)
            _dag_info = []
            for _qk, _rs in _q_groups.items():
                if len(_rs) > 1:
                    _spread = max(_rs) - min(_rs)
                    _dag_info.append(f"{_qk[:30]}...(n={len(_rs)},spread={_spread:.2f})")
            if _dag_info:
                logger.info(f"  DAG groups ({len(_dag_info)}): {'; '.join(_dag_info[:3])}")

            # 9b. 每5步 dump 一条轨迹样本（用于验证多轮交互）
            if step % 5 == 0:
                self._dump_sample_trajectory(step, trajectories)

            # 9c. 验证集评估（暂时关闭，加速训练）
            # if step % 5 == 0 and self._val_data:
            #     self._run_validation(step)

            # 10. Weight sync 已移至 step 2.5（rollout 之前执行，避免 unload 阻塞）

            # 11. 异步 rollout 已在 step 2.5 提前启动（和 GPU 计算完全重叠）

            # 12. Flow-driven per-type tip evolution（v2，和 rollout 并行执行）
            self._try_evolve(step)

            # 13. 保存 checkpoint
            if step > 0 and step % self.save_every == 0:
                self._save_checkpoint(step)

            acc_summary = self.accuracy_tracker.summary() if hasattr(self, 'accuracy_tracker') else ""
            logger.info(
                f"Step {step:04d} | loss={loss.item():.4f} | "
                f"reward={stats['avg_reward']:.3f} | "
                f"ans={stats['avg_answer']:.3f} | "
                f"H_flow={stats.get('flow_entropy', 0):.3f} | "
                f"H_skill={stats.get('h_skill_ratio', 0):.3f} | "
                f"skills={self.workspace.size} | "
                f"acc=[{acc_summary}]"
            )

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time/3600:.1f}h")
        self._save_checkpoint(self.max_steps, final=True)

    # ──────────────────────────────────────────────
    # Episode 收集
    # ──────────────────────────────────────────────

    def _collect_episodes(self, batch: List[Dict]) -> List[Trajectory]:
        """
        FlowSteer 风格并行 rollout + 独立 episode 完成。

        核心设计（参考 FlowSteer train_interactive.py）：
        - Phase 1: 所有 active episodes 的 Supervisor 调用并行发送
        - Phase 2: 所有 active episodes 的 env.step() 并行执行（含 M_exec）
        - 每个 episode 独立完成（done 后退出），不等其他 episode
        - vLLM continuous batching 在同一 round 收到批量请求，GPU 利用率最高
        """
        n = len(batch)
        max_workers = min(n, self.config.get("rollout_workers", 24))

        # 日志：打印所有 episode 的任务信息
        for ep_idx, question in enumerate(batch):
            source = question.get("extra", {}).get("source", question.get("task_type", "?"))
            logger.info(f"  Episode {ep_idx+1}/{n}: {source} | q={str(question.get('question',''))[:60]}...")

        # 并行执行所有 episodes
        results: Dict[int, Trajectory] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for ep_idx, question in enumerate(batch):
                fut = pool.submit(self._run_episode, question)
                futures[fut] = ep_idx

            for fut in as_completed(futures):
                ep_idx = futures[fut]
                try:
                    traj = fut.result()
                except Exception as e:
                    logger.error(f"  Episode {ep_idx+1} crashed: {e}")
                    # 创建一个失败轨迹
                    traj = Trajectory(
                        question=str(batch[ep_idx].get("question", "")),
                        gold_answer=str(batch[ep_idx].get("answer", "")),
                        task_type=str(batch[ep_idx].get("task_type", "unknown")),
                    )
                    traj.reward = 0.0  # 崩溃 episode 的 reward（无 BASE_PENALTY）
                    traj.completed = True
                results[ep_idx] = traj

        # 按原始顺序排列（保持 batch 内 episode 顺序确定性）
        trajectories = [results[i] for i in range(n)]

        for ep_idx, traj in enumerate(trajectories):
            self._phase_trajectories.append(traj)
            # Per-task-type 准确率追踪（OpenClaw-RL 风格）
            if hasattr(self, 'accuracy_tracker') and traj.task_type:
                correct = traj.reward >= 0.5  # 有 R_answer > 0 才算正确（无 BASE_PENALTY）
                self.accuracy_tracker.track_result(traj.task_type, correct)
            logger.info(f"  Episode {ep_idx+1} done: {len(traj.turns)} turns, reward={traj.reward:.3f}")

        return trajectories

    def _run_episode(self, question: Dict) -> Trajectory:
        """
        运行单个 episode（线程安全，通过 vLLM API 调用 Supervisor）。

        每次调用创建独立的 GenericTaskEnvironment 实例，
        避免多线程共享状态冲突。
        """
        task_type = str(question.get("task_type", "unknown"))

        # ═══ ReAct 路径：WebShop / ALFWorld（SkillRL 格式）═══
        if task_type in ("webshop", "alfworld"):
            return self._run_react_episode(question)

        # ═══ 现有 Tool Calling 路径（其他 task_type 不变）═══
        # 推导 g_q（episode-level goal）
        episode_goal = self._derive_episode_goal(question)

        # Experience 不注入 prompt — 只注入 non-obvious、minimal 的信息
        # experience_store 和 experience_buffer 的 prompt 注入已移除
        # 原因：experience 膨胀 prompt (~230 extra words)，导致模型把 prompt 扔给 ask_llm
        # experience_store 仍用于 reward shaping（compute_experience_reward，max bonus 0.1）

        # 每个 episode 创建独立 env（线程安全的关键）
        # per-task max_episode_steps（QA 5-6步，SWE/ALFWorld 15步）
        from training.task_prompts import TASK_CONFIGS
        task_type = str(question.get("task_type", "unknown"))
        _task_cfg = TASK_CONFIGS.get(task_type, {})
        _ep_steps = _task_cfg.get("max_episode_steps", self.max_episode_steps)

        env = GenericTaskEnvironment(
            m_exec=self.m_exec,
            max_episode_steps=_ep_steps,
            epsilon_min=self.epsilon_min,
            skill_workspace=self.workspace,
            experience_store=None,  # 仅用于 reward shaping
            reward_mode=self.config.get("reward_mode", "outcome_only"),
            skill_mode=self.config.get("skill_mode", "policy_action"),
        )

        messages, traj = env.reset(
            question,
            episode_goal=episode_goal,
        )
        traj.task_type_id = TASK_TYPE_TO_ID.get(question.get("task_type", "unknown"), 9)
        # v6: Use task-type-filtered tools from env
        tools = env._tools

        from training.batch_inference import supervisor_call
        max_retries = 3
        import time as _time
        _episode_start = _time.time()
        _episode_timeout = 600  # 10 min wall-clock timeout

        # Selective thinking: only for tasks that benefit from extended reasoning
        _thinking_types = set()  # disabled: thinking 不影响训练但增加推理时间
        _use_thinking = task_type in _thinking_types

        # v7: 动态过滤 answer tool - 只有用过至少 1 个非 answer/skill_invoke 工具后
        # 才提供 answer tool 给模型选择. 这是根本方案阻止"直接答题"退化轨迹.
        def _tools_for_step(base_tools, trajectory):
            used_real_tool = any(
                getattr(t, 'action_type', '') not in ('answer', 'skill_invoke', '', 'parse_error')
                for t in trajectory.turns
            )
            if used_real_tool:
                return base_tools  # 已用工具, 开放 answer
            # 没用过工具 → 移除 answer tool (若存在)
            return [t for t in base_tools if t.get('function', {}).get('name') != 'answer']

        for step_idx in range(_ep_steps):
            if _time.time() - _episode_start > _episode_timeout:
                logger.warning(f"    [S{step_idx+1}] Episode wall-clock timeout ({_episode_timeout}s)")
                break
            # Supervisor 生成（multi-turn tool calling via vLLM API）
            content, tool_name, tool_args = "", None, None
            _tools_this_step = _tools_for_step(tools, traj)
            for attempt in range(max_retries):
                try:
                    content, tool_name, tool_args = supervisor_call(
                        messages=messages,
                        api_base=self.config["supervisor_api_base"],
                        model=self.config.get("supervisor_model", "Qwen3-8B"),
                        tools=_tools_this_step,
                        max_tokens=512,
                        temperature=0.8,
                        enable_thinking=_use_thinking,
                    )
                    break
                except Exception as e:
                    logger.warning(f"vLLM call failed (attempt {attempt+1}): {e}")
                    content, tool_name, tool_args = "", None, None

            import time as _time
            _step_start = _time.time()
            reward, done, info = env.step(content, tool_name, tool_args, traj)
            _step_elapsed = _time.time() - _step_start
            _action_type = tool_name or "answer"

            # 每步完整日志（不截断）
            _obs = info.get("observation", "")
            _args_str = json.dumps(tool_args, ensure_ascii=False) if tool_args else ""
            logger.info(
                f"    [S{step_idx+1}] tool={_action_type} | "
                f"args={_args_str} | obs_len={len(_obs)} | "
                f"content_len={len(content)} | done={done} | {_step_elapsed:.1f}s"
            )
            logger.debug(
                f"    [S{step_idx+1}] CONTENT:\n{content}\n"
                f"    OBS:\n{_obs}\n"
                f"    ARGS:\n{_args_str}"
            )
            if _step_elapsed > 5.0:
                logger.warning(f"    Slow step {step_idx}: action={_action_type}, took {_step_elapsed:.1f}s")

            # v6: Populate supervisor_input with chat-template-rendered text
            # (messages_snapshot is already set by env.step; render it to text here)
            if traj.turns and traj.turns[-1].messages_snapshot:
                try:
                    traj.turns[-1].supervisor_input = self._messages_to_text(
                        traj.turns[-1].messages_snapshot
                    )
                except Exception:
                    pass  # Keep the text fallback from env.step

            if done:
                break

        # Experience 功能已移除

        return traj

    def _run_react_episode(self, question: Dict) -> Trajectory:
        """ReAct path for WebShop/ALFWorld (SkillRL format)."""
        from training.batch_inference import react_call
        from training.task_prompts import TASK_CONFIGS

        task_type = str(question.get("task_type", "unknown"))
        _task_cfg = TASK_CONFIGS.get(task_type, {})
        _ep_steps = _task_cfg.get("max_episode_steps", self.max_episode_steps)

        env = GenericTaskEnvironment(
            m_exec=self.m_exec,
            max_episode_steps=_ep_steps,
            epsilon_min=self.epsilon_min,
            skill_workspace=self.workspace,
            reward_mode=self.config.get("reward_mode", "outcome_only"),
            skill_mode=self.config.get("skill_mode", "policy_action"),
        )

        prompt, traj = env.reset_react(question)
        traj.task_type_id = TASK_TYPE_TO_ID.get(question.get("task_type", "unknown"), 9)

        import time as _time
        _episode_start = _time.time()
        _episode_timeout = 600  # 10 min wall-clock timeout per episode

        for step_idx in range(_ep_steps):
            # Wall-clock 超时保护
            if _time.time() - _episode_start > _episode_timeout:
                logger.warning(f"    [R{step_idx+1}] Episode wall-clock timeout ({_episode_timeout}s)")
                break

            full_content, action_str = "", None
            for attempt in range(3):
                try:
                    full_content, action_str = react_call(
                        prompt=prompt,
                        api_base=self.config["supervisor_api_base"],
                        model=self.config.get("supervisor_model", "Qwen3-8B"),
                        max_tokens=512,
                        temperature=0.8,
                    )
                    break
                except Exception as e:
                    logger.warning(f"react_call failed (attempt {attempt+1}): {e}")

            _t0 = _time.time()
            prompt, reward, done, info = env.react_step(full_content, action_str, traj)
            _elapsed = _time.time() - _t0

            _action = action_str or "invalid"
            logger.info(f"    [R{step_idx+1}] action={_action[:50]} | done={done} | {_elapsed:.1f}s")

            if done:
                break

        traj.completed = True
        return traj

    # ──────────────────────────────────────────────
    # Gradient Accumulation（OOM Fix）
    # ──────────────────────────────────────────────

    def _fill_turn_logprobs_no_grad(self, trajectories: List[Trajectory]) -> None:
        """
        填充每个 turn 的 forward/backward logprob floats（no_grad，用于 flow 计算）。

        v9: GPU 1 本地批量计算 forward + backward logprobs。
        共享模型已在 GPU 1 上，切换 adapter 即可：
          - θ-LoRA → forward logprobs (batch)
          - φ-LoRA → backward logprobs (batch)
        无 SGLang API 调用，无网络开销。
        """
        from training.backward_policy import split_think_and_action

        # ── 收集所有 turns 的 token 数据 ──
        all_items = []  # (traj_idx, turn_idx, full_ids_tensor, ctx_len, n_act)
        for i, traj in enumerate(trajectories):
            for j, turn in enumerate(traj.turns):
                # 仅 legacy auto-injected skill marker 是虚拟追踪 turn，不参与 logprob。
                # paper-aligned policy-selected skill_invoke has real supervisor
                # input/output and must be scored like any other action.
                if (
                    getattr(turn, 'action_type', '') == 'skill_invoke'
                    and not getattr(turn, 'supervisor_input', '')
                    and not getattr(turn, 'messages_snapshot', None)
                ):
                    turn.forward_logprob = 0.0
                    turn.backward_logprob = 0.0
                    turn.action_token_count = 0
                    continue
                context = turn.supervisor_input
                if turn.messages_snapshot:
                    try:
                        context = self._messages_to_text(turn.messages_snapshot)
                    except Exception:
                        pass
                item = self._prepare_turn_tokens(context, turn.supervisor_output)
                if item is not None:
                    full_ids_1d, ctx_len = item
                    n_act = full_ids_1d.shape[0] - ctx_len
                    all_items.append((i, j, full_ids_1d, ctx_len, n_act))
                else:
                    traj.turns[j].forward_logprob = 0.0
                    traj.turns[j].action_token_count = 0

        if not all_items:
            return

        # micro_bs=8 是验证过的稳定值。
        # (曾试 16 OOM — base model 77GB + logits [16×4096×150K vocab×2B]=19.7GB 超 80GB GPU)
        # fallback 会把 forward_logprob 置 0 污染训练信号, 故必须稳妥。
        micro_bs = 8
        max_len = 4096  # 截断长序列

        # ── 强制关闭 gradient checkpointing（logprob 不需要，开着会拖慢 forward）──
        # 每次强制设置，不依赖状态检查（weight sync 的 merge_adapter 会破坏 GC 状态）
        _base_model = getattr(self.shared_model, 'base_model', self.shared_model)
        _inner_model = getattr(_base_model, 'model', _base_model)
        _inner_model.gradient_checkpointing_disable()

        # ── Forward logprobs（θ-LoRA，本地批量）──
        self.shared_model.set_adapter("theta")
        self.shared_model.eval()

        with torch.no_grad():
            for start in range(0, len(all_items), micro_bs):
                batch = all_items[start:start + micro_bs]

                # Pad to same length
                batch_max_len = min(max(item[2].shape[0] for item in batch), max_len)
                padded_ids = torch.zeros(len(batch), batch_max_len, dtype=torch.long, device=self.device)
                attention_mask = torch.zeros(len(batch), batch_max_len, dtype=torch.long, device=self.device)

                for bi, (ti, tj, full_ids, ctx_len, n_act) in enumerate(batch):
                    seq_len = min(full_ids.shape[0], batch_max_len)
                    # 如果超长，保留尾部（context 尾部 + 全部 action）
                    if full_ids.shape[0] > batch_max_len:
                        full_ids = full_ids[-batch_max_len:]
                        # 调整 ctx_len
                        batch[bi] = (ti, tj, full_ids, max(0, seq_len - n_act), n_act)
                    padded_ids[bi, :seq_len] = full_ids[:seq_len].to(self.device)
                    attention_mask[bi, :seq_len] = 1

                try:
                    outputs = self.shared_model(padded_ids, attention_mask=attention_mask)
                    logits = outputs.logits  # [bs, seq_len, vocab_size]

                    for bi, (ti, tj, full_ids, ctx_len, n_act) in enumerate(batch):
                        seq_len = min(full_ids.shape[0], batch_max_len)
                        act_start = ctx_len
                        act_end = min(ctx_len + n_act, seq_len)

                        if act_start >= seq_len or act_end <= act_start:
                            trajectories[ti].turns[tj].forward_logprob = 0.0
                            trajectories[ti].turns[tj].action_token_count = n_act
                            continue

                        action_logits = logits[bi, act_start - 1:act_end - 1, :]
                        action_targets = padded_ids[bi, act_start:act_end]
                        log_probs = torch.log_softmax(action_logits, dim=-1)
                        token_lps = log_probs[
                            torch.arange(action_targets.shape[0], device=self.device),
                            action_targets,
                        ]
                        trajectories[ti].turns[tj].forward_logprob = token_lps.sum().item()
                        trajectories[ti].turns[tj].action_token_count = n_act
                    del outputs, logits  # 释放 GPU 显存
                except Exception as e:
                    logger.warning(f"Batch forward logprob failed: {e}")
                    for bi, (ti, tj, _, _, n_act) in enumerate(batch):
                        trajectories[ti].turns[tj].forward_logprob = 0.0
                        trajectories[ti].turns[tj].action_token_count = n_act
                del padded_ids, attention_mask  # 释放 batch tensor

        # ── Backward logprobs（φ-LoRA，本地批量）──
        bwd_float_lists = self.backward_policy.compute_logprobs_batch(trajectories)
        for i, traj in enumerate(trajectories):
            bwd_per = bwd_float_lists[i]
            for j, turn in enumerate(traj.turns):
                turn.backward_logprob = bwd_per[j] if j < len(bwd_per) else 0.0

        # 恢复 theta adapter + 强制开启 gradient checkpointing + 清理显存
        self.shared_model.set_adapter("theta")
        self.shared_model.train()
        _inner_model.gradient_checkpointing_enable()
        # Bug fix: set_adapter("theta") 关闭了 phi 的 requires_grad，必须恢复
        for _n, _p in self.shared_model.named_parameters():
            if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                _p.requires_grad_(True)
        torch.cuda.empty_cache()

    def _prepare_turn_tokens(self, context_text: str, action_text: str):
        """Pre-tokenize a turn's (context, action) pair for batched processing.

        Returns (full_ids_1d, ctx_len) or None if empty.
        """
        think_part, json_part = split_think_and_action(action_text)
        if json_part:
            eff_ctx = context_text + think_part
            tgt = json_part
        else:
            eff_ctx = context_text
            tgt = action_text

        ctx_ids = self.tokenizer.encode(eff_ctx, add_special_tokens=False)
        act_ids = self.tokenizer.encode(tgt, add_special_tokens=False)
        if not act_ids:
            return None

        full = ctx_ids + act_ids
        ctx_len = len(ctx_ids)

        max_len = 4096  # 平衡覆盖率与显存：大部分 turn < 2048 tokens
        if len(full) > max_len:
            keep = max(64, max_len - len(act_ids) - 10)  # 至少保留 64 token context
            full = ctx_ids[-keep:] + act_ids if keep > 0 else act_ids
            ctx_len = keep if keep > 0 else 0

        if ctx_len == 0 or not act_ids:
            return None
        return (torch.tensor(full, dtype=torch.long, device=self.device), ctx_len)

    def _compute_kl_ref_on_replica(self, trajectories):
        """GPU 3 上计算 KL ref logprobs（和 GPU 1 的 logprobs 并行）。
        返回 list[float]，对应每个非虚拟 turn（policy-selected skill_invoke 也计入）。"""
        from training.backward_policy import split_think_and_action
        device = torch.device(self.extra_device)
        self.replica_model.disable_adapter_layers()
        ref_lps = []
        # micro_bs=8 稳定值（16 会 OOM on GPU 3 同样理由）
        micro_bs = 8
        # 收集所有 turns 的 tokens
        all_items = []
        for traj in trajectories:
            for turn in traj.turns:
                if (
                    getattr(turn, 'action_type', '') == 'skill_invoke'
                    and not getattr(turn, 'supervisor_input', '')
                    and not getattr(turn, 'messages_snapshot', None)
                ):
                    continue
                ctx = turn.supervisor_input
                if turn.messages_snapshot:
                    try: ctx = self._messages_to_text(turn.messages_snapshot)
                    except: pass
                item = self._prepare_turn_tokens(ctx, turn.supervisor_output)
                if item is not None:
                    all_items.append(item)
                else:
                    all_items.append(None)
        # 批量 forward
        valid = [(i, it) for i, it in enumerate(all_items) if it is not None]
        ref_map = {}
        with torch.no_grad():
            for s in range(0, len(valid), micro_bs):
                batch = valid[s:s+micro_bs]
                mx = max(it[1][0].shape[0] for it in batch)
                bs = len(batch)
                ids = torch.zeros(bs, mx, dtype=torch.long, device=device)
                mask = torch.zeros(bs, mx, dtype=torch.long, device=device)
                for j, (_, (tok, _)) in enumerate(batch):
                    sl = tok.shape[0]
                    ids[j,:sl] = tok.to(device)
                    mask[j,:sl] = 1
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.replica_model(input_ids=ids, attention_mask=mask).logits
                for j, (orig_idx, (tok, cl)) in enumerate(batch):
                    sl = tok.shape[0]
                    if cl >= sl:
                        ref_map[orig_idx] = 0.0
                        continue
                    lp = torch.log_softmax(logits[j, cl-1:sl-1, :], dim=-1)
                    tgt = ids[j, cl:sl]
                    ref_map[orig_idx] = lp[torch.arange(tgt.shape[0], device=device), tgt].sum().item()
                del logits
        self.replica_model.enable_adapter_layers()
        self.replica_model.set_adapter("theta")
        # 只返回 valid items 的 ref logprobs（和 theta_items 一一对应）
        return [ref_map[orig_idx] for orig_idx, _ in valid]

    def _sync_replica_weights(self):
        if not self.replica_model: return
        src = {n: p.data for n,p in self.shared_model.named_parameters() if "lora_" in n}
        for n,p in self.replica_model.named_parameters():
            if n in src: p.data.copy_(src[n].to(p.device))

    def _average_replica_grads(self):
        if not self.replica_model: return
        rg = {n: p.grad.data for n,p in self.replica_model.named_parameters()
              if "lora_" in n and p.grad is not None}
        for n,p in self.shared_model.named_parameters():
            if "lora_" in n and n in rg:
                g = rg[n].to(p.device)
                if p.grad is not None: p.grad.data.add_(g).mul_(0.5)
                else: p.grad = g.clone().mul_(0.5)
        for p in self.replica_model.parameters():
            if p.grad is not None: p.grad.zero_()

    @staticmethod
    def _run_micro_batches(items, model, device, micro_bs, kl_coeff=0.0, batch_size=1):
        """Forward+backward with optional KL merged into loss (零额外开销)。
        items: 3-tuple (ids, ctx_len, gs) 或 4-tuple (ids, ctx_len, gs, ref_lp)。
        """
        for start in range(0, len(items), micro_bs):
            batch = items[start:start + micro_bs]
            max_seq = max(it[0].shape[0] for it in batch)
            bs = len(batch)
            input_ids = torch.zeros(bs, max_seq, dtype=torch.long, device=device)
            attn_mask = torch.zeros(bs, max_seq, dtype=torch.long, device=device)
            for j in range(bs):
                sl = batch[j][0].shape[0]
                input_ids[j, :sl] = batch[j][0].to(device)
                attn_mask[j, :sl] = 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            micro_loss = None
            for j in range(bs):
                ids, ctx_len, gs = batch[j][0], batch[j][1], batch[j][2]
                ref_lp = batch[j][3] if len(batch[j]) > 3 else None
                sl = ids.shape[0]
                if ctx_len >= sl: continue
                a_logits = logits[j, ctx_len - 1: sl - 1, :]
                a_targets = input_ids[j, ctx_len: sl]
                lp = torch.log_softmax(a_logits, dim=-1)
                tok_lp = lp[torch.arange(a_targets.shape[0], device=device), a_targets]
                term = gs * tok_lp.sum()
                # KL gradient 合并: 一次 forward+backward 同时算 TTB + KL
                if ref_lp is not None and kl_coeff > 0:
                    n_tok = max(a_targets.shape[0], 1)
                    term = term + kl_coeff * (tok_lp.sum() - ref_lp) / n_tok / batch_size
                micro_loss = term if micro_loss is None else micro_loss + term
            if micro_loss is not None:
                micro_loss.backward()
            del micro_loss, logits

    def _batched_logprob_backward(self, items, adapter_name: str, micro_bs: int = 4):
        """Data parallel backward + KL merged: GPU 1 + GPU 3 via threading.

        KL gradient 已合并进 loss（4-tuple items 含 ref_lp）→ 无单独 KL phase。
        micro_bs=8: 每卡 checkpointing 峰值 ~65GB < 80GB。
        """
        if not items:
            return

        kl_coeff = self.config.get("kl_coeff", 0.0)
        batch_size = self.config.get("batch_size", 28)
        self.shared_model.set_adapter(adapter_name)

        if self.replica_model is None:
            self._run_micro_batches(items, self.shared_model, self.device, micro_bs,
                                    kl_coeff, batch_size)
            return

        # ── Dual GPU data parallel ──
        from threading import Thread
        self.replica_model.set_adapter(adapter_name)
        for _n, _p in self.replica_model.named_parameters():
            if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                _p.requires_grad_(True)

        mid = len(items) // 2
        errors = [None, None]

        def _safe_run(idx, items_slice, model, device):
            try:
                self._run_micro_batches(items_slice, model, device, micro_bs,
                                        kl_coeff, batch_size)
            except Exception as e:
                errors[idx] = e
                logger.error(f"Thread-{idx} error: {e}")

        t1 = Thread(target=_safe_run, args=(0, items[:mid], self.shared_model, self.device))
        t2 = Thread(target=_safe_run, args=(1, items[mid:], self.replica_model,
                                             torch.device(self.extra_device)))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        if errors[0]:
            logger.error(f"GPU 1 failed: {errors[0]}. Retrying full batch on GPU 1.")
            # GPU1 失败 → 梯度不完整，需要重跑全部
            for p in self.shared_model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            self._run_micro_batches(items, self.shared_model, self.device, micro_bs,
                                    kl_coeff, batch_size)
        elif errors[1]:
            logger.warning(f"GPU 3 failed: {errors[1]}. Fallback to GPU 1.")
            self._run_micro_batches(items[mid:], self.shared_model, self.device, micro_bs,
                                    kl_coeff, batch_size)
        else:
            self._average_replica_grads()

    def _kl_gradient_with_refs(self, theta_items, ref_logprobs, kl_coeff, batch_size):
        """KL gradient 用预算好的 ref_logprobs（只需一次 θ forward+backward）。"""
        micro_bs = 4  # backward 有梯度，4 更安全
        kl_total = 0.0
        self.shared_model.set_adapter("theta")
        for start in range(0, len(theta_items), micro_bs):
            batch = theta_items[start:start + micro_bs]
            refs = ref_logprobs[start:start + len(batch)]
            max_seq = max(it[0].shape[0] for it in batch)
            bs = len(batch)
            input_ids = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
            attn_mask = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
            for j in range(bs):
                sl = batch[j][0].shape[0]
                input_ids[j, :sl] = batch[j][0]
                attn_mask[j, :sl] = 1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.supervisor_model(input_ids=input_ids, attention_mask=attn_mask).logits
            kl_loss = None
            for j in range(bs):
                ids, ctx_len, _ = batch[j][0], batch[j][1], batch[j][2]
                sl = ids.shape[0]
                if ctx_len >= sl: continue
                a_logits = logits[j, ctx_len-1:sl-1, :]
                a_targets = input_ids[j, ctx_len:sl]
                lp = torch.log_softmax(a_logits, dim=-1)
                tok_lp = lp[torch.arange(a_targets.shape[0], device=self.device), a_targets]
                n_tok = max(a_targets.shape[0], 1)
                kl_term = (tok_lp.sum() - refs[j]) / n_tok
                kl_total += kl_term.item()
                scaled = kl_coeff * kl_term / batch_size
                kl_loss = scaled if kl_loss is None else kl_loss + scaled
            if kl_loss is not None:
                kl_loss.backward()
            del kl_loss, logits
            torch.cuda.empty_cache()
        avg_kl = kl_total / max(len(theta_items), 1)
        logger.info(f"  KL penalty: avg_kl={avg_kl:.4f}, coeff={kl_coeff} (ref from GPU 3)")

    def _compute_kl_penalty(self, theta_items, kl_coeff: float, batch_size: int):
        """KL(π_θ || π_ref) penalty — fallback 串行版。

        Flow of Reasoning (NeurIPS 2024): GFlowNet 需要 KL 正则化防止 π_θ/P_φ 共同退化。

        实现：disable LoRA = π_ref (base model)。
        只需 ref logprobs（常数），θ logprob 已在 Phase 2 中计算了梯度。
        额外梯度：∂KL/∂θ = kl_coeff × ∂log π_θ/∂θ（和 Phase 2 方向相同但量级不同）。
        """
        micro_bs = 4  # backward 有梯度，4 更安全

        # Step 1: 计算 ref logprobs（base model, no LoRA, no_grad）
        ref_logprobs = []
        self.shared_model.disable_adapter_layers()
        with torch.no_grad():
            for start in range(0, len(theta_items), micro_bs):
                batch = theta_items[start:start + micro_bs]
                max_seq = max(it[0].shape[0] for it in batch)
                bs = len(batch)

                input_ids = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
                attn_mask = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
                for j, (ids, _, _) in enumerate(batch):
                    sl = ids.shape[0]
                    input_ids[j, :sl] = ids
                    attn_mask[j, :sl] = 1

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.shared_model.model(
                        input_ids=input_ids, attention_mask=attn_mask
                    ).logits

                for j, (ids, ctx_len, _) in enumerate(batch):
                    sl = ids.shape[0]
                    if ctx_len >= sl:
                        ref_logprobs.append(0.0)
                        continue
                    a_logits = logits[j, ctx_len - 1: sl - 1, :]
                    a_targets = input_ids[j, ctx_len: sl]
                    lp = torch.log_softmax(a_logits, dim=-1)
                    tok_lp = lp[torch.arange(a_targets.shape[0], device=self.device), a_targets]
                    ref_logprobs.append(tok_lp.sum().item())
                del logits
                torch.cuda.empty_cache()

        self.shared_model.enable_adapter_layers()
        self.shared_model.set_adapter("theta")

        # Step 2: KL backward — 用 θ-LoRA 的 logprobs 减去 ref logprobs
        # KL = E[log π_θ - log π_ref]，梯度只需 ∂log π_θ/∂θ
        kl_total = 0.0
        for start in range(0, len(theta_items), micro_bs):
            batch_items = theta_items[start:start + micro_bs]
            batch_refs = ref_logprobs[start:start + len(batch_items)]
            max_seq = max(it[0].shape[0] for it in batch_items)
            bs = len(batch_items)

            input_ids = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
            attn_mask = torch.zeros(bs, max_seq, dtype=torch.long, device=self.device)
            for j, (ids, _, _) in enumerate(batch_items):
                sl = ids.shape[0]
                input_ids[j, :sl] = ids
                attn_mask[j, :sl] = 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.supervisor_model(
                    input_ids=input_ids, attention_mask=attn_mask
                ).logits

            kl_loss = None
            for j, ((ids, ctx_len, _), ref_lp) in enumerate(zip(batch_items, batch_refs)):
                sl = ids.shape[0]
                if ctx_len >= sl:
                    continue
                a_logits = logits[j, ctx_len - 1: sl - 1, :]
                a_targets = input_ids[j, ctx_len: sl]
                lp = torch.log_softmax(a_logits, dim=-1)
                tok_lp = lp[torch.arange(a_targets.shape[0], device=self.device), a_targets]
                # KL for this turn = (θ_logprob - ref_logprob) / n_tokens
                n_tok = a_targets.shape[0]
                kl_term = (tok_lp.sum() - ref_lp) / max(n_tok, 1)
                kl_total += kl_term.item()
                scaled = kl_coeff * kl_term / batch_size
                kl_loss = scaled if kl_loss is None else kl_loss + scaled

            if kl_loss is not None:
                kl_loss.backward()
            del kl_loss, logits
            torch.cuda.empty_cache()

        avg_kl = kl_total / max(len(theta_items), 1)
        logger.info(f"  KL penalty: avg_kl={avg_kl:.4f}, coeff={kl_coeff}, loss_contrib={avg_kl*kl_coeff:.6f}")

    def _gradient_accumulation_step(self, trajectories: List[Trajectory]) -> float:
        """
        Batched TTB gradient accumulation — veRL/OpenRLHF 风格 micro-batch。

        Phase 1: 预计算 balance/grad_scale + Z head backward（轻量）
        Phase 2: 批量 θ forward+backward（micro-batch）
        Phase 3: 批量 φ forward+backward（micro-batch）
        Phase 4: Optimizer step

        关键：theta 和 phi 的激活不同时存在 → 峰值显存从
        O(T_theta + T_phi) 降为 O(max(T_theta, T_phi))。

        已在 _fill_turn_logprobs_no_grad 中预计算的 turn.forward_logprob/backward_logprob
        用于计算 balance（不需要重过模型）；gradient backward 只需 θ/φ 各一次前向。
        """
        self._supervisor_optimizer.zero_grad()
        self._partition_optimizer.zero_grad()
        if self._phi_optimizer is not None:
            self._phi_optimizer.zero_grad()
            # Bug fix: 确保 phi requires_grad 在每步开始时都是开启的
            # set_adapter("theta") 会关闭它，多处调用后可能遗留为 False
            for _n, _p in self.shared_model.named_parameters():
                if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                    _p.requires_grad_(True)

        total_loss = 0.0
        n = len(trajectories)

        # v7: K<3 过滤器已移除 — answer 变成正规工具 + 动态过滤后, 退化轨迹根本不会产生。
        # 所有 trajectories 的 tool call JSON 结构保证 K ≥ 15 per turn。

        # ── Phase 1: 预计算 balance/grad_scale，收集 turn items，Z backward ──
        theta_items = []  # (full_ids_1d, ctx_len, grad_scale)
        phi_items = []

        for i, traj in enumerate(trajectories):
            # ── Paper-aligned TTB Loss (论文 Eq. 13 + Appendix gradient) ──
            #
            # Stored turn logprobs are token-sums from the LM scorer.  The paper
            # defines each edge by per-token tempering:
            #   log π̃(a_t) = (1/K_t) Σ_j log π(tok_j), same for P̃_φ.
            # Therefore Δ and every flow metric must use edge_logprob_tilde().
            # Loss: L(τ) = (Δ(τ) / T)², T = number of scored policy edges.
            # Gradient coefficient before ∇Δ is 2Δ/T² (then averaged over batch).
            fwd_sum_det = sum(edge_logprob_tilde(t, "forward") for t in traj.turns)
            bwd_sum_det = sum(edge_logprob_tilde(t, "backward") for t in traj.turns)
            r_tilde_val = max(traj.r_tilde, self.epsilon_min)
            log_r = float(torch.log(torch.tensor(r_tilde_val)).item())

            # Paper: Z_θ(q) is conditioned on the task query q, not the rendered
            # system/tool prompt.  Keep this pure so partition learning matches
            # the written objective.
            query_text = traj.question[:512]
            task_type_id_val = traj.task_type_id

            query_hidden_det = self._get_query_hidden(query_text)
            task_type_id = torch.tensor([task_type_id_val], device=self.device, dtype=torch.long)
            with torch.no_grad():
                log_z_det = self.partition_fn(query_hidden_det.unsqueeze(0), task_type_id).item()

            # Δ(τ) = log Z + Σ_t log π̃_θ(a_t) − β·log R̃ − Σ_t log P̃_φ(a_t)
            delta_raw = log_z_det + fwd_sum_det - self.beta * log_r - bwd_sum_det
            n_steps = effective_paper_steps(traj)
            # L_TTB = (Δ / T)²
            balance = delta_raw / n_steps
            # 稳定性 clamp：只限制极端 early-training 梯度；正常区间仍是 2Δ/T²。
            balance = max(min(balance, 10.0), -10.0)
            loss_i = balance ** 2
            total_loss += loss_i

            if i < 2:
                logger.info(
                    f"  TTB[{i}]: log_z={log_z_det:.2f} fwd={fwd_sum_det:.2f} "
                    f"bwd={bwd_sum_det:.2f} log_r={log_r:.2f} "
                    f"Δ={delta_raw:.2f} T={n_steps} "
                    f"Δ/T={balance:.4f} loss_i={loss_i:.4f}"
                )

            # Paper gradient: ∂L/∂Δ = 2Δ/T² = 2(Δ/T)/T.
            # For token-sum LM backward, ∂ log π̃ / ∂ token_sum = 1/K_t,
            # so θ/φ turn gradients get an extra per-turn 1/K_t below.
            z_grad_scale = (2.0 * balance) / (n_steps * n)

            # 收集 θ turns — per-token-tempered edge gradient includes /K_t.
            for turn in traj.turns:
                if (
                    getattr(turn, 'action_type', '') == 'skill_invoke'
                    and not getattr(turn, 'supervisor_input', '')
                    and not getattr(turn, 'messages_snapshot', None)
                ):
                    continue
                ctx = turn.supervisor_input
                if turn.messages_snapshot:
                    try:
                        ctx = self._messages_to_text(turn.messages_snapshot)
                    except Exception:
                        pass
                item = self._prepare_turn_tokens(ctx, turn.supervisor_output)
                if item is not None:
                    k_t = max(int(getattr(turn, "action_token_count", 0) or 0), 1)
                    theta_items.append((*item, z_grad_scale / k_t))

            # 收集 φ turns — backward policy 梯度（符号相反）
            if self._phi_optimizer is not None:
                for t_idx, turn in enumerate(traj.turns):
                    if (
                        getattr(turn, 'action_type', '') == 'skill_invoke'
                        and not getattr(turn, 'supervisor_input', '')
                        and not getattr(turn, 'messages_snapshot', None)
                    ):
                        continue
                    bwd_text = traj.to_backward_text_per_turn(t_idx)
                    item = self._prepare_turn_tokens(bwd_text, turn.supervisor_output)
                    if item is not None:
                        k_t = max(int(getattr(turn, "action_token_count", 0) or 0), 1)
                        phi_items.append((*item, -z_grad_scale / k_t))

            # Z head backward（per-trajectory，极轻量）
            log_z_grad = self.partition_fn(query_hidden_det.unsqueeze(0), task_type_id).squeeze()
            log_z_grad.backward(torch.tensor(z_grad_scale, device=self.device, dtype=torch.float32))

        # ── KL ref logprobs 已在 logprobs 阶段并行计算（_kl_ref_logprobs）──
        kl_coeff = self.config.get("kl_coeff", 0.01)
        ref_logprobs = getattr(self, '_kl_ref_logprobs', None)
        self._kl_ref_logprobs = None

        # ── Phase 2: θ backward + KL merged (GPU 1+3 parallel, micro_bs=8) ──
        # KL gradient 已合并进 loss（4-tuple theta_items 含 ref_lp）
        # → 无单独 KL phase → 可以用更大 micro_bs
        _micro_bs = 4  # backward 有梯度，micro_bs=4 更安全（8 会 OOM on long sequences）
        if ref_logprobs and len(ref_logprobs) == len(theta_items):
            theta_items = [(*item, ref_lp) for item, ref_lp in zip(theta_items, ref_logprobs)]
            logger.info(f"  Batched θ backward: {len(theta_items)} turns, micro_bs={_micro_bs} (KL merged)")
        else:
            logger.info(f"  Batched θ backward: {len(theta_items)} turns, micro_bs={_micro_bs}")
        # 清理 logprobs 阶段的残留显存
        torch.cuda.empty_cache()
        self._batched_logprob_backward(theta_items, "theta", micro_bs=_micro_bs)

        # ── Phase 3: φ backward (GPU 1+3 parallel) ──
        if phi_items:
            torch.cuda.empty_cache()
            logger.info(f"  Batched φ backward: {len(phi_items)} turns, micro_bs={_micro_bs}")
            self._batched_logprob_backward(phi_items, "phi", micro_bs=_micro_bs)
            self.shared_model.set_adapter("theta")

        # Grad clipping & optimizer step
        # theta params
        theta_params = [
            p for n, p in self.shared_model.named_parameters()
            if p.requires_grad and (".lora_A.theta" in n or ".lora_B.theta" in n)
        ] or [p for p in self.shared_model.parameters() if p.requires_grad]
        max_gn = getattr(self, "_max_grad_norm", 1.0)
        theta_grad_norm = torch.nn.utils.clip_grad_norm_(theta_params, max_gn)
        self._supervisor_optimizer.step()

        z_grad_norm = torch.nn.utils.clip_grad_norm_(self.partition_fn.parameters(), max_gn)
        self._partition_optimizer.step()

        phi_grad_norm = 0.0
        if self._phi_optimizer is not None:
            # NOTE: 不用 requires_grad 过滤——set_adapter("theta") 会关闭 phi 的 requires_grad，
            # 但 .grad 张量在 backward 阶段已经正确累积。按名称匹配即可。
            phi_params = [
                p for n, p in self.shared_model.named_parameters()
                if ".lora_A.phi" in n or ".lora_B.phi" in n
            ]
            phi_with_grad = [p for p in phi_params if p.grad is not None]
            if phi_with_grad:
                phi_grad_norm = torch.nn.utils.clip_grad_norm_(phi_with_grad, max_gn)
            self._phi_optimizer.step()

        logger.info(
            f"  Grad norms: θ={theta_grad_norm:.4f} Z={z_grad_norm:.4f} φ={phi_grad_norm:.4f} "
            f"(clip={max_gn})"
        )

        # Sync updated LoRA weights to replica
        self._sync_replica_weights()

        return total_loss / max(n, 1)  # mean over batch（论文 Eq.9 的 1/B 归一化）

    # ──────────────────────────────────────────────
    # Logprob 计算
    # ──────────────────────────────────────────────

    def _compute_logprobs(
        self,
        trajectories: List[Trajectory],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 forward（θ-LoRA）和 backward（φ-LoRA）logprob sums。

        关键：backward logprobs 通过 compute_logprobs_with_grad() 计算，保留计算图，
        使得 L_TTB.backward() 能同时更新 θ（forward）和 φ（backward）。

        Returns:
            (fwd_sum [B], bwd_sum [B])，两者均保留梯度
        """
        fwd_sums = []
        bwd_tensor_sums = []  # 保留梯度，供 L_TTB backward 使用

        # Forward logprobs（保留梯度，供 θ 更新）
        for traj in trajectories:
            fwd_sum = self._compute_forward_logprob_sum(traj)
            fwd_sums.append(fwd_sum)

        # Backward logprobs（保留梯度，供 φ 通过 L_TTB 更新）
        bwd_sum_tensors = self.backward_policy.compute_logprobs_with_grad(trajectories)
        bwd_tensor_sums = bwd_sum_tensors  # List[Tensor], 每个 shape []

        # 填充每个 turn 的 forward/backward logprobs（float，用于 flow 计算，不需梯度）
        fwd_per_turn_lists = []
        with torch.no_grad():
            for traj in trajectories:
                fwd_per_turn_lists.append(self._compute_forward_logprob_per_turn(traj))

        # backward per-turn 也单独用 no_grad 版本填充 float（只用于 flow metrics）
        bwd_float_lists = self.backward_policy.compute_logprobs_batch(trajectories)

        for i, traj in enumerate(trajectories):
            fwd_per = fwd_per_turn_lists[i]
            bwd_per = bwd_float_lists[i]
            for j, turn in enumerate(traj.turns):
                if j < len(fwd_per):
                    turn.forward_logprob = fwd_per[j][0]
                    turn.action_token_count = fwd_per[j][1]
                else:
                    turn.forward_logprob = 0.0
                    turn.action_token_count = 0
                turn.backward_logprob = bwd_per[j] if j < len(bwd_per) else 0.0

        fwd_tensor = torch.stack(fwd_sums)
        bwd_tensor = torch.stack(bwd_tensor_sums)
        return fwd_tensor, bwd_tensor

    def _compute_forward_logprob_sum(self, traj: Trajectory) -> torch.Tensor:
        """计算单条轨迹所有步骤的 forward logprob sum（保留梯度）"""
        total = torch.zeros(1, device=self.device, requires_grad=True)

        for turn in traj.turns:
            # v6: Use messages_snapshot → chat template text for proper context
            context = turn.supervisor_input
            if turn.messages_snapshot:
                try:
                    context = self._messages_to_text(turn.messages_snapshot)
                except Exception:
                    pass  # Fall back to existing supervisor_input
            lp = self._compute_action_logprob_forward(
                context, turn.supervisor_output
            )
            total = total + lp

        return total.squeeze()

    def _compute_forward_logprob_per_turn(self, traj: Trajectory) -> List[tuple]:
        """计算每个 turn 的 forward logprob 和 action token 数（不含梯度，用于 flow 计算）。

        Returns:
            List[(logprob_float, token_count_int)] — 每个 turn 的 (log π_θ, |a_t|_tokens)
        """
        with torch.no_grad():
            result = []
            for turn in traj.turns:
                # v6: Use messages_snapshot → chat template text for proper context
                context = turn.supervisor_input
                if turn.messages_snapshot:
                    try:
                        context = self._messages_to_text(turn.messages_snapshot)
                    except Exception:
                        pass  # Fall back to existing supervisor_input
                lp, n_tokens = self._compute_action_logprob_forward(
                    context, turn.supervisor_output,
                    return_token_count=True,
                )
                result.append((lp.item(), n_tokens))
        return result

    def _compute_action_logprob_forward(
        self, context_text: str, action_text: str,
        return_token_count: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, int]":
        """
        π_θ 的 action logprob（θ-LoRA）。

        论文 §3.2：a_t = (α_t, a_t^out)，不含 a_t^think。
        Qwen3-8B 输出 = <think>...</think> + JSON 动作两部分：
          - think 部分 (a_t^think)：并入 context，teacher-forcing 时模型能看到，
            但不进入 log-sum——避免 thinking token 污染 TTB balance error。
          - JSON 部分 (a_t^out)：唯一的 logprob target，计入 Σ log π_θ。

        若 JSON 部分为空（think 块被截断 / 无输出），降级到对整个 action_text 计算，
        防止 logprob=0 导致 TTB loss 退化。
        """
        # ── M6 fix: 拆分 think 和 JSON 动作部分 ──────────────────────────────
        think_part, json_part = split_think_and_action(action_text)
        if json_part:
            # 正常路径：think 扩展进 context，仅对 JSON 计算 logprob
            effective_context = context_text + think_part
            target_text = json_part
        else:
            # 降级：无法找到 JSON（截断 / 无 thinking 模式）→ 对整个 action_text 计算
            effective_context = context_text
            target_text = action_text
        # ──────────────────────────────────────────────────────────────────────

        ctx_ids = self.tokenizer.encode(
            effective_context, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        act_ids = self.tokenizer.encode(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        n_action_tokens = act_ids.shape[1]

        if n_action_tokens == 0:
            zero = torch.zeros(1, device=self.device)
            return (zero, 0) if return_token_count else zero

        full_ids = torch.cat([ctx_ids, act_ids], dim=1)
        ctx_len = ctx_ids.shape[1]

        # 截断（保留 context 尾部 + 全部 target）
        # 使用 1024 降低 GPU 显存峰值（GPU 与 vLLM 共享时需要更紧凑的内存）
        max_len = 4096
        if full_ids.shape[1] > max_len:
            keep_ctx = max(0, max_len - act_ids.shape[1] - 10)
            ctx_ids = ctx_ids[:, -keep_ctx:] if keep_ctx > 0 else ctx_ids[:, :0]
            full_ids = torch.cat([ctx_ids, act_ids], dim=1)
            ctx_len = ctx_ids.shape[1]

        if ctx_len == 0 or act_ids.shape[1] == 0:
            zero = torch.zeros(1, device=self.device)
            return (zero, 0) if return_token_count else zero

        # 确保 theta 适配器激活（共享模型时 backward_policy 可能切换过适配器）
        self.shared_model.set_adapter("theta")

        # 直接前向传播（不用 gradient checkpointing）
        # GPU 2 有 63GB 空闲，无需省激活内存；去掉 ckpt 避免 forward 执行两遍
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.supervisor_model(full_ids).logits

        action_logits = logits[0, ctx_len - 1: -1, :]
        action_targets = full_ids[0, ctx_len:]

        if action_targets.shape[0] == 0:
            zero = torch.zeros(1, device=self.device)
            return (zero, 0) if return_token_count else zero

        log_probs = torch.log_softmax(action_logits, dim=-1)
        token_logprobs = log_probs[
            torch.arange(action_targets.shape[0], device=self.device),
            action_targets,
        ]
        logprob_sum = token_logprobs.sum()
        if return_token_count:
            return logprob_sum, n_action_tokens
        return logprob_sum

    def _messages_to_text(self, messages: List[Dict]) -> str:
        """Convert multi-turn messages to text using chat template for logprob computation.

        v6: Uses the tokenizer's chat_template to render the full message history
        (system + user + assistant + tool messages) into a single text string.
        This produces the exact token sequence the model saw during generation,
        ensuring logprob context alignment between rollout and training.
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ──────────────────────────────────────────────
    # Partition Function
    # ──────────────────────────────────────────────

    def _compute_partition_function(self, trajectories: List[Trajectory]) -> torch.Tensor:
        """
        计算 batch 中每个 query 的 log Z_θ(q)。

        使用 Supervisor 对纯 question q 的最后 token hidden state 作为 query 表示。
        """
        log_z_list = []

        for traj in trajectories:
            # Paper: Z_θ(q) is query-conditioned.  Do not include system prompt,
            # retrieved skill catalog, or trajectory history here.
            query_text = traj.question

            query_hidden = self._get_query_hidden(query_text)
            task_type_id = torch.tensor(
                [traj.task_type_id], device=self.device, dtype=torch.long
            )
            log_z = self.partition_fn(query_hidden.unsqueeze(0), task_type_id)
            log_z_list.append(log_z)

        return torch.cat(log_z_list)

    def _get_query_hidden(self, text: str) -> torch.Tensor:
        """获取文本的 last token hidden state（使用 theta 适配器）"""
        ids = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt",
            max_length=512, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            self.shared_model.set_adapter("theta")
            outputs = self.supervisor_model(ids, output_hidden_states=True)
            # 最后一层的最后一个 token
            last_hidden = outputs.hidden_states[-1][0, -1, :]

        return last_hidden.detach()

    # ──────────────────────────────────────────────
    # Backward + Optimizer
    # ──────────────────────────────────────────────

    def _backward_step(
        self,
        loss: torch.Tensor,
        bwd_logprob_sums: torch.Tensor,  # 保留签名兼容，但不再单独处理
    ) -> None:
        """
        TTB loss 反向传播，联合更新 θ-LoRA + φ-LoRA + Z_θ head。

        论文 Eq.12："optimize (θ, φ) via L_TTB"
        L_TTB 同时包含 Σ log π_θ（θ 梯度）和 Σ log P_φ（φ 梯度），
        一次 loss.backward() 同时更新所有三个可训练组件。
        """
        self._supervisor_optimizer.zero_grad()
        self._partition_optimizer.zero_grad()
        if self._phi_optimizer is not None:
            self._phi_optimizer.zero_grad()

        loss.backward()

        max_gn = getattr(self, "_max_grad_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(self.supervisor_model.parameters(), max_gn)
        torch.nn.utils.clip_grad_norm_(self.partition_fn.parameters(), max_gn)
        if self.backward_policy.model is not None:
            torch.nn.utils.clip_grad_norm_(self.backward_policy.model.parameters(), max_gn)

        self._supervisor_optimizer.step()
        self._partition_optimizer.step()
        if self._phi_optimizer is not None:
            self._phi_optimizer.step()

    def _setup_optimizers(self) -> None:
        """设置各组件的优化器"""
        lr = self.config.get("learning_rate", 1e-4)
        phi_lr_ratio = self.config.get("phi_lr_ratio", 0.5)
        self._max_grad_norm = self.config.get("max_grad_norm", 1.0)

        # θ-LoRA 参数（共享模型中名称含 "lora_A.theta" 或 "lora_B.theta" 的参数）
        theta_params = [
            p for n, p in self.shared_model.named_parameters()
            if p.requires_grad and (".lora_A.theta" in n or ".lora_B.theta" in n)
        ]
        if not theta_params:
            # fallback：无法按名字过滤时取全部 requires_grad 参数
            logger.warning("Cannot filter theta params by name, using all trainable params")
            theta_params = [p for p in self.shared_model.parameters() if p.requires_grad]

        self._supervisor_optimizer = torch.optim.AdamW(theta_params, lr=lr, weight_decay=0.01)
        logger.info(f"θ-LoRA optimizer: {len(theta_params)} param groups")

        # Z_θ head（更大 lr，因为是新初始化的）
        self._partition_optimizer = torch.optim.AdamW(
            self.partition_fn.parameters(), lr=lr * 1.5, weight_decay=0.01  # Z head lr: 从 3× 降到 1.5×（防止 log_Z 漂移）
        )

        # φ-LoRA 参数（按名称匹配，不依赖 requires_grad 状态——
        # 因为 set_adapter("theta") 会把 phi 的 requires_grad 关掉，
        # 我们在 setup() 中已手动重新开启，但这里按名称选择更安全）
        phi_params = [
            p for n, p in self.shared_model.named_parameters()
            if ".lora_A.phi" in n or ".lora_B.phi" in n
        ]
        # 确保这些参数的 requires_grad=True
        for p in phi_params:
            p.requires_grad_(True)
        if phi_params:
            self._phi_optimizer = torch.optim.AdamW(
                phi_params, lr=lr * phi_lr_ratio, weight_decay=0.01
            )
            logger.info(f"φ-LoRA optimizer: {len(phi_params)} param groups")
        else:
            logger.warning("No φ-LoRA params found, phi optimizer disabled")
            self._phi_optimizer = None

    # ──────────────────────────────────────────────
    # Flow 度量 & 进化
    # ──────────────────────────────────────────────

    def _update_flow_metrics(
        self,
        trajectories: List[Trajectory],
        log_z: torch.Tensor,
    ) -> None:
        """更新技能的 flow 统计（usage count + F̂(s) + Var_skill(s)）"""
        all_skill_ids = self.workspace.get_all_ids()

        # 计算 F̂(s)（论文 Eq.7）
        skill_marginal_flows = compute_skill_marginal_flows(trajectories, all_skill_ids)

        # 将批次计算的 log F̂(s) 写回 workspace（EMA 平滑），用于检索排序和进化触发
        # 注：flow 现在是 log 空间 [-20, 20]，-20 表示未使用
        for sid, flow in skill_marginal_flows.items():
            if flow > -20.0:  # 只更新本批次有调用的 skill
                self.workspace.update_flow_score(sid, flow)

        # 计算并更新 Var_skill(s) — 论文 Eq.11
        skill_variances = compute_skill_variances(trajectories, all_skill_ids)
        for sid, var in skill_variances.items():
            # 只要有任何调用就更新方差（包括 var==0 的单条轨迹情况）
            if skill_variances[sid] > 0.0 or any(
                turn.skill_id == sid
                for traj in trajectories
                for turn in traj.turns
                if turn.action_type == "skill_invoke"
            ):
                self.workspace.update_reward_variance(sid, var)

        # 更新 workspace 使用统计
        for traj in trajectories:
            success = traj.reward > 0.5
            for sid in traj.skills_invoked:
                self.workspace.record_usage(sid, self._current_step, success=success)

        self._current_marginal_flows = skill_marginal_flows

    def _try_extract_experiences(self, step: int, trajectories: List[Trajectory]) -> None:
        """
        XSkill 风格：每步检查 contrast signal，有则提取 experiences。

        Contrast signal = batch 中同时有高分和低分轨迹。
        SkillRL 要求 10-90% success rate，我们类似。
        """
        if not self.experience_store:
            return

        high = [t for t in trajectories if t.reward >= 0.5]  # 正确的（R_answer ≈ 1.0 + R_process ≈ 0.2）
        low = [t for t in trajectories if t.reward <= 0.1]   # 失败的（R_answer = 0, R_process ≈ 0.1）

        # 需要 contrast signal（既有成功又有失败）
        if not high or not low:
            return

        success_rate = len(high) / len(trajectories)
        if success_rate < 0.1 or success_rate > 0.9:
            return  # 太不平衡，跳过

        # 用 cross-trajectory critique 提取 experiences
        try:
            critique_exps = self.skill_creator.cross_trajectory_critique(
                high_flow_trajs=high[:3],
                low_flow_trajs=low[:3],
            )
            if critique_exps:
                from training.experience_store import Experience
                avg_reward = sum(t.reward for t in high) / len(high)
                for exp_dict in critique_exps:
                    exp = Experience(
                        condition=exp_dict["condition"],
                        action=exp_dict["action"],
                        task_types=exp_dict.get("task_types", []),
                        reward_signal=avg_reward,
                        source_step=step,
                    )
                    self.experience_store.add(exp)
                logger.info(
                    f"[Experience] Step {step}: extracted {len(critique_exps)} "
                    f"experiences (contrast: {len(high)}hi/{len(low)}lo)"
                )
        except Exception as e:
            logger.warning(f"[Experience] Extraction failed at step {step}: {e}")

    def _update_per_type_balance(self, trajectories: List[Trajectory]) -> None:
        """追踪每个 task_type 的 TTB balance error 历史（进化触发信号）。"""
        from collections import defaultdict
        by_type = defaultdict(list)
        beta = self.config.get("beta", 1.0)
        for traj in trajectories:
            fwd_sum = sum(edge_logprob_tilde(t, "forward") for t in traj.turns)
            bwd_sum = sum(edge_logprob_tilde(t, "backward") for t in traj.turns)
            log_R_beta = beta * math.log(max(traj.r_tilde, 1e-10))
            delta = traj.log_z + fwd_sum - log_R_beta - bwd_sum
            be = abs(delta / effective_paper_steps(traj))
            by_type[traj.task_type].append(be)

        for tt, errors in by_type.items():
            avg_be = sum(errors) / len(errors)
            self._per_type_balance_history[tt].append(avg_be)
            # 保留最近 10 步
            if len(self._per_type_balance_history[tt]) > 10:
                self._per_type_balance_history[tt] = self._per_type_balance_history[tt][-10:]

    def _get_struggling_types(self, step: int) -> List[str]:
        """Paper Definition F.1 phase trigger.

        The paper triggers skill evolution at a phase boundary when the running
        mean squared TTB residual plateaus.  Task selection here is intentionally
        minimal: after a plateau, evolve every task type represented in the
        recent phase buffer that has enough trajectories.  We do not add the
        earlier engineering accuracy/cooldown heuristics, so the active trigger
        matches the appendix semantics.
        """
        from collections import defaultdict

        max_skills = self.config.get("max_skills_total", 60)
        n_traj = self.config.get("n_trajectories_per_question", 4)

        if self.workspace.size >= max_skills:
            return []

        if not self._plateau_detector.should_trigger(step):
            logger.info("[Evolution] Plateau check: no phase boundary")
            return []

        recent = self._phase_trajectories[-(self.batch_size * 5):]
        by_type: Dict[str, List[Trajectory]] = defaultdict(list)
        for traj in recent:
            by_type[traj.task_type].append(traj)

        struggling = [
            tt for tt, trajs in sorted(by_type.items())
            if len(trajs) >= max(n_traj, 2)
        ]
        logger.info(
            "[Evolution] Plateau trigger at step %s: task_types=%s",
            step, struggling
        )
        return struggling

    def _try_evolve(self, step: int) -> None:
        """论文 §4.4 / Appendix F：plateau-triggered skill evolution.

        触发：PlateauDetector 监控 batch-mean TTB residual squared。
        诊断：same-query success/failure pairs + log I(t) trigger steps。
        Curation：CGF 的 D⁻/R/U 划分约束 Skill Creator 的 prune/refine/add。
        """
        from training.flow_metrics import (
            split_by_flow_quartile,
            compute_flow_bottlenecks,
            extract_counterfactual_pairs,
            build_bottleneck_diagnoses,
        )
        from training.skill_evolution import partition_skills_DRU
        from collections import defaultdict

        if not self._phase_trajectories:
            return

        struggling = self._get_struggling_types(step)
        if not struggling:
            return

        logger.info(f"[Evolution v5] Step {step}: struggling types = {struggling}")

        # ── Step 1 (全局一次): Flow Residual → bottleneck clusters ──
        try:
            bottleneck_map = compute_flow_bottlenecks(
                self._phase_trajectories,
                bucket_size=2,
                min_samples=4,
                var_threshold=2.0,
                mean_threshold=1.0,
            )
        except Exception as e:
            logger.warning(f"[Evolution v5] compute_flow_bottlenecks failed: {e}")
            bottleneck_map = {}

        # ── Step 2 (全局一次): 同问题 Counterfactual pair 提取 ──
        try:
            cf_map = extract_counterfactual_pairs(
                self._phase_trajectories,
                min_reward_gap=0.3,
            )
        except Exception as e:
            logger.warning(f"[Evolution v5] extract_counterfactual_pairs failed: {e}")
            cf_map = {}

        # ── Optional: SkillAcceptanceGate (config-gated) ──
        gate = None
        gate_n = int(self.config.get("acceptance_gate_episodes", 0) or 0)
        if gate_n > 0:
            try:
                from training.skill_evolution import SkillAcceptanceGate
                gate = SkillAcceptanceGate(
                    run_episode_fn=None,  # lightweight sanity mode (no rollout)
                    n_eval_episodes=gate_n,
                    min_improvement=float(self.config.get("acceptance_gate_threshold", 0.05)),
                )
            except Exception as e:
                logger.warning(f"[Evolution v5] Could not initialize gate: {e}")

        skills_updated = False

        for tt in struggling:
            type_trajs = [t for t in self._phase_trajectories if t.task_type == tt]
            if len(type_trajs) < 4:
                continue

            # ── flow quartile 分 ──
            low_flow, high_flow = split_by_flow_quartile(type_trajs)

            # ── Initial trigger candidates; recomputed after CGF curation below
            # with the paper coverage test t ∉ cov(R ∪ U').
            critical_steps = []
            for traj in high_flow:
                for i, turn in enumerate(traj.turns):
                    log_i = edge_log_i(turn)
                    if log_i >= float(self.config.get("zeta_trig", 1.0)):
                        critical_steps.append({
                            'step': i + 1,
                            'action': (getattr(turn, 'instruction', '') or ''),
                            'log_I_t': round(log_i, 3),
                            'I_t': round(getattr(turn, 'step_importance', 0), 2),
                            'observation': (getattr(turn, 'observation', '') or ''),
                            'from_reward': round(traj.r_tilde, 2),
                        })

            dag_comparisons = []
            by_question = defaultdict(list)
            for t in type_trajs:
                by_question[t.question].append(t)
            for q_key, q_trajs in by_question.items():
                if len(q_trajs) < 2:
                    continue
                q_trajs.sort(key=lambda t: t.r_tilde, reverse=True)
                best, worst = q_trajs[0], q_trajs[-1]
                if best.r_tilde - worst.r_tilde > 0.3:
                    dag_comparisons.append({
                        'question': q_key,
                        'success_actions': [(getattr(t, 'instruction', '') or '') for t in best.turns],
                        'failure_actions': [(getattr(t, 'instruction', '') or '') for t in worst.turns],
                        'reward_gap': round(best.r_tilde - worst.r_tilde, 2),
                    })

            # ── Step 3: 构造结构化诊断 ──
            existing_skills_for_tt = []
            try:
                if hasattr(self.workspace, 'get_by_task_type'):
                    existing_skills_for_tt = self.workspace.get_by_task_type(tt)
                else:
                    existing_skills_for_tt = [
                        s for s in self.workspace.get_all()
                        if tt in (getattr(s.meta, 'task_types', None) or [])
                    ]
            except Exception:
                existing_skills_for_tt = []

            bottlenecks_tt = bottleneck_map.get(tt, [])
            cf_pairs_tt = cf_map.get(tt, [])
            curation = None
            try:
                all_skill_ids = [
                    getattr(s.meta, "skill_id", "")
                    for s in existing_skills_for_tt
                    if getattr(s, "meta", None) is not None and getattr(s.meta, "skill_id", "")
                ]
                curation = partition_skills_DRU(
                    trajectories=type_trajs,
                    skill_ids=all_skill_ids,
                    n_minus_counter=self._skill_negative_counter,
                    G_threshold=float(self.config.get("cgf_G_threshold", 0.0)),
                    J_threshold=float(self.config.get("cgf_J_threshold", 1.0)),
                    K_minus=int(self.config.get("cgf_K_minus", 2)),
                )
                logger.info(
                    f"[Evolution CGF] {tt}: D-={len(curation.prune)}, "
                    f"R={len(curation.retain)}, U={len(curation.refine)}"
                )
            except Exception as e:
                logger.warning(f"[Evolution CGF] partition_skills_DRU({tt}) failed: {e}")
                curation = None

            # Paper Definition F.4: trigger steps from successful/high-flow
            # trajectories must satisfy log I(t) >= ζ_trig and not be covered by
            # retained/refined skills.  If a retained/refined skill was invoked
            # earlier in the same trajectory, treat subsequent steps as covered.
            zeta_trig = float(self.config.get("zeta_trig", 1.0))
            covered_skill_ids = set()
            if curation is not None:
                covered_skill_ids = set(curation.retain) | set(curation.refine)
            critical_steps = []
            for traj in high_flow:
                covered = False
                for i, turn in enumerate(traj.turns):
                    if getattr(turn, "action_type", "") == "skill_invoke":
                        if getattr(turn, "skill_id", None) in covered_skill_ids:
                            covered = True
                        continue
                    if covered:
                        continue
                    log_i = edge_log_i(turn)
                    if log_i >= zeta_trig:
                        critical_steps.append({
                            "step": i + 1,
                            "action": (getattr(turn, "instruction", "") or ""),
                            "log_I_t": round(log_i, 3),
                            "I_t": round(getattr(turn, "step_importance", 0), 2),
                            "observation": (getattr(turn, "observation", "") or ""),
                            "from_reward": round(traj.r_tilde, 2),
                        })

            diagnoses = []
            if bottlenecks_tt:
                try:
                    diagnoses = build_bottleneck_diagnoses(
                        bottlenecks=bottlenecks_tt,
                        counterfactual_pairs=cf_pairs_tt,
                        existing_skills=existing_skills_for_tt,
                        task_type=tt,
                    )
                except Exception as e:
                    logger.warning(f"[Evolution v5] build_bottleneck_diagnoses({tt}) failed: {e}")
                    diagnoses = []

            logger.info(
                f"[Evolution v5] {tt}: bottlenecks={len(bottlenecks_tt)}, "
                f"cf_pairs={len(cf_pairs_tt)}, diagnoses={len(diagnoses)}"
            )

            # ── Step 4: 调用进化 ──
            # Paper path: CGF curation class D⁻/R/U constrains existing skills;
            # same-query comparisons and uncovered trigger steps provide evidence
            # to Ψ for refining U and generating new skills.
            ids_before = set(self.workspace.get_all_ids()) if self.workspace else set()
            new_tips = self.evolution_manager.evolve_for_type(
                task_type=tt,
                observations=self._observation_buffer.get(tt) or [],
                failed_trajectories=low_flow[:3],
                step=step,
                high_flow_trajs=high_flow[:3],
                critical_steps=critical_steps[:5],
                dag_comparisons=dag_comparisons[:3],
                bottleneck_diagnoses=None,  # v5.1: 禁用 3+2+1, 走 v3 fallback
                counterfactual_pairs=None,
                acceptance_gate=gate,
                curation=curation,
            )
            ids_after = set(self.workspace.get_all_ids()) if self.workspace else set()

            if new_tips or ids_before != ids_after:
                skills_updated = True
                self._observation_buffer.clear(tt)
                self._per_type_last_evolution[tt] = step
                logger.info(
                    f"[Evolution v5] {tt}: generated {len(new_tips)} tips "
                    f"(diagnoses={len(diagnoses)}, cf_pairs={len(cf_pairs_tt)}, "
                    f"I(t) steps={len(critical_steps)}, DAG pairs={len(dag_comparisons)}), "
                    f"workspace={self.workspace.size}"
                )

        if skills_updated:
            self._skills_just_updated = True
            self._reset_partition_function()

        if struggling:
            recent_count = self.batch_size * 5
            if len(self._phase_trajectories) > recent_count:
                self._phase_trajectories = self._phase_trajectories[-recent_count:]

    def _reset_partition_function(self) -> None:
        """
        技能库更新后按论文 Algorithm 1 重新初始化 Z_θ(q)。

        论文步骤 8 写的是：warm-start π_θ/P_φ, reinitialize the partition
        function Z_θ(q) for the new action space. 因此这里不做额外工程化
        warm-up，只重置 partition head，使下一 phase 从新的 Z 初始化开始学习。
        """
        if self.partition_fn is None:
            self._skills_just_updated = False
            return

        try:
            self.partition_fn.task_embed.reset_parameters()
            nn.init.zeros_(self.partition_fn.head.weight)
            nn.init.constant_(
                self.partition_fn.head.bias,
                math.log(max(float(self.epsilon_min), 1e-8)),
            )
        finally:
            self._skills_just_updated = False

        # set_adapter side effects elsewhere may disable φ-LoRA gradients; keep them
        # enabled for the next TTB update.
        phi_grad_count = 0
        if self.shared_model is not None:
            for name, param in self.shared_model.named_parameters():
                if ".lora_A.phi" in name or ".lora_B.phi" in name:
                    param.requires_grad_(True)
                    phi_grad_count += 1

        logger.info(
            f"[Z_θ reset] partition function reinitialized after skill update "
            f"(skills: {self.workspace.size}, re-enabled {phi_grad_count} φ-LoRA params)"
        )

    # ──────────────────────────────────────────────
    # 跨 Episode 经验管理（M_mem 的 E 组件）
    # ──────────────────────────────────────────────

    def _derive_episode_goal(self, question: Dict) -> str:
        """
        推导 g_q（episode-level goal）。

        论文 §3.3：g_q 为本 episode 的目标描述，帮助 Supervisor 了解成功标准。
        与 q 的区别：q 是具体问题，g_q 是元目标（"如何解题"而不是"解什么题"）。

        推导依据（不硬编码任务类型→目标的映射表）：
          1. 当前技能库中 flow 最高的 2 个 skill 名称（反映已有能力）
          2. 任务类型（用于描述成功标准）
          3. 问题特征（问题长度、是否包含数字/代码等）
        """
        task_type = question.get("task_type", "unknown")
        q_text = str(question.get("question", ""))

        # 从 skill 库中找出当前 flow 最高的 skills（反映已有能力方向）
        top_skills_info = ""
        if self.workspace and self.workspace.size > 0:
            top_skills = self.workspace.retrieve(q_text, task_type=task_type, top_k=2)
            if top_skills:
                skill_names = [s.name for s in top_skills]
                top_skills_info = f" Relevant skills: {', '.join(skill_names)}."

        # 根据问题特征推断期望的答案形式
        q_lower = q_text.lower()
        if any(kw in q_lower for kw in ["true", "false", "supports", "refutes"]):
            expected_form = "a factual verdict (supports/refutes/not enough info)"
        elif any(kw in q_lower for kw in ["calculate", "how many", "what is the value"]):
            expected_form = "a precise numerical answer"
        elif any(kw in q_lower for kw in ["def ", "function", "implement", "code"]):
            expected_form = "working code"
        elif any(kw in q_lower for kw in ["yes", "no", "would", "could", "did"]):
            expected_form = "a direct yes/no answer with brief justification"
        else:
            expected_form = "a concise, accurate answer"

        # Goal 极简，不加通用建议（"decompose" 已在 system prompt 中）
        return f"Produce {expected_form}."

    def _retrieve_experience(self, question: Dict, top_k: int = 2) -> List[Dict]:
        """
        从经验缓冲中检索与当前问题相似的高奖励经验（E_ret）。

        检索方式：优先 embedding 语义相似度（复用 workspace 的 bge 模型），
        回退到 Jaccard + task_type bonus。
        用于注入 H_0，帮助 Supervisor 了解类似任务的成功策略。
        线程安全：读取 experience_buffer 时加锁。
        """
        with self._experience_lock:
            if not self._experience_buffer:
                return []
            # 快照（避免长时间持锁）
            buffer_snapshot = list(self._experience_buffer)

        task_type = question.get("task_type", "")
        q_text = str(question.get("question", ""))

        # 尝试 embedding 检索（复用 workspace 的 bge 模型）
        try:
            from src.skills.workspace import _get_embedding_model
            model = _get_embedding_model()
            if model is not None and len(buffer_snapshot) >= 3:
                import numpy as np
                q_emb = model.encode([q_text], normalize_embeddings=True)
                exp_texts = [exp.get("question_summary", "") for exp in buffer_snapshot]
                exp_embs = model.encode(exp_texts, normalize_embeddings=True)
                sims = np.dot(exp_embs, q_emb.T).squeeze()  # (N,)

                scored: List[Tuple[float, Dict]] = []
                for i, exp in enumerate(buffer_snapshot):
                    type_bonus = 0.15 if exp.get("task_type") == task_type else 0.0
                    score = float(sims[i]) + type_bonus
                    scored.append((score, exp))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [exp for score, exp in scored[:top_k] if score > 0.3]
        except Exception:
            pass

        # Fallback: Jaccard + task_type bonus
        q_words = set(q_text.lower().split())
        scored: List[Tuple[float, Dict]] = []
        for exp in buffer_snapshot:
            type_bonus = 0.3 if exp.get("task_type") == task_type else 0.0
            exp_words = set(exp.get("question_summary", "").lower().split())
            if q_words and exp_words:
                jaccard = len(q_words & exp_words) / len(q_words | exp_words)
            else:
                jaccard = 0.0
            score = jaccard + type_bonus
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:top_k] if scored[0][0] > 0.05]

    def _store_experience(self, traj: Trajectory) -> None:
        """
        将高奖励轨迹摘要存入 experience buffer。

        仅存储摘要（不存完整轨迹），控制内存占用。
        """
        # 提取 key steps 摘要（最高 step_importance 的步骤）
        key_steps = []
        if traj.turns:
            sorted_turns = sorted(traj.turns, key=lambda t: t.step_importance, reverse=True)
            for turn in sorted_turns[:2]:
                step_desc = f"{turn.action_type}"
                if turn.skill_id:
                    step_desc += f"[{turn.skill_id}]"
                if turn.instruction:
                    step_desc += f": {turn.instruction[:80]}"
                key_steps.append(step_desc)

        summary = {
            "traj_id": traj.traj_id,
            "task_type": traj.task_type,
            "question_summary": traj.question[:120],
            "skills_invoked": traj.unique_skills,
            "tools_used": list(dict.fromkeys(traj.tools_used)),  # 去重保序
            "key_steps_summary": " → ".join(key_steps),
            "reward": round(traj.reward, 3),
            "n_steps": traj.n_steps,
        }

        with self._experience_lock:
            self._experience_buffer.append(summary)

            # 超出容量时，移除最低奖励的经验
            if len(self._experience_buffer) > self._experience_buffer_max:
                self._experience_buffer.sort(key=lambda e: e["reward"])
                self._experience_buffer = self._experience_buffer[
                    len(self._experience_buffer) - self._experience_buffer_max:
                ]

    # ──────────────────────────────────────────────
    # Genesis
    # ──────────────────────────────────────────────

    def _run_genesis(self, train_data: List[Dict]) -> None:
        """
        从训练数据中采样 seed questions，可选运行探索轨迹，执行 genesis。

        论文 §3.3：Ψ(c, T, S) 从空集 S₀=∅ 自举初始技能集。
        genesis 阶段应基于实际探索行为（而非纯文本分析），
        因此先用 base model 高温跑若干 exploratory episodes，
        将轨迹传入 genesis 供分析成功/失败模式。

        P3 修复：若配置 genesis_explore_episodes > 0，
        先调用 _run_exploratory_episodes 生成探索轨迹，
        再将轨迹传给 genesis(exploration_trajectories=...)。
        """
        import random

        # 每种 task_type 各 8 个
        task_type_samples: Dict[str, List[Dict]] = {}
        for q in train_data:
            tt = q.get("task_type", "unknown")
            if tt not in task_type_samples:
                task_type_samples[tt] = []
            if len(task_type_samples[tt]) < 8:
                task_type_samples[tt].append(q)

        seed_questions = [q for qs in task_type_samples.values() for q in qs]
        logger.info(
            f"Genesis with {len(seed_questions)} seed questions "
            f"across {len(task_type_samples)} task types"
        )

        # 探索轨迹收集（P3 修复）
        n_explore = self.config.get("genesis_explore_episodes", 0)
        exploration_trajectories: List[Trajectory] = []

        if n_explore > 0:
            logger.info(
                f"Genesis: running {n_explore} exploratory episodes "
                f"(base model, high temperature) to bootstrap skill patterns..."
            )
            exploration_trajectories = self._run_exploratory_episodes(
                seed_questions=seed_questions,
                n_episodes=n_explore,
            )
            logger.info(
                f"Genesis: collected {len(exploration_trajectories)} exploratory trajectories, "
                f"avg_reward={sum(t.reward for t in exploration_trajectories) / max(1, len(exploration_trajectories)):.3f}"
            )
        else:
            logger.info(
                "Genesis: skipping exploratory episodes (genesis_explore_episodes=0). "
                "Set genesis_explore_episodes > 0 in config for paper-compliant genesis."
            )

        genesis_skills = self.skill_creator.genesis(
            seed_questions=seed_questions,
            exploration_trajectories=exploration_trajectories if exploration_trajectories else None,
            target_count=self.config.get("genesis_count", 12),
        )

        added = self.workspace.add_batch(genesis_skills)
        logger.info(f"Genesis added {added} skills to workspace")

    def _run_exploratory_episodes(
        self,
        seed_questions: List[Dict],
        n_episodes: int,
    ) -> List["Trajectory"]:
        """
        用当前 Supervisor（无 LoRA 训练，高温）运行 n_episodes 个探索 episode。

        论文 §3.3：genesis 自举前，用 base model（T=0.9~1.0）跑探索轨迹，
        覆盖多种策略尝试（包括很多失败），从中提取成功/失败模式。

        实现细节：
          - 从 seed_questions 中均匀采样（各 task_type 均等机会）
          - 使用较高的 inference temperature（genesis_explore_temperature，默认 0.95）
            以最大化行为多样性（base model 无 LoRA 时多样性最大）
          - episode 最大步数设为 max_episode_steps，不截断
          - 返回完整轨迹列表（reward 已填充，logprobs 未填充，
            genesis 仅使用 action_type/instruction/observation/reward 字段）

        注意：此阶段 logprob 不填充（backward_policy 未参与），
        仅供 genesis prompt 中的模式分析使用。
        """
        import random
        from training.batch_inference import supervisor_call

        if not seed_questions:
            return []

        # 按 task_type 分组，确保各类型均等采样
        by_type: Dict[str, List[Dict]] = {}
        for q in seed_questions:
            tt = q.get("task_type", "unknown")
            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append(q)

        explore_temperature = self.config.get("genesis_explore_temperature", 0.95)
        trajectories: List[Trajectory] = []

        # 循环采样并运行 episode
        task_types = list(by_type.keys())
        for ep_idx in range(n_episodes):
            # 均等轮换各 task_type
            tt = task_types[ep_idx % len(task_types)]
            question = random.choice(by_type[tt])

            try:
                # 推导 g_q 和检索 E_ret（genesis 阶段无历史，二者可能为空）
                episode_goal = self._derive_episode_goal(question)
                retrieved_exp = self._retrieve_experience(question)

                messages, traj = self.env.reset(
                    question,
                    episode_goal=episode_goal,
                    retrieved_experience=retrieved_exp,
                )
                traj.task_type_id = TASK_TYPE_TO_ID.get(
                    question.get("task_type", "unknown"), 9
                )
                # v6: Use task-type-filtered tools from env
                tools = self.env._tools

                max_retries = 2
                for _step_idx in range(self.max_episode_steps):
                    # 高温采样（提高多样性）
                    content, tool_name, tool_args = "", None, None
                    for attempt in range(max_retries):
                        try:
                            content, tool_name, tool_args = supervisor_call(
                                messages=messages,
                                api_base=self.config["supervisor_api_base"],
                                model=self.config.get("supervisor_model", "Qwen3-8B"),
                                tools=tools,
                                max_tokens=512,
                                temperature=explore_temperature,
                            )
                            break
                        except Exception as e:
                            logger.warning(
                                f"Exploratory episode {ep_idx} vLLM call failed "
                                f"(attempt {attempt+1}): {e}"
                            )
                            content, tool_name, tool_args = "", None, None

                    reward, done, info = self.env.step(content, tool_name, tool_args, traj)

                    # v6: Populate supervisor_input with chat-template-rendered text
                    if traj.turns and traj.turns[-1].messages_snapshot:
                        try:
                            traj.turns[-1].supervisor_input = self._messages_to_text(
                                traj.turns[-1].messages_snapshot
                            )
                        except Exception:
                            pass  # Keep the text fallback from env.step

                    if done:
                        break

                trajectories.append(traj)
                logger.debug(
                    f"Exploratory episode {ep_idx+1}/{n_episodes}: "
                    f"task={tt}, reward={traj.reward:.3f}, steps={traj.n_steps}"
                )

            except Exception as e:
                logger.warning(f"Exploratory episode {ep_idx} failed: {e}")
                continue

        return trajectories

    # ──────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────

    def _sample_batch(self) -> List[Dict]:
        """从训练数据中采样 batch_size 个任务（按 source 均衡采样，每个数据集 2 个）"""
        import random
        # 固定种子 = step 号，确保每次重启相同 step 用相同数据
        random.seed(42 + getattr(self, '_current_step', 0))

        # 按 source 分组（MATH-Hard, AIME, HumanEval+, MBPP+, MuSiQue, HotpotQA-hard）
        by_source: Dict[str, List[Dict]] = {}
        for q in self._train_data:
            src = q.get("extra", {}).get("source", q.get("task_type", "unknown"))
            by_source.setdefault(src, []).append(q)

        sources = list(by_source.keys())
        per_source = max(1, self.batch_size // len(sources)) if sources else 1

        batch = []
        for src, qs in by_source.items():
            if qs:
                batch.extend(random.sample(qs, min(per_source, len(qs))))

        # 补足
        while len(batch) < self.batch_size:
            batch.append(random.choice(self._train_data))

        random.shuffle(batch)
        return batch[:self.batch_size]

    def _collect_stats(
        self,
        step: int,
        loss: torch.Tensor,
        trajectories: List[Trajectory],
        log_z: torch.Tensor,
    ) -> Dict:
        """收集训练统计"""
        rewards = [t.reward for t in trajectories]
        answer_rewards = [t.answer_reward for t in trajectories]
        n_steps = [t.n_steps for t in trajectories]

        h_flow = compute_flow_entropy(trajectories)

        # 技能流量熵（用于进化触发监控）
        h_skill_ratio = 0.0
        if self.evolution_manager and hasattr(self, "_current_marginal_flows"):
            _, _, h_skill_ratio = self.evolution_manager._compute_skill_flow_entropy(
                self._current_marginal_flows
            )

        return {
            "step": step,
            "loss": round(loss.item(), 4),
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_answer": round(sum(answer_rewards) / len(answer_rewards), 4),
            "avg_steps": round(sum(n_steps) / len(n_steps), 2),
            "flow_entropy": round(h_flow, 4),
            "h_skill_ratio": round(h_skill_ratio, 4),
            "log_z_mean": round(
                (log_z.mean().item() if isinstance(log_z, torch.Tensor)
                 else sum(log_z) / max(1, len(log_z))),
                4,
            ),
            "workspace_size": self.workspace.size,
            "n_trajs": len(trajectories),
            "n_completed": sum(1 for t in trajectories if t.completed),
            "task_rewards": {
                tt: round(
                    sum(t.answer_reward for t in trajectories if t.task_type == tt) /
                    max(1, sum(1 for t in trajectories if t.task_type == tt)),
                    4
                )
                for tt in set(t.task_type for t in trajectories)
            },
        }

    def _log_step(self, stats: Dict) -> None:
        """写入 JSONL 日志 + wandb"""
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

        # wandb logging
        if self._wandb:
            log_dict = {
                "loss": stats.get("loss", 0),
                "reward": stats.get("avg_reward", 0),
                "answer_reward": stats.get("avg_answer", 0),
                "avg_steps": stats.get("avg_steps", 0),
                "flow_entropy": stats.get("flow_entropy", 0),
                "h_skill_ratio": stats.get("h_skill_ratio", 0),
                "log_z_mean": stats.get("log_z_mean", 0),
                "skills": stats.get("workspace_size", 0),
                "n_trajs": stats.get("n_trajs", 0),
            }
            # per-task accuracy
            for tt, acc in stats.get("task_rewards", {}).items():
                log_dict[f"acc/{tt}"] = acc
            self._wandb.log(log_dict, step=stats.get("step", 0))

    def _run_validation(self, step: int) -> None:
        """在固定验证集上评估，排除 batch 方差。

        每次用相同的 14 道题（每个 type 2 道，固定种子），
        这样不同 step 之间的分数可以直接对比。
        """
        import random as _rng
        val_rng = _rng.Random(42)  # 固定种子，每次抽相同的题

        # 每个 type 抽 2 道（和训练 batch 一样的分布）
        from collections import defaultdict
        by_type = defaultdict(list)
        for q in self._val_data:
            by_type[q.get("task_type", "unknown")].append(q)

        val_batch = []
        for tt in sorted(by_type.keys()):
            samples = by_type[tt]
            if len(samples) >= 2:
                picked = val_rng.sample(samples, 2)
            else:
                picked = samples[:2]
            val_batch.extend(picked)

        if not val_batch:
            return

        # 运行 episodes（复用训练的 _collect_episodes）
        logger.info(f"[Validation] Step {step}: running {len(val_batch)} fixed val episodes...")
        val_trajs = self._collect_episodes(val_batch)

        # 统计
        avg_r = sum(t.reward for t in val_trajs) / max(len(val_trajs), 1)
        avg_ans = sum(t.answer_reward for t in val_trajs) / max(len(val_trajs), 1)
        per_type = defaultdict(list)
        for t in val_trajs:
            per_type[t.task_type].append(t.answer_reward)

        type_str = " ".join(
            f"{tt[:3]}={sum(rs)/len(rs):.2f}" for tt, rs in sorted(per_type.items())
        )

        logger.info(
            f"[Validation] Step {step}: r={avg_r:+.3f} ans={avg_ans:.3f} "
            f"| {type_str}"
        )

        # 写入单独的 val log
        val_log = self.output_dir / "validation_log.jsonl"
        with open(val_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step,
                "val_reward": round(avg_r, 4),
                "val_answer": round(avg_ans, 4),
                "val_per_type": {
                    tt: round(sum(rs)/len(rs), 4)
                    for tt, rs in per_type.items()
                },
                "val_n": len(val_trajs),
            }, ensure_ascii=False) + "\n")

    def _dump_sample_trajectory(self, step: int, trajectories: List[Trajectory]) -> None:
        """Dump 多样化轨迹的详细 per-turn 信息到文件。

        修复：原 trajectories[:2] 总是取 batch 前两条（固定 task_type），
        改为每个 task_type 各采样 1 条，最多 4 条，确保 dump 反映真实多样性。
        """
        import random
        dump_dir = self.output_dir / "trajectory_dumps"
        dump_dir.mkdir(exist_ok=True)
        dump_file = dump_dir / f"step_{step:04d}.jsonl"
        try:
            # 按 task_type 分组，每组随机选 1 条
            by_type: Dict[str, List] = {}
            for traj in trajectories:
                tt = getattr(traj, 'task_type', 'unknown')
                by_type.setdefault(tt, []).append(traj)
            samples = [random.choice(trajs) for trajs in by_type.values()]
            # 最多 dump 4 条
            if len(samples) > 4:
                samples = random.sample(samples, 4)

            with open(dump_file, "w", encoding="utf-8") as f:
                for traj in samples:
                    f.write(traj.to_jsonl() + "\n")
            logger.info(f"Trajectory dump: {dump_file} ({len(samples)} samples from {len(by_type)} task types)")
        except Exception as e:
            logger.warning(f"Trajectory dump failed: {e}")

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """保存 checkpoint"""
        tag = "final" if final else f"step_{step:04d}"
        ckpt_dir = self.output_dir / f"checkpoint_{tag}"
        ckpt_dir.mkdir(exist_ok=True)

        # θ-LoRA（共享模型时仅保存 theta 适配器）
        self.shared_model.set_adapter("theta")
        self.shared_model.save_pretrained(
            ckpt_dir / "supervisor_lora",
            selected_adapters=["theta"],
        )

        # set_adapter("theta") 会重置 phi 的 requires_grad，恢复它
        for name, param in self.shared_model.named_parameters():
            if ".lora_A.phi" in name or ".lora_B.phi" in name:
                param.requires_grad_(True)

        # φ-LoRA
        self.backward_policy.save(str(ckpt_dir / "backward_lora"))

        # Z_θ head
        torch.save(self.partition_fn.state_dict(), ckpt_dir / "partition_fn.pt")

        # Skill workspace
        self.workspace.save_all()

        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _sync_lora_to_vllm(self, step: int) -> None:
        """v10 tensor-based LoRA sync via /load_lora_adapter_from_tensors (verl pattern).

        前提: SGLang 以 mp.Process spawn as child of trainer (authkey 共享).
        由 training.sglang_manager.SGLangSupervisorManager 启动.

        - 固定 adapter name "theta_live" (永不 LRU)
        - 内存 tensor 传输 (.cpu() + MultiprocessingSerializer), 无磁盘 I/O
        - Timeout 300s 容忍长 in-flight decode (不中断 rollout 正确性)
        - Post-sync 验证 /v1/models 有 theta_live, 防 silent fallback
        """
        try:
            import time as _time
            import requests
            from dataclasses import asdict
            from peft.utils import get_peft_model_state_dict
            from sglang.srt.utils import MultiprocessingSerializer

            _t0 = _time.time()

            # 1. 提取 θ-LoRA tensors (GPU → CPU; 必要因 SGLang 子进程 CUDA 可能不可跨 device)
            self.shared_model.set_adapter("theta")
            lora_state = get_peft_model_state_dict(
                self.shared_model, adapter_name="theta"
            )
            # 恢复 phi 的 requires_grad (set_adapter("theta") 会重置)
            for name, param in self.shared_model.named_parameters():
                if ".lora_A.phi" in name or ".lora_B.phi" in name:
                    param.requires_grad_(True)

            processed_weights = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in lora_state.items()
            }

            # 2. PEFT config → JSON-serializable dict
            peft_config = self.shared_model.peft_config["theta"]
            peft_config_json = asdict(peft_config)
            pt = peft_config_json.get("peft_type")
            if hasattr(pt, "value"):
                peft_config_json["peft_type"] = pt.value
            tt = peft_config_json.get("task_type")
            if hasattr(tt, "value"):
                peft_config_json["task_type"] = tt.value
            tm = peft_config_json.get("target_modules")
            if isinstance(tm, (set, frozenset)):
                peft_config_json["target_modules"] = sorted(list(tm))

            _t1 = _time.time()

            # 3. Serialize via MultiprocessingSerializer (ForkingPickler, 需 shared authkey)
            serialized = MultiprocessingSerializer.serialize(
                processed_weights, output_str=True
            )
            _t_ser = _time.time()

            _api_key = self.config.get("supervisor_api_key") or os.environ.get("SUPERVISOR_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY")
            headers = {"Authorization": f"Bearer {_api_key}"}
            api_base = self.config['supervisor_api_base'].rstrip('/v1')
            FIXED_ADAPTER_NAME = "theta_live"

            from training.batch_inference import _sglang_pause_event, set_current_adapter_name
            _sglang_pause_event.set()
            load_ok = False
            load_time = 0.0
            try:
                _time.sleep(0.3)  # drain: 让新 supervisor_call 等
                _t_drain = _time.time()

                # 4a. Unload old theta_live (first call fails w/ 400 — ignore)
                try:
                    requests.post(
                        f"{api_base}/unload_lora_adapter",
                        headers=headers,
                        json={"lora_name": FIXED_ADAPTER_NAME},
                        timeout=60,
                    )
                except Exception as ue:
                    logger.debug(f"Unload (expected first): {ue}")

                # 4b. Load from tensors (timeout 300s 容忍长 in-flight decode)
                try:
                    resp = requests.post(
                        f"{api_base}/load_lora_adapter_from_tensors",
                        headers=headers,
                        json={
                            "lora_name": FIXED_ADAPTER_NAME,
                            "config_dict": peft_config_json,
                            "serialized_tensors": serialized,
                            "pinned": False,
                        },
                        timeout=300,
                    )
                    load_ok = resp.status_code == 200 and resp.json().get("success", False)
                    if not load_ok:
                        logger.warning(
                            f"LoRA sync (step {step}): HTTP {resp.status_code}: {resp.text[:300]}"
                        )
                except requests.exceptions.ReadTimeout:
                    logger.warning(f"LoRA sync (step {step}): load timeout (>300s)")
                    load_ok = False
                except Exception as le:
                    logger.warning(f"LoRA sync (step {step}): load failed: {le}")

                _t2 = _time.time()
                load_time = _t2 - _t_drain

                # 4c. flush_cache + post-verify /v1/models
                if load_ok:
                    try:
                        requests.post(f"{api_base}/flush_cache", headers=headers, timeout=10)
                    except Exception as fe:
                        logger.debug(f"flush_cache (non-fatal): {fe}")

                    # Post-sync verify: confirm theta_live exists in /v1/models
                    try:
                        v = requests.get(f"{api_base}/v1/models", headers=headers, timeout=5)
                        if FIXED_ADAPTER_NAME not in v.text:
                            logger.warning(
                                f"LoRA sync (step {step}): /v1/models lacks {FIXED_ADAPTER_NAME}; load was silent-fail"
                            )
                            load_ok = False
                    except Exception as ve:
                        logger.debug(f"Post-verify failed (non-fatal): {ve}")

                if load_ok:
                    set_current_adapter_name(FIXED_ADAPTER_NAME)
                    logger.info(
                        f"LoRA sync OK (step {step}): name={FIXED_ADAPTER_NAME}, "
                        f"extract={_t1-_t0:.2f}s, serialize={_t_ser-_t1:.2f}s, "
                        f"http={load_time:.2f}s, total={_t2-_t0:.2f}s"
                    )
                else:
                    logger.warning(f"LoRA sync failed (step {step}), http={load_time:.2f}s")

                # Fallback: restart SGLang child + retry
                if not load_ok:
                    logger.warning(
                        f"LoRA sync degraded (http={load_time:.1f}s, ok={load_ok}), "
                        f"restarting SGLang child..."
                    )
                    self._restart_sglang_supervisor()
                    try:
                        resp2 = requests.post(
                            f"{api_base}/load_lora_adapter_from_tensors",
                            headers=headers,
                            json={
                                "lora_name": FIXED_ADAPTER_NAME,
                                "config_dict": peft_config_json,
                                "serialized_tensors": serialized,
                                "pinned": False,
                            },
                            timeout=300,
                        )
                        if resp2.status_code == 200 and resp2.json().get("success", False):
                            # verify
                            v2 = requests.get(f"{api_base}/v1/models", headers=headers, timeout=5)
                            if FIXED_ADAPTER_NAME in v2.text:
                                set_current_adapter_name(FIXED_ADAPTER_NAME)
                                logger.info("SGLang restarted and adapter reloaded (tensor)")
                                load_ok = True
                            else:
                                logger.warning("Post-restart verify failed: adapter not in /v1/models")
                        else:
                            logger.warning(
                                f"Post-restart load returned {resp2.status_code}: {resp2.text[:200]}"
                            )
                    except Exception as e:
                        logger.warning(f"Post-restart load failed: {e}")

                # CRITICAL: if still not ok, keep pause set so rollouts don't use base model.
                # Supervisor_call will hang until resolved — safer than silent pollution.
                if not load_ok:
                    logger.error(
                        f"LoRA sync UNRECOVERABLE at step {step}; pause event held to block rollouts"
                    )
                    return  # keep pause set; skip pause.clear() to block new rollouts
            finally:
                if load_ok:
                    _sglang_pause_event.clear()

            # LoRA B ���重监控：追踪学习进度
            b_max = 0.0
            b_norms = []
            for pn, pv in self.shared_model.named_parameters():
                if ".lora_B.theta" in pn:
                    b_max = max(b_max, pv.data.abs().max().item())
                    b_norms.append(pv.data.norm().item())
            if b_norms:
                b_avg_norm = sum(b_norms) / len(b_norms)
                logger.info(
                    f"  LoRA B θ: max_abs={b_max:.6f} avg_norm={b_avg_norm:.6f} "
                    f"({len(b_norms)} matrices)"
                )
        except Exception as e:
            logger.warning(f"LoRA sync error (step {step}): {e}")

    def _restart_sglang_supervisor(self) -> None:
        """v10: restart via sglang_manager (mp.Process.terminate + spawn).

        前提: self.sglang_mgr 已由 train() 入口启动.
        不再用 pkill/nohup — terminate() 同步等待 child 退出, 然后重新 spawn.
        """
        if not hasattr(self, "sglang_mgr") or self.sglang_mgr is None:
            logger.error("[SGLang Restart] sglang_mgr not initialized; cannot restart cleanly")
            return
        try:
            self.sglang_mgr.restart()
            logger.info("[SGLang Restart] child restarted and ready (via sglang_manager)")
        except Exception as e:
            logger.error(f"[SGLang Restart] sglang_manager.restart() failed: {e}")


    def resume(self, checkpoint_path: str) -> None:
        """从 checkpoint 恢复训练（加载 θ-LoRA weights + Z_θ）"""
        ckpt_dir = Path(checkpoint_path)
        logger.info(f"Resuming from {ckpt_dir}")

        # 加载 θ-LoRA weights（save_pretrained 存在 supervisor_lora/theta/ 子目录）
        theta_dir = ckpt_dir / "supervisor_lora" / "theta"
        if not theta_dir.exists():
            theta_dir = ckpt_dir / "supervisor_lora"  # fallback

        if theta_dir.exists() and (theta_dir / "adapter_config.json").exists():
            # 先删除旧的 theta adapter，再从 checkpoint 加载
            self.shared_model.delete_adapter("theta")
            self.shared_model.load_adapter(str(theta_dir), adapter_name="theta")
            self.shared_model.set_adapter("theta")
            # 恢复 phi 的 requires_grad（set_adapter 会重置）
            for name, param in self.shared_model.named_parameters():
                if ".lora_A.phi" in name or ".lora_B.phi" in name:
                    param.requires_grad_(True)
            logger.info(f"θ-LoRA loaded from {theta_dir}")
        else:
            logger.warning(f"No θ-LoRA checkpoint found at {theta_dir}")

        # 加载 Z_θ
        if (ckpt_dir / "partition_fn.pt").exists():
            self.partition_fn.load_state_dict(
                torch.load(ckpt_dir / "partition_fn.pt", map_location=self.device)
            )

        logger.info("Resume complete")
