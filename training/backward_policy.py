"""
BackwardPolicy — P_φ（共享 Qwen3-8B 基础模型，独立 φ-LoRA 适配器）。

GFlowNet TTB 中，backward policy P_φ 的作用：
  - 给定后继状态 H_t = H_{t-1} ⊕ o_t，估计 log P_φ(a_t | H_{t-1} ⊕ o_t)
  - 论文 Eq.4 记法：P_φ(a_{t-1} | H_t)（t 从 1 开始计，a_{t-1} 对应第 t 步动作）
    本实现 t 从 0 开始计，等价写作 P_φ(a_t | H_{t-1} ⊕ o_t)
  - P_φ 比 π_θ 多看到 o_t（执行结果），因此能识别哪些动作真正有效
  - 高 P_φ(a_t | H_{t-1} ⊕ o_t) 意味着该动作在事后（给定执行结果）看来是"合理的"

记法说明（避免混淆论文与代码）：
  - 论文 Eq.4 中 H_t 指「第 t 步执行后的完整历史」= H_{t-1} ⊕ a_t ⊕ o_t
  - 论文记 P_φ(a_{t-1} | H_t) 是用 H_t（含 o_t）反推「刚才执行的」动作 a_{t-1}
  - to_backward_text_per_turn(t) 构建的是「H_{t-1} ⊕ o_t」（不含 a_t），
    作为 P_φ 的 context，目标 token 为 a_t（supervisor_output）
  - TTB balance: log Z + Σ log π_θ(a_t) - β·log R̃ - Σ log P_φ(a_t | H_{t-1} ⊕ o_t)

共享模型设计（OOM 优化）：
  - P_φ 与 π_θ 共享同一 Qwen3-8B base model
  - 通过命名适配器区分："theta"（π_θ）和 "phi"（P_φ）
  - 计算时 set_adapter("phi")，完成后恢复 set_adapter("theta")
  - 避免加载第二个 16.4 GB 基础模型，节省约 16 GB 显存
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.trajectory import Trajectory, split_think_and_action

logger = logging.getLogger(__name__)


class BackwardPolicy:
    """
    P_φ：反向策略。

    在 GFlowNet TTB 框架中：
      log P_φ(a_t | H_{t-1} ⊕ o_t) 表示给定先前状态和本步执行结果，
      反推当前动作的概率（即事后评估该动作是否"合理"）

      高 P_φ(a_t | H_{t-1} ⊕ o_t) → 该动作在事后（给定 o_t）看来有效
      低 P_φ(a_t | H_{t-1} ⊕ o_t) → 即使 forward policy 选择了该动作，
                                      backward flow 认为它在该 context 下不理想

    实现约定：
      context_text = to_backward_text_per_turn(t) = supervisor_input + "Observation: " + o_t
      target_text  = supervisor_output (= a_t，含 think 部分，会由 split_think_and_action 拆分)

    步骤重要性 I(t) = exp(log π_θ(a_t) - log P_φ(a_t | H_{t-1} ⊕ o_t))：
      I(t) > 1 → Supervisor 比 backward 更确信（探索性动作，可能过于激进）
      I(t) < 1 → backward 比 Supervisor 更确信（事后"关键"动作，backward flow 高度认可）

    共享模型模式：
      shared_model 是已加载 "theta" + "phi" 两个适配器的 PeftModel。
      BackwardPolicy 在计算时切换到 "phi" 适配器，完成后恢复 "theta"。
      GFlowNetTrainer 负责管理模型加载和优化器，BackwardPolicy 只负责计算。
    """

    def __init__(
        self,
        # ── 共享模型模式（推荐，节省 ~16 GB 显存）──────────────
        shared_model: Optional[PeftModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        # ── 独立模型模式（兼容旧代码，不推荐）──────────────────
        base_model_path: Optional[str] = None,
        lora_rank: int = 16,         # 论文 Appendix H.2 / J: φ-LoRA rank=16
        lora_alpha: int = 32,        # 论文 Appendix H.2: α=32
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.base_model_path = base_model_path

        # 共享模型模式
        self._shared_model = shared_model
        self._use_shared = shared_model is not None

        if self._use_shared:
            self._model = shared_model
            logger.info("BackwardPolicy: using shared model with 'phi' adapter")
        else:
            # 独立模型模式（legacy）
            self._model: Optional[PeftModel] = None
            self.lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            logger.info("BackwardPolicy: will load independent model")

        self._optimizer: Optional[torch.optim.Optimizer] = None

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """加载模型（共享模型模式下直接返回）"""
        if self._use_shared:
            logger.info("BackwardPolicy.load() skipped: using shared model")
            return

        # 独立模型模式（legacy path）
        logger.info(f"Loading backward policy base from {self.base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True,
        )

        if checkpoint_path:
            logger.info(f"Loading φ-LoRA from {checkpoint_path}")
            self._model = PeftModel.from_pretrained(base_model, checkpoint_path)
        else:
            logger.info("Initializing fresh φ-LoRA")
            self._model = get_peft_model(base_model, self.lora_config)

        self._model.train()
        self._setup_optimizer()

    def compute_logprobs_batch(
        self,
        trajectories: List[Trajectory],
    ) -> List[List[float]]:
        """
        批量计算所有轨迹的 backward log probs（micro_bs=8，~10x 加速）。

        对每条轨迹 τ，对每个 step t，计算 log P_φ(a_t | H_t)：
          H_t = H_{t-1} + a_t + o_t（backward policy 额外看到 observation）
        """
        if self._model is None:
            raise RuntimeError("BackwardPolicy not loaded. Call .load() first.")

        if self._use_shared:
            self._shared_model.set_adapter("phi")

        try:
            # 收集所有 turns
            all_items = []  # (traj_idx, turn_idx, ctx_ids, act_ids)
            for i, traj in enumerate(trajectories):
                for t, turn in enumerate(traj.turns):
                    # Only legacy auto-injected skill markers are virtual.  A
                    # policy-selected skill_invoke tool call has real context and
                    # must be scored by P_φ like other actions.
                    if (
                        getattr(turn, 'action_type', '') == 'skill_invoke'
                        and not getattr(turn, 'supervisor_input', '')
                        and not getattr(turn, 'messages_snapshot', None)
                    ):
                        all_items.append((i, t, None, None))
                        continue
                    backward_text = traj.to_backward_text_per_turn(t)
                    action_text = turn.supervisor_output

                    think_part, json_part = split_think_and_action(action_text)
                    if json_part:
                        effective_context = backward_text + think_part
                        target_text = json_part
                    else:
                        effective_context = backward_text
                        target_text = action_text

                    ctx_ids = self.tokenizer.encode(
                        effective_context, add_special_tokens=False, return_tensors="pt"
                    )[0]
                    act_ids = self.tokenizer.encode(
                        target_text, add_special_tokens=False, return_tensors="pt"
                    )[0]

                    if act_ids.shape[0] == 0:
                        all_items.append((i, t, None, None))
                        continue

                    # 截断
                    max_len = 4096
                    full_len = ctx_ids.shape[0] + act_ids.shape[0]
                    if full_len > max_len:
                        keep_ctx = max(0, max_len - act_ids.shape[0] - 10)
                        ctx_ids = ctx_ids[-keep_ctx:] if keep_ctx > 0 else ctx_ids[:0]

                    if ctx_ids.shape[0] == 0:
                        all_items.append((i, t, None, None))
                        continue

                    full_ids = torch.cat([ctx_ids, act_ids])
                    all_items.append((i, t, full_ids, ctx_ids.shape[0]))

            # 初始化结果
            result_map = {}  # (traj_idx, turn_idx) → float

            # 批量计算
            micro_bs = 4  # 与 forward 一致，适配 max_len=4096
            valid_items = [(i, t, fids, cl) for i, t, fids, cl in all_items if fids is not None]

            with torch.no_grad():
                for start in range(0, len(valid_items), micro_bs):
                    batch = valid_items[start:start + micro_bs]
                    max_seq_len = max(fids.shape[0] for _, _, fids, _ in batch)

                    padded = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=self.device)
                    mask = torch.zeros(len(batch), max_seq_len, dtype=torch.long, device=self.device)

                    for bi, (_, _, fids, _) in enumerate(batch):
                        padded[bi, :fids.shape[0]] = fids.to(self.device)
                        mask[bi, :fids.shape[0]] = 1

                    outputs = self._model(padded, attention_mask=mask)
                    logits = outputs.logits

                    for bi, (ti, tj, fids, cl) in enumerate(batch):
                        seq_len = fids.shape[0]
                        action_logits = logits[bi, cl - 1:seq_len - 1, :]
                        action_targets = padded[bi, cl:seq_len]

                        if action_targets.shape[0] == 0:
                            result_map[(ti, tj)] = 0.0
                            continue

                        log_probs = torch.log_softmax(action_logits, dim=-1)
                        token_lps = log_probs[
                            torch.arange(action_targets.shape[0], device=self.device),
                            action_targets,
                        ]
                        result_map[(ti, tj)] = token_lps.sum().item()

                    del outputs, logits, padded, mask  # 释放 GPU 显存

            # 组装结果
            all_logprobs = []
            for i, traj in enumerate(trajectories):
                traj_lps = []
                for t in range(len(traj.turns)):
                    traj_lps.append(result_map.get((i, t), 0.0))
                all_logprobs.append(traj_lps)

            return all_logprobs
        finally:
            if self._use_shared:
                self._shared_model.set_adapter("theta")

    def _compute_traj_backward_logprobs(self, traj: Trajectory) -> List[float]:
        """计算单条轨迹所有步骤的 backward logprobs"""
        logprobs = []

        for t, turn in enumerate(traj.turns):
            if (
                getattr(turn, 'action_type', '') == 'skill_invoke'
                and not getattr(turn, 'supervisor_input', '')
                and not getattr(turn, 'messages_snapshot', None)
            ):
                logprobs.append(0.0)
                continue
            # H_t = supervisor_input + supervisor_output + observation
            backward_text = traj.to_backward_text_per_turn(t)
            # 目标：supervisor_output 部分的 log prob
            action_text = turn.supervisor_output

            lp = self._compute_action_logprob(backward_text, action_text)
            logprobs.append(lp)

        return logprobs

    def _compute_action_logprob(self, context_text: str, action_text: str) -> float:
        """
        计算 log P_φ(action_text | context_text)（no_grad，用于 flow 指标）。

        context_text 已含 H_{t-1} ⊕ o_t（backward policy 的后继状态 H_t，
        参见 to_backward_text_per_turn）。

        论文 §3.2：action_text = supervisor_output 同样包含 think tokens。
        与 forward policy 保持一致：
          - think 部分 (a_t^think)：并入 context，不进入 log-sum
          - JSON 部分 (a_t^out)：唯一的 logprob target

        这确保 forward 和 backward 都只对相同的 JSON token 集合打分，
        TTB balance error = log Z + Σ log π_θ(JSON) - β·log R̃ - Σ log P_φ(JSON)
        不因 thinking token 数量差异而偏移。
        """
        if self._use_shared:
            # 切换到 phi 适配器
            self._shared_model.set_adapter("phi")

        try:
            result = self._do_compute_logprob(context_text, action_text, with_grad=False)
        finally:
            if self._use_shared:
                # 恢复 theta 适配器
                self._shared_model.set_adapter("theta")

        return result

    def _do_compute_logprob(
        self, context_text: str, action_text: str, with_grad: bool = False
    ) -> float:
        """底层 logprob 计算，供 no_grad 和 with_grad 两种模式共用。"""
        ctx_manager = torch.no_grad() if not with_grad else torch.enable_grad()

        with ctx_manager:
            # ── 拆分 think 和 JSON 动作部分 ──────────────────────────────────
            think_part, json_part = split_think_and_action(action_text)
            if json_part:
                effective_context = context_text + think_part
                target_text = json_part
            else:
                effective_context = context_text
                target_text = action_text
            # ──────────────────────────────────────────────────────────────────

            ctx_ids = self.tokenizer.encode(
                effective_context, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            act_ids = self.tokenizer.encode(
                target_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)

            if act_ids.shape[1] == 0:
                return 0.0 if not with_grad else torch.zeros(1, device=self.device, requires_grad=True)

            full_ids = torch.cat([ctx_ids, act_ids], dim=1)
            ctx_len = ctx_ids.shape[1]

            # 截断到 max_length（保留 context 尾部 + 全部 target）
            max_len = 1024
            if full_ids.shape[1] > max_len:
                keep_ctx = max(0, max_len - act_ids.shape[1] - 10)
                ctx_ids = ctx_ids[:, -keep_ctx:] if keep_ctx > 0 else ctx_ids[:, :0]
                full_ids = torch.cat([ctx_ids, act_ids], dim=1)
                ctx_len = ctx_ids.shape[1]

            if ctx_len == 0:
                if with_grad:
                    return torch.zeros(1, device=self.device, requires_grad=True)
                return 0.0

            outputs = self._model(full_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            action_logits = logits[0, ctx_len - 1: -1, :]
            action_targets = full_ids[0, ctx_len:]

            if action_targets.shape[0] == 0:
                if with_grad:
                    return torch.zeros(1, device=self.device, requires_grad=True)
                return 0.0

            log_probs = torch.log_softmax(action_logits, dim=-1)
            token_logprobs = log_probs[
                torch.arange(action_targets.shape[0], device=self.device),
                action_targets,
            ]

            if with_grad:
                return token_logprobs.sum()
            else:
                return token_logprobs.sum().item()

    def compute_logprobs_with_grad(
        self,
        trajectories: List[Trajectory],
    ) -> List[torch.Tensor]:
        """
        计算 backward logprobs（保留梯度，用于训练）。

        Returns:
            List[Tensor]: 每条轨迹所有步骤的 backward logprob sum（保留梯度）
        """
        if self._model is None:
            raise RuntimeError("BackwardPolicy not loaded.")

        all_sum_logprobs = []

        for traj in trajectories:
            traj_sum = self._compute_traj_logprob_with_grad(traj)
            all_sum_logprobs.append(traj_sum)

        return all_sum_logprobs

    def _compute_traj_logprob_with_grad(self, traj: Trajectory) -> torch.Tensor:
        """计算单条轨迹所有步骤的 backward logprob sum（保留梯度）"""
        traj_sum = torch.zeros(1, device=self.device).squeeze()

        for t, turn in enumerate(traj.turns):
            backward_text = traj.to_backward_text_per_turn(t)
            action_text = turn.supervisor_output

            lp = self._compute_action_logprob_with_grad(backward_text, action_text)
            traj_sum = traj_sum + lp

        return traj_sum

    def _compute_action_logprob_with_grad(
        self, context_text: str, action_text: str
    ) -> torch.Tensor:
        """
        计算 backward logprob（保留梯度，用于 φ 参数优化）。

        与 _compute_action_logprob() 逻辑完全一致，区别仅在于：
          - 不使用 torch.no_grad()
          - 返回 Tensor 而非 float（保留计算图，用于 L_TTB.backward()）

        论文 §3.2：同样只对 JSON 动作部分 (a_t^out) 计算 logprob，
        think 部分扩展进 context，不进入梯度链。
        """
        if self._use_shared:
            self._shared_model.set_adapter("phi")

        try:
            result = self._do_compute_logprob_with_grad(context_text, action_text)
        finally:
            if self._use_shared:
                self._shared_model.set_adapter("theta")

        return result

    def _do_compute_logprob_with_grad(
        self, context_text: str, action_text: str
    ) -> torch.Tensor:
        """保留梯度的底层 logprob 计算。"""
        # ── 拆分 think 和 JSON 动作部分 ──────────────────────────────────────
        think_part, json_part = split_think_and_action(action_text)
        if json_part:
            effective_context = context_text + think_part
            target_text = json_part
        else:
            effective_context = context_text
            target_text = action_text
        # ──────────────────────────────────────────────────────────────────────

        ctx_ids = self.tokenizer.encode(
            effective_context, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        act_ids = self.tokenizer.encode(
            target_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        if act_ids.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        full_ids = torch.cat([ctx_ids, act_ids], dim=1)
        ctx_len = ctx_ids.shape[1]

        max_len = 1024
        if full_ids.shape[1] > max_len:
            keep_ctx = max(0, max_len - act_ids.shape[1] - 10)
            ctx_ids = ctx_ids[:, -keep_ctx:] if keep_ctx > 0 else ctx_ids[:, :0]
            full_ids = torch.cat([ctx_ids, act_ids], dim=1)
            ctx_len = ctx_ids.shape[1]

        if ctx_len == 0 or act_ids.shape[1] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 直接前向传播（不用 gradient checkpointing）
        # GPU 2 有 63GB 空闲，无需省激活内存；去掉 ckpt 避免 forward 执行两遍
        if self._use_shared:
            self._shared_model.set_adapter("phi")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self._model(full_ids).logits
        action_logits = logits[0, ctx_len - 1: -1, :]
        action_targets = full_ids[0, ctx_len:]

        if action_targets.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        log_probs = torch.log_softmax(action_logits, dim=-1)
        token_logprobs = log_probs[
            torch.arange(action_targets.shape[0], device=self.device),
            action_targets,
        ]
        return token_logprobs.sum()

    def _setup_optimizer(self) -> None:
        """设置 φ-LoRA 优化器（独立模型模式使用，共享模型模式由 Trainer 管理）"""
        if self._model is None or self._use_shared:
            return
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable_params,
            lr=3e-5,
            weight_decay=0.01,
        )

    def optimizer_step(self, loss: torch.Tensor) -> None:
        """反向传播 + optimizer step（独立模型模式使用）"""
        if self._optimizer is None:
            return
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

    def save(self, path: str) -> None:
        """保存 φ-LoRA checkpoint"""
        if self._use_shared and self._shared_model is not None:
            # 共享模型模式：只保存 phi 适配器
            self._shared_model.save_pretrained(path, selected_adapters=["phi"])
            logger.info(f"BackwardPolicy (phi adapter) saved to {path}")
        elif self._model is not None:
            self._model.save_pretrained(path)
            logger.info(f"BackwardPolicy saved to {path}")

    @property
    def model(self) -> Optional[PeftModel]:
        return self._model
