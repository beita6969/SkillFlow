

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


    def __init__(
        self,

        shared_model: Optional[PeftModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,

        base_model_path: Optional[str] = None,
        lora_rank: int = 16,         
        lora_alpha: int = 32,        
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.base_model_path = base_model_path


        self._shared_model = shared_model
        self._use_shared = shared_model is not None

        if self._use_shared:
            self._model = shared_model
            logger.info("BackwardPolicy: using shared model with 'phi' adapter")
        else:

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

        if self._use_shared:
            logger.info("BackwardPolicy.load() skipped: using shared model")
            return


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

        if self._model is None:
            raise RuntimeError("BackwardPolicy not loaded. Call .load() first.")

        if self._use_shared:
            self._shared_model.set_adapter("phi")

        try:

            all_items = []  
            for i, traj in enumerate(trajectories):
                for t, turn in enumerate(traj.turns):


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


            result_map = {}  


            micro_bs = 4  
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

                    del outputs, logits, padded, mask  


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

        logprobs = []

        for t, turn in enumerate(traj.turns):
            if (
                getattr(turn, 'action_type', '') == 'skill_invoke'
                and not getattr(turn, 'supervisor_input', '')
                and not getattr(turn, 'messages_snapshot', None)
            ):
                logprobs.append(0.0)
                continue

            backward_text = traj.to_backward_text_per_turn(t)

            action_text = turn.supervisor_output

            lp = self._compute_action_logprob(backward_text, action_text)
            logprobs.append(lp)

        return logprobs

    def _compute_action_logprob(self, context_text: str, action_text: str) -> float:

        if self._use_shared:

            self._shared_model.set_adapter("phi")

        try:
            result = self._do_compute_logprob(context_text, action_text, with_grad=False)
        finally:
            if self._use_shared:

                self._shared_model.set_adapter("theta")

        return result

    def _do_compute_logprob(
        self, context_text: str, action_text: str, with_grad: bool = False
    ) -> float:

        ctx_manager = torch.no_grad() if not with_grad else torch.enable_grad()

        with ctx_manager:

            think_part, json_part = split_think_and_action(action_text)
            if json_part:
                effective_context = context_text + think_part
                target_text = json_part
            else:
                effective_context = context_text
                target_text = action_text


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
            logits = outputs.logits  

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

        if self._model is None:
            raise RuntimeError("BackwardPolicy not loaded.")

        all_sum_logprobs = []

        for traj in trajectories:
            traj_sum = self._compute_traj_logprob_with_grad(traj)
            all_sum_logprobs.append(traj_sum)

        return all_sum_logprobs

    def _compute_traj_logprob_with_grad(self, traj: Trajectory) -> torch.Tensor:

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


        think_part, json_part = split_think_and_action(action_text)
        if json_part:
            effective_context = context_text + think_part
            target_text = json_part
        else:
            effective_context = context_text
            target_text = action_text


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

        if self._model is None or self._use_shared:
            return
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable_params,
            lr=3e-5,
            weight_decay=0.01,
        )

    def optimizer_step(self, loss: torch.Tensor) -> None:

        if self._optimizer is None:
            return
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

    def save(self, path: str) -> None:

        if self._use_shared and self._shared_model is not None:

            self._shared_model.save_pretrained(path, selected_adapters=["phi"])
            logger.info(f"BackwardPolicy (phi adapter) saved to {path}")
        elif self._model is not None:
            self._model.save_pretrained(path)
            logger.info(f"BackwardPolicy saved to {path}")

    @property
    def model(self) -> Optional[PeftModel]:
        return self._model
