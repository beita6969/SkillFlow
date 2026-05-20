

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


def _variance(xs: list) -> float:

    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _spearman_rank(xs: list, ys: list) -> float:

    n = len(xs)
    if n < 4:
        return 0.0

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


class ObservationBuffer:


    def __init__(self, max_per_type: int = 20, importance_threshold: float = 0.3):
        from collections import defaultdict
        self._buffer: Dict[str, List[Dict]] = defaultdict(list)
        self._max_per_type = max_per_type
        self._threshold = importance_threshold

    def collect(self, trajectories: List[Trajectory]) -> int:

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


class PartitionFunctionHead(nn.Module):


    def __init__(self, hidden_size: int = 3584, num_task_types: int = 10):
        super().__init__()
        self.task_embed = nn.Embedding(num_task_types, 64)
        self.head = nn.Linear(hidden_size + 64, 1, bias=True)


        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.3)  

    def forward(
        self,
        query_hidden: torch.Tensor,  
        task_type_ids: torch.Tensor, 
    ) -> torch.Tensor:

        task_vec = self.task_embed(task_type_ids)  
        combined = torch.cat([query_hidden, task_vec], dim=-1)  
        log_z = self.head(combined).squeeze(-1)  

        return torch.clamp(log_z, min=-10.0, max=10.0)


def compute_ttb_loss(
    log_z: torch.Tensor,                        
    forward_logprob_sums: torch.Tensor,         
    backward_logprob_sums: torch.Tensor,        
    r_tilde: torch.Tensor,                      
    beta: float = 1.0,
    normalizer: Optional[torch.Tensor] = None,  
) -> torch.Tensor:

    log_r_tilde = torch.log(r_tilde.clamp(min=1e-8))
    balance = log_z + forward_logprob_sums - beta * log_r_tilde - backward_logprob_sums
    if normalizer is not None:
        balance = balance / normalizer.clamp(min=1)
    loss = (balance ** 2).mean()
    return loss


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


    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
        self.output_dir = Path(config.get("output_dir", "outputs/skillflow"))
        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.base_model_path = config["base_model"]
        self.beta = config.get("beta", 1.0)
        self.epsilon_min = config.get("epsilon_min", EPSILON_MIN)


        self.max_steps = config.get("max_steps", 300)
        self.batch_size = config.get("batch_size", 8)
        self.max_episode_steps = config.get("max_episode_steps", 8)
        self.save_every = config.get("save_every", 50)
        self.sync_lora_every = config.get("sync_lora_every", 5)
        self.evolution_phase_steps = config.get("evolution_phase_steps", 20)
        self.lora_rank = config.get("lora_rank", 64)


        self.tokenizer: Optional[AutoTokenizer] = None
        self.supervisor_model: Optional[PeftModel] = None  
        self.partition_fn: Optional[PartitionFunctionHead] = None
        self.backward_policy: Optional[BackwardPolicy] = None
        self.m_exec: Optional[MExec] = None
        self.workspace: Optional[SkillWorkspace] = None
        self.skill_creator: Optional[SkillCreator] = None
        self.evolution_manager: Optional[FlowSkillEvolutionManager] = None
        self.env: Optional[GenericTaskEnvironment] = None


        self._supervisor_optimizer: Optional[torch.optim.Optimizer] = None
        self._partition_optimizer: Optional[torch.optim.Optimizer] = None
        self._phi_optimizer: Optional[torch.optim.Optimizer] = None


        self._current_step = 0
        self._skills_just_updated = False
        self._phase_trajectories: List[Trajectory] = []


        self._observation_buffer = ObservationBuffer(
            max_per_type=20,
            importance_threshold=self.config.get("observation_I_threshold", 0.3),
        )

        from collections import defaultdict
        self._per_type_balance_history: Dict[str, List[float]] = defaultdict(list)
        self._per_type_last_evolution: Dict[str, int] = defaultdict(int)  
        self._per_type_evolution_helped: Dict[str, bool] = defaultdict(lambda: True)  
        self._per_type_acc_history: Dict[str, List[float]] = defaultdict(list)  
        self._skill_negative_counter: Dict[str, int] = defaultdict(int)  
        self._plateau_detector = PlateauDetector(
            window_size=self.config.get("plateau_window_size", self.evolution_phase_steps),
            rho=self.config.get("plateau_rho", 0.05),
            m_consecutive=self.config.get("plateau_m_consecutive", 2),
        )


        self.experience_store = None
        self._experience_buffer: List[Dict] = []
        self._experience_lock = threading.Lock()


        self._log_file = self.output_dir / "training_log.jsonl"


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

        logger.info("Setting up GFlowNetTrainer...")


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        logger.info("Loading shared base model (θ-LoRA + φ-LoRA)...")
        self.extra_device = self.config.get("extra_device", None)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )


        theta_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.shared_model = get_peft_model(base_model, theta_config, adapter_name="theta")


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


        self.shared_model.set_adapter("theta")
        self.shared_model.train()


        phi_grad_count = 0
        for name, param in self.shared_model.named_parameters():
            if ".lora_A.phi" in name or ".lora_B.phi" in name:
                param.requires_grad_(True)
                phi_grad_count += 1
        logger.info(f"Re-enabled requires_grad for {phi_grad_count} φ-LoRA params")
        self.supervisor_model = self.shared_model  


        self.shared_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for batched micro-batch training")


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


        hidden_size = base_model.config.hidden_size
        self.partition_fn = PartitionFunctionHead(
            hidden_size=hidden_size,
            num_task_types=len(TASK_TYPE_TO_ID),
        ).to(self.device)


        self.backward_policy = BackwardPolicy(
            shared_model=self.shared_model,
            tokenizer=self.tokenizer,
            device=self.device,
        )


        self.m_exec = MExec(
            api_base=self.config["executor_api_base"],
            model_name=self.config.get("executor_model", "gpt-oss-120b"),
            api_key=self.config.get("executor_api_key") or os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY"),
        )
        logger.info(f"M_exec connectivity: {self.m_exec.test_connectivity()}")


        skills_dir = self.output_dir / "skills"
        self.workspace = SkillWorkspace(
            skills_dir=skills_dir,
            max_skills=self.config.get("max_skills_total", 60),
        )


        self.skill_creator = SkillCreator(
            m_exec=self.m_exec,
            skill_workspace=self.workspace,
            backward_policy=self.backward_policy,
        )


        self.env = GenericTaskEnvironment(
            m_exec=self.m_exec,
            max_episode_steps=self.max_episode_steps,
            epsilon_min=self.epsilon_min,
            skill_workspace=self.workspace,
            reward_mode=self.config.get("reward_mode", "outcome_only"),
            skill_mode=self.config.get("skill_mode", "policy_action"),
        )


        if self.workspace.size == 0:
            genesis_count = self.config.get("genesis_count", 12)
            if genesis_count > 0:
                logger.info("Workspace empty, running genesis...")
                self._run_genesis(train_data)
            else:
                logger.info("Workspace empty, S₀=∅ — skills will be discovered from trajectories via evolution")


        self.accuracy_tracker = TaskTypeAccuracyTracker(
            threshold=self.config.get("accuracy_threshold", 0.5),
            min_count=10,
        )


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


        self._setup_optimizers()

        self._train_data = train_data
        self._val_data = val_data or []

        logger.info(
            f"Setup complete: workspace={self.workspace.size} skills, "
            f"train={len(train_data)}, val={len(self._val_data)}"
        )

    def _sample_batch_for_step(self, step: int) -> List[Dict]:

        import random as _random
        _seed_offset = int(self.config.get("seed_offset", 0))
        rng = _random.Random(42 + step + _seed_offset * 131)

        n_traj = self.config.get("n_trajectories_per_question", 2)

        by_source: Dict[str, List[Dict]] = {}
        for q in self._train_data:
            src = q.get("extra", {}).get("source", q.get("task_type", "unknown"))
            by_source.setdefault(src, []).append(q)


        n_unique = max(1, self.batch_size // n_traj)
        sources = list(by_source.keys())
        per_source = max(1, n_unique // len(sources)) if sources else 1

        unique_questions = []
        for src, qs in by_source.items():
            if qs:
                unique_questions.extend(rng.sample(qs, min(per_source, len(qs))))


        while len(unique_questions) < n_unique:
            unique_questions.append(rng.choice(self._train_data))
        unique_questions = unique_questions[:n_unique]


        batch = []
        for q in unique_questions:
            for _ in range(n_traj):
                batch.append(q.copy())

        rng.shuffle(batch)
        return batch[:self.batch_size]

    def train(self) -> None:

        logger.info(f"Starting GFlowNet training for {self.max_steps} steps")
        start_time = time.time()
        from concurrent.futures import ThreadPoolExecutor, Future


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


        if not hasattr(self, "sglang_mgr") or self.sglang_mgr is None:
            from training.sglang_manager import SGLangSupervisorManager, _set_shared_authkey
            _set_shared_authkey()  
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


        if self._current_step == 0:
            logger.info("Initial weight sync: resetting sglang to current LoRA state")
            self._sync_lora_to_vllm(step=-1)


        _next_future: Future = None
        _async_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async_rollout")

        for step in range(self._current_step, self.max_steps):
            self._current_step = step


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


            if step % self.sync_lora_every == 0:
                self._sync_lora_to_vllm(step)


            if step + 1 < self.max_steps and _next_future is None:
                _next_batch = self._sample_batch_for_step(step + 1)
                _next_future = _async_pool.submit(self._collect_episodes, _next_batch)
                logger.info(f"  Async: step {step+1} rollout launched (overlaps with GPU compute)")


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


            with torch.no_grad():
                log_z = self._compute_partition_function(trajectories)
            log_z_floats = log_z.detach().tolist()


            for i, traj in enumerate(trajectories):
                fill_turn_flows(traj, log_z_floats[i])


            logger.info("Gradient accumulation...")
            loss_val = self._gradient_accumulation_step(trajectories)
            self._plateau_detector.update(step, loss_val)
            _t3 = _time.time()
            logger.info(f"Gradient done in {_t3-_t2:.1f}s")
            loss = torch.tensor(loss_val, device=self.device, dtype=torch.float32)


            self._update_flow_metrics(trajectories, log_z)


            n_obs = self._observation_buffer.collect(trajectories)
            if n_obs > 0:
                logger.debug(f"[ObsBuffer] Collected {n_obs} I(t)<{self._observation_buffer._threshold} observations, total={self._observation_buffer.total_count()}")


            self._update_per_type_balance(trajectories)


            stats = self._collect_stats(step, loss, trajectories, log_z)
            self._log_step(stats)


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


            if step % 5 == 0:
                self._dump_sample_trajectory(step, trajectories)


            self._try_evolve(step)


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


    def _collect_episodes(self, batch: List[Dict]) -> List[Trajectory]:

        n = len(batch)
        max_workers = min(n, self.config.get("rollout_workers", 24))


        for ep_idx, question in enumerate(batch):
            source = question.get("extra", {}).get("source", question.get("task_type", "?"))
            logger.info(f"  Episode {ep_idx+1}/{n}: {source} | q={str(question.get('question',''))[:60]}...")


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

                    traj = Trajectory(
                        question=str(batch[ep_idx].get("question", "")),
                        gold_answer=str(batch[ep_idx].get("answer", "")),
                        task_type=str(batch[ep_idx].get("task_type", "unknown")),
                    )
                    traj.reward = 0.0  
                    traj.completed = True
                results[ep_idx] = traj


        trajectories = [results[i] for i in range(n)]

        for ep_idx, traj in enumerate(trajectories):
            self._phase_trajectories.append(traj)

            if hasattr(self, 'accuracy_tracker') and traj.task_type:
                correct = traj.reward >= 0.5  
                self.accuracy_tracker.track_result(traj.task_type, correct)
            logger.info(f"  Episode {ep_idx+1} done: {len(traj.turns)} turns, reward={traj.reward:.3f}")

        return trajectories

    def _run_episode(self, question: Dict) -> Trajectory:

        task_type = str(question.get("task_type", "unknown"))


        if task_type in ("webshop", "alfworld"):
            return self._run_react_episode(question)


        episode_goal = self._derive_episode_goal(question)


        from training.task_prompts import TASK_CONFIGS
        task_type = str(question.get("task_type", "unknown"))
        _task_cfg = TASK_CONFIGS.get(task_type, {})
        _ep_steps = _task_cfg.get("max_episode_steps", self.max_episode_steps)

        env = GenericTaskEnvironment(
            m_exec=self.m_exec,
            max_episode_steps=_ep_steps,
            epsilon_min=self.epsilon_min,
            skill_workspace=self.workspace,
            experience_store=None,  
            reward_mode=self.config.get("reward_mode", "outcome_only"),
            skill_mode=self.config.get("skill_mode", "policy_action"),
        )

        messages, traj = env.reset(
            question,
            episode_goal=episode_goal,
        )
        traj.task_type_id = TASK_TYPE_TO_ID.get(question.get("task_type", "unknown"), 9)

        tools = env._tools

        from training.batch_inference import supervisor_call
        max_retries = 3
        import time as _time
        _episode_start = _time.time()
        _episode_timeout = 600  


        _thinking_types = set()  
        _use_thinking = task_type in _thinking_types


        def _tools_for_step(base_tools, trajectory):
            used_real_tool = any(
                getattr(t, 'action_type', '') not in ('answer', 'skill_invoke', '', 'parse_error')
                for t in trajectory.turns
            )
            if used_real_tool:
                return base_tools  

            return [t for t in base_tools if t.get('function', {}).get('name') != 'answer']

        for step_idx in range(_ep_steps):
            if _time.time() - _episode_start > _episode_timeout:
                logger.warning(f"    [S{step_idx+1}] Episode wall-clock timeout ({_episode_timeout}s)")
                break

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


            if traj.turns and traj.turns[-1].messages_snapshot:
                try:
                    traj.turns[-1].supervisor_input = self._messages_to_text(
                        traj.turns[-1].messages_snapshot
                    )
                except Exception:
                    pass  

            if done:
                break


        return traj

    def _run_react_episode(self, question: Dict) -> Trajectory:

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
        _episode_timeout = 600  

        for step_idx in range(_ep_steps):

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


    def _fill_turn_logprobs_no_grad(self, trajectories: List[Trajectory]) -> None:

        from training.backward_policy import split_think_and_action


        all_items = []  
        for i, traj in enumerate(trajectories):
            for j, turn in enumerate(traj.turns):


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


        micro_bs = 8
        max_len = 4096  


        _base_model = getattr(self.shared_model, 'base_model', self.shared_model)
        _inner_model = getattr(_base_model, 'model', _base_model)
        _inner_model.gradient_checkpointing_disable()


        self.shared_model.set_adapter("theta")
        self.shared_model.eval()

        with torch.no_grad():
            for start in range(0, len(all_items), micro_bs):
                batch = all_items[start:start + micro_bs]


                batch_max_len = min(max(item[2].shape[0] for item in batch), max_len)
                padded_ids = torch.zeros(len(batch), batch_max_len, dtype=torch.long, device=self.device)
                attention_mask = torch.zeros(len(batch), batch_max_len, dtype=torch.long, device=self.device)

                for bi, (ti, tj, full_ids, ctx_len, n_act) in enumerate(batch):
                    seq_len = min(full_ids.shape[0], batch_max_len)

                    if full_ids.shape[0] > batch_max_len:
                        full_ids = full_ids[-batch_max_len:]

                        batch[bi] = (ti, tj, full_ids, max(0, seq_len - n_act), n_act)
                    padded_ids[bi, :seq_len] = full_ids[:seq_len].to(self.device)
                    attention_mask[bi, :seq_len] = 1

                try:
                    outputs = self.shared_model(padded_ids, attention_mask=attention_mask)
                    logits = outputs.logits  

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
                    del outputs, logits  
                except Exception as e:
                    logger.warning(f"Batch forward logprob failed: {e}")
                    for bi, (ti, tj, _, _, n_act) in enumerate(batch):
                        trajectories[ti].turns[tj].forward_logprob = 0.0
                        trajectories[ti].turns[tj].action_token_count = n_act
                del padded_ids, attention_mask  


        bwd_float_lists = self.backward_policy.compute_logprobs_batch(trajectories)
        for i, traj in enumerate(trajectories):
            bwd_per = bwd_float_lists[i]
            for j, turn in enumerate(traj.turns):
                turn.backward_logprob = bwd_per[j] if j < len(bwd_per) else 0.0


        self.shared_model.set_adapter("theta")
        self.shared_model.train()
        _inner_model.gradient_checkpointing_enable()

        for _n, _p in self.shared_model.named_parameters():
            if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                _p.requires_grad_(True)
        torch.cuda.empty_cache()

    def _prepare_turn_tokens(self, context_text: str, action_text: str):

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

        max_len = 4096  
        if len(full) > max_len:
            keep = max(64, max_len - len(act_ids) - 10)  
            full = ctx_ids[-keep:] + act_ids if keep > 0 else act_ids
            ctx_len = keep if keep > 0 else 0

        if ctx_len == 0 or not act_ids:
            return None
        return (torch.tensor(full, dtype=torch.long, device=self.device), ctx_len)

    def _compute_kl_ref_on_replica(self, trajectories):

        from training.backward_policy import split_think_and_action
        device = torch.device(self.extra_device)
        self.replica_model.disable_adapter_layers()
        ref_lps = []

        micro_bs = 8

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

                if ref_lp is not None and kl_coeff > 0:
                    n_tok = max(a_targets.shape[0], 1)
                    term = term + kl_coeff * (tok_lp.sum() - ref_lp) / n_tok / batch_size
                micro_loss = term if micro_loss is None else micro_loss + term
            if micro_loss is not None:
                micro_loss.backward()
            del micro_loss, logits

    def _batched_logprob_backward(self, items, adapter_name: str, micro_bs: int = 4):

        if not items:
            return

        kl_coeff = self.config.get("kl_coeff", 0.0)
        batch_size = self.config.get("batch_size", 28)
        self.shared_model.set_adapter(adapter_name)

        if self.replica_model is None:
            self._run_micro_batches(items, self.shared_model, self.device, micro_bs,
                                    kl_coeff, batch_size)
            return


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

        micro_bs = 4  
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

        micro_bs = 4  


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

        self._supervisor_optimizer.zero_grad()
        self._partition_optimizer.zero_grad()
        if self._phi_optimizer is not None:
            self._phi_optimizer.zero_grad()


            for _n, _p in self.shared_model.named_parameters():
                if ".lora_A.phi" in _n or ".lora_B.phi" in _n:
                    _p.requires_grad_(True)

        total_loss = 0.0
        n = len(trajectories)


        theta_items = []  
        phi_items = []

        for i, traj in enumerate(trajectories):


            fwd_sum_det = sum(edge_logprob_tilde(t, "forward") for t in traj.turns)
            bwd_sum_det = sum(edge_logprob_tilde(t, "backward") for t in traj.turns)
            r_tilde_val = max(traj.r_tilde, self.epsilon_min)
            log_r = float(torch.log(torch.tensor(r_tilde_val)).item())


            query_text = traj.question[:512]
            task_type_id_val = traj.task_type_id

            query_hidden_det = self._get_query_hidden(query_text)
            task_type_id = torch.tensor([task_type_id_val], device=self.device, dtype=torch.long)
            with torch.no_grad():
                log_z_det = self.partition_fn(query_hidden_det.unsqueeze(0), task_type_id).item()


            delta_raw = log_z_det + fwd_sum_det - self.beta * log_r - bwd_sum_det
            n_steps = effective_paper_steps(traj)

            balance = delta_raw / n_steps

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


            z_grad_scale = (2.0 * balance) / (n_steps * n)


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


            log_z_grad = self.partition_fn(query_hidden_det.unsqueeze(0), task_type_id).squeeze()
            log_z_grad.backward(torch.tensor(z_grad_scale, device=self.device, dtype=torch.float32))


        kl_coeff = self.config.get("kl_coeff", 0.01)
        ref_logprobs = getattr(self, '_kl_ref_logprobs', None)
        self._kl_ref_logprobs = None


        _micro_bs = 4  
        if ref_logprobs and len(ref_logprobs) == len(theta_items):
            theta_items = [(*item, ref_lp) for item, ref_lp in zip(theta_items, ref_logprobs)]
            logger.info(f"  Batched θ backward: {len(theta_items)} turns, micro_bs={_micro_bs} (KL merged)")
        else:
            logger.info(f"  Batched θ backward: {len(theta_items)} turns, micro_bs={_micro_bs}")

        torch.cuda.empty_cache()
        self._batched_logprob_backward(theta_items, "theta", micro_bs=_micro_bs)


        if phi_items:
            torch.cuda.empty_cache()
            logger.info(f"  Batched φ backward: {len(phi_items)} turns, micro_bs={_micro_bs}")
            self._batched_logprob_backward(phi_items, "phi", micro_bs=_micro_bs)
            self.shared_model.set_adapter("theta")


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


        self._sync_replica_weights()

        return total_loss / max(n, 1)  


    def _compute_logprobs(
        self,
        trajectories: List[Trajectory],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        fwd_sums = []
        bwd_tensor_sums = []  


        for traj in trajectories:
            fwd_sum = self._compute_forward_logprob_sum(traj)
            fwd_sums.append(fwd_sum)


        bwd_sum_tensors = self.backward_policy.compute_logprobs_with_grad(trajectories)
        bwd_tensor_sums = bwd_sum_tensors  


        fwd_per_turn_lists = []
        with torch.no_grad():
            for traj in trajectories:
                fwd_per_turn_lists.append(self._compute_forward_logprob_per_turn(traj))


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

        total = torch.zeros(1, device=self.device, requires_grad=True)

        for turn in traj.turns:

            context = turn.supervisor_input
            if turn.messages_snapshot:
                try:
                    context = self._messages_to_text(turn.messages_snapshot)
                except Exception:
                    pass  
            lp = self._compute_action_logprob_forward(
                context, turn.supervisor_output
            )
            total = total + lp

        return total.squeeze()

    def _compute_forward_logprob_per_turn(self, traj: Trajectory) -> List[tuple]:

        with torch.no_grad():
            result = []
            for turn in traj.turns:

                context = turn.supervisor_input
                if turn.messages_snapshot:
                    try:
                        context = self._messages_to_text(turn.messages_snapshot)
                    except Exception:
                        pass  
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

        n_action_tokens = act_ids.shape[1]

        if n_action_tokens == 0:
            zero = torch.zeros(1, device=self.device)
            return (zero, 0) if return_token_count else zero

        full_ids = torch.cat([ctx_ids, act_ids], dim=1)
        ctx_len = ctx_ids.shape[1]


        max_len = 4096
        if full_ids.shape[1] > max_len:
            keep_ctx = max(0, max_len - act_ids.shape[1] - 10)
            ctx_ids = ctx_ids[:, -keep_ctx:] if keep_ctx > 0 else ctx_ids[:, :0]
            full_ids = torch.cat([ctx_ids, act_ids], dim=1)
            ctx_len = ctx_ids.shape[1]

        if ctx_len == 0 or act_ids.shape[1] == 0:
            zero = torch.zeros(1, device=self.device)
            return (zero, 0) if return_token_count else zero


        self.shared_model.set_adapter("theta")


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

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


    def _compute_partition_function(self, trajectories: List[Trajectory]) -> torch.Tensor:

        log_z_list = []

        for traj in trajectories:


            query_text = traj.question

            query_hidden = self._get_query_hidden(query_text)
            task_type_id = torch.tensor(
                [traj.task_type_id], device=self.device, dtype=torch.long
            )
            log_z = self.partition_fn(query_hidden.unsqueeze(0), task_type_id)
            log_z_list.append(log_z)

        return torch.cat(log_z_list)

    def _get_query_hidden(self, text: str) -> torch.Tensor:

        ids = self.tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt",
            max_length=512, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            self.shared_model.set_adapter("theta")
            outputs = self.supervisor_model(ids, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1][0, -1, :]

        return last_hidden.detach()


    def _backward_step(
        self,
        loss: torch.Tensor,
        bwd_logprob_sums: torch.Tensor,  
    ) -> None:

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

        lr = self.config.get("learning_rate", 1e-4)
        phi_lr_ratio = self.config.get("phi_lr_ratio", 0.5)
        self._max_grad_norm = self.config.get("max_grad_norm", 1.0)


        theta_params = [
            p for n, p in self.shared_model.named_parameters()
            if p.requires_grad and (".lora_A.theta" in n or ".lora_B.theta" in n)
        ]
        if not theta_params:

            logger.warning("Cannot filter theta params by name, using all trainable params")
            theta_params = [p for p in self.shared_model.parameters() if p.requires_grad]

        self._supervisor_optimizer = torch.optim.AdamW(theta_params, lr=lr, weight_decay=0.01)
        logger.info(f"θ-LoRA optimizer: {len(theta_params)} param groups")


        self._partition_optimizer = torch.optim.AdamW(
            self.partition_fn.parameters(), lr=lr * 1.5, weight_decay=0.01  
        )


        phi_params = [
            p for n, p in self.shared_model.named_parameters()
            if ".lora_A.phi" in n or ".lora_B.phi" in n
        ]

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


    def _update_flow_metrics(
        self,
        trajectories: List[Trajectory],
        log_z: torch.Tensor,
    ) -> None:

        all_skill_ids = self.workspace.get_all_ids()


        skill_marginal_flows = compute_skill_marginal_flows(trajectories, all_skill_ids)


        for sid, flow in skill_marginal_flows.items():
            if flow > -20.0:  
                self.workspace.update_flow_score(sid, flow)


        skill_variances = compute_skill_variances(trajectories, all_skill_ids)
        for sid, var in skill_variances.items():

            if skill_variances[sid] > 0.0 or any(
                turn.skill_id == sid
                for traj in trajectories
                for turn in traj.turns
                if turn.action_type == "skill_invoke"
            ):
                self.workspace.update_reward_variance(sid, var)


        for traj in trajectories:
            success = traj.reward > 0.5
            for sid in traj.skills_invoked:
                self.workspace.record_usage(sid, self._current_step, success=success)

        self._current_marginal_flows = skill_marginal_flows

    def _try_extract_experiences(self, step: int, trajectories: List[Trajectory]) -> None:

        if not self.experience_store:
            return

        high = [t for t in trajectories if t.reward >= 0.5]  
        low = [t for t in trajectories if t.reward <= 0.1]   


        if not high or not low:
            return

        success_rate = len(high) / len(trajectories)
        if success_rate < 0.1 or success_rate > 0.9:
            return  


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

            if len(self._per_type_balance_history[tt]) > 10:
                self._per_type_balance_history[tt] = self._per_type_balance_history[tt][-10:]

    def _get_struggling_types(self, step: int) -> List[str]:

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


        try:
            cf_map = extract_counterfactual_pairs(
                self._phase_trajectories,
                min_reward_gap=0.3,
            )
        except Exception as e:
            logger.warning(f"[Evolution v5] extract_counterfactual_pairs failed: {e}")
            cf_map = {}


        gate = None
        gate_n = int(self.config.get("acceptance_gate_episodes", 0) or 0)
        if gate_n > 0:
            try:
                from training.skill_evolution import SkillAcceptanceGate
                gate = SkillAcceptanceGate(
                    run_episode_fn=None,  
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


            low_flow, high_flow = split_by_flow_quartile(type_trajs)


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


            ids_before = set(self.workspace.get_all_ids()) if self.workspace else set()
            new_tips = self.evolution_manager.evolve_for_type(
                task_type=tt,
                observations=self._observation_buffer.get(tt) or [],
                failed_trajectories=low_flow[:3],
                step=step,
                high_flow_trajs=high_flow[:3],
                critical_steps=critical_steps[:5],
                dag_comparisons=dag_comparisons[:3],
                bottleneck_diagnoses=None,  
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


    def _derive_episode_goal(self, question: Dict) -> str:

        task_type = question.get("task_type", "unknown")
        q_text = str(question.get("question", ""))


        top_skills_info = ""
        if self.workspace and self.workspace.size > 0:
            top_skills = self.workspace.retrieve(q_text, task_type=task_type, top_k=2)
            if top_skills:
                skill_names = [s.name for s in top_skills]
                top_skills_info = f" Relevant skills: {', '.join(skill_names)}."


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


        return f"Produce {expected_form}."

    def _retrieve_experience(self, question: Dict, top_k: int = 2) -> List[Dict]:

        with self._experience_lock:
            if not self._experience_buffer:
                return []

            buffer_snapshot = list(self._experience_buffer)

        task_type = question.get("task_type", "")
        q_text = str(question.get("question", ""))


        try:
            from src.skills.workspace import _get_embedding_model
            model = _get_embedding_model()
            if model is not None and len(buffer_snapshot) >= 3:
                import numpy as np
                q_emb = model.encode([q_text], normalize_embeddings=True)
                exp_texts = [exp.get("question_summary", "") for exp in buffer_snapshot]
                exp_embs = model.encode(exp_texts, normalize_embeddings=True)
                sims = np.dot(exp_embs, q_emb.T).squeeze()  

                scored: List[Tuple[float, Dict]] = []
                for i, exp in enumerate(buffer_snapshot):
                    type_bonus = 0.15 if exp.get("task_type") == task_type else 0.0
                    score = float(sims[i]) + type_bonus
                    scored.append((score, exp))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [exp for score, exp in scored[:top_k] if score > 0.3]
        except Exception:
            pass


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
            "tools_used": list(dict.fromkeys(traj.tools_used)),  
            "key_steps_summary": " → ".join(key_steps),
            "reward": round(traj.reward, 3),
            "n_steps": traj.n_steps,
        }

        with self._experience_lock:
            self._experience_buffer.append(summary)


            if len(self._experience_buffer) > self._experience_buffer_max:
                self._experience_buffer.sort(key=lambda e: e["reward"])
                self._experience_buffer = self._experience_buffer[
                    len(self._experience_buffer) - self._experience_buffer_max:
                ]


    def _run_genesis(self, train_data: List[Dict]) -> None:

        import random


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

        import random
        from training.batch_inference import supervisor_call

        if not seed_questions:
            return []


        by_type: Dict[str, List[Dict]] = {}
        for q in seed_questions:
            tt = q.get("task_type", "unknown")
            if tt not in by_type:
                by_type[tt] = []
            by_type[tt].append(q)

        explore_temperature = self.config.get("genesis_explore_temperature", 0.95)
        trajectories: List[Trajectory] = []


        task_types = list(by_type.keys())
        for ep_idx in range(n_episodes):

            tt = task_types[ep_idx % len(task_types)]
            question = random.choice(by_type[tt])

            try:

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

                tools = self.env._tools

                max_retries = 2
                for _step_idx in range(self.max_episode_steps):

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


                    if traj.turns and traj.turns[-1].messages_snapshot:
                        try:
                            traj.turns[-1].supervisor_input = self._messages_to_text(
                                traj.turns[-1].messages_snapshot
                            )
                        except Exception:
                            pass  

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


    def _sample_batch(self) -> List[Dict]:

        import random

        random.seed(42 + getattr(self, '_current_step', 0))


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

        rewards = [t.reward for t in trajectories]
        answer_rewards = [t.answer_reward for t in trajectories]
        n_steps = [t.n_steps for t in trajectories]

        h_flow = compute_flow_entropy(trajectories)


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

        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")


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

            for tt, acc in stats.get("task_rewards", {}).items():
                log_dict[f"acc/{tt}"] = acc
            self._wandb.log(log_dict, step=stats.get("step", 0))

    def _run_validation(self, step: int) -> None:

        import random as _rng
        val_rng = _rng.Random(42)  


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


        logger.info(f"[Validation] Step {step}: running {len(val_batch)} fixed val episodes...")
        val_trajs = self._collect_episodes(val_batch)


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

        import random
        dump_dir = self.output_dir / "trajectory_dumps"
        dump_dir.mkdir(exist_ok=True)
        dump_file = dump_dir / f"step_{step:04d}.jsonl"
        try:

            by_type: Dict[str, List] = {}
            for traj in trajectories:
                tt = getattr(traj, 'task_type', 'unknown')
                by_type.setdefault(tt, []).append(traj)
            samples = [random.choice(trajs) for trajs in by_type.values()]

            if len(samples) > 4:
                samples = random.sample(samples, 4)

            with open(dump_file, "w", encoding="utf-8") as f:
                for traj in samples:
                    f.write(traj.to_jsonl() + "\n")
            logger.info(f"Trajectory dump: {dump_file} ({len(samples)} samples from {len(by_type)} task types)")
        except Exception as e:
            logger.warning(f"Trajectory dump failed: {e}")

    def _save_checkpoint(self, step: int, final: bool = False) -> None:

        tag = "final" if final else f"step_{step:04d}"
        ckpt_dir = self.output_dir / f"checkpoint_{tag}"
        ckpt_dir.mkdir(exist_ok=True)


        self.shared_model.set_adapter("theta")
        self.shared_model.save_pretrained(
            ckpt_dir / "supervisor_lora",
            selected_adapters=["theta"],
        )


        for name, param in self.shared_model.named_parameters():
            if ".lora_A.phi" in name or ".lora_B.phi" in name:
                param.requires_grad_(True)


        self.backward_policy.save(str(ckpt_dir / "backward_lora"))


        torch.save(self.partition_fn.state_dict(), ckpt_dir / "partition_fn.pt")


        self.workspace.save_all()

        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def _sync_lora_to_vllm(self, step: int) -> None:

        try:
            import time as _time
            import requests
            from dataclasses import asdict
            from peft.utils import get_peft_model_state_dict
            from sglang.srt.utils import MultiprocessingSerializer

            _t0 = _time.time()


            self.shared_model.set_adapter("theta")
            lora_state = get_peft_model_state_dict(
                self.shared_model, adapter_name="theta"
            )

            for name, param in self.shared_model.named_parameters():
                if ".lora_A.phi" in name or ".lora_B.phi" in name:
                    param.requires_grad_(True)

            processed_weights = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in lora_state.items()
            }


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
                _time.sleep(0.3)  
                _t_drain = _time.time()


                try:
                    requests.post(
                        f"{api_base}/unload_lora_adapter",
                        headers=headers,
                        json={"lora_name": FIXED_ADAPTER_NAME},
                        timeout=60,
                    )
                except Exception as ue:
                    logger.debug(f"Unload (expected first): {ue}")


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


                if load_ok:
                    try:
                        requests.post(f"{api_base}/flush_cache", headers=headers, timeout=10)
                    except Exception as fe:
                        logger.debug(f"flush_cache (non-fatal): {fe}")


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


                if not load_ok:
                    logger.error(
                        f"LoRA sync UNRECOVERABLE at step {step}; pause event held to block rollouts"
                    )
                    return  
            finally:
                if load_ok:
                    _sglang_pause_event.clear()


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

        if not hasattr(self, "sglang_mgr") or self.sglang_mgr is None:
            logger.error("[SGLang Restart] sglang_mgr not initialized; cannot restart cleanly")
            return
        try:
            self.sglang_mgr.restart()
            logger.info("[SGLang Restart] child restarted and ready (via sglang_manager)")
        except Exception as e:
            logger.error(f"[SGLang Restart] sglang_manager.restart() failed: {e}")


    def resume(self, checkpoint_path: str) -> None:

        ckpt_dir = Path(checkpoint_path)
        logger.info(f"Resuming from {ckpt_dir}")


        theta_dir = ckpt_dir / "supervisor_lora" / "theta"
        if not theta_dir.exists():
            theta_dir = ckpt_dir / "supervisor_lora"  

        if theta_dir.exists() and (theta_dir / "adapter_config.json").exists():

            self.shared_model.delete_adapter("theta")
            self.shared_model.load_adapter(str(theta_dir), adapter_name="theta")
            self.shared_model.set_adapter("theta")

            for name, param in self.shared_model.named_parameters():
                if ".lora_A.phi" in name or ".lora_B.phi" in name:
                    param.requires_grad_(True)
            logger.info(f"θ-LoRA loaded from {theta_dir}")
        else:
            logger.warning(f"No θ-LoRA checkpoint found at {theta_dir}")


        if (ckpt_dir / "partition_fn.pt").exists():
            self.partition_fn.load_state_dict(
                torch.load(ckpt_dir / "partition_fn.pt", map_location=self.device)
            )

        logger.info("Resume complete")
