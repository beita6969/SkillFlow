# SkillFlow

Flow-driven recursive skill evolution for agentic orchestration.

SkillFlow trains a tool-using LLM supervisor to solve multi-step tasks while growing a reusable skill library from its own trajectories. The method is described in arXiv:2605.14089, "SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration".

## Overview

SkillFlow is built around four components:

- a trainable Supervisor policy `π_θ` that chooses orchestration actions;
- a frozen Executor `M_exec` that carries out delegated reasoning, coding, search, and environment actions;
- a backward policy `P_φ` learned jointly with the forward policy for per-step credit assignment;
- a dynamic skill library that can add, update, retain, or prune skills during training.

The training objective is Tempered Trajectory Balance. It matches trajectory flow to smoothed outcome reward while using per-token-normalized edge log probabilities. The same flow quantities produce interpretable step scores such as `I(t)=π_θ/P_φ` and marginal skill flow estimates. These signals are then used by the recursive skill-evolution loop to decide when evolution is needed and which decisions should become reusable skills.

## Repository layout

```text
configs/skillflow.yaml          main training configuration
run_training.py                 training entry point
training/gflownet_trainer.py    TTB training loop, LoRA sync, skill evolution trigger
training/flow_metrics.py        flow, step-importance, and TTB diagnostics
training/backward_policy.py     backward policy P_phi
training/skill_evolution.py     plateau trigger, CGF curation, D/R/U partitioning
training/environment.py         tool-use environment and task handlers
training/reward.py              outcome-only reward and R_tilde smoothing
src/executor/m_exec.py          frozen executor API wrapper
src/skills/                    skill format, workspace, and skill creator
data/prepare_v3.py              dataset preparation script
scripts/                        utility scripts
```

## Environment setup

```bash
git clone https://github.com/beita6969/SkillFlow.git
cd SkillFlow

conda create -n skillflow python=3.10 -y
conda activate skillflow
pip install -r requirements.txt
```

For local OpenAI-compatible SGLang services, set:

```bash
export SGLANG_API_KEY=EMPTY
export SKILLFLOW_BASE_MODEL=/path/to/supervisor/base/model
export SKILLFLOW_EXECUTOR_MODEL=m_exec
```

If the Skill Creator LLM is served separately, set:

```bash
export SKILL_CREATOR_API_BASE=http://127.0.0.1:3456/v1/messages
export SKILL_CREATOR_MODEL=skill-creator-model
export SKILL_CREATOR_API_KEY=EMPTY
```

## Data format

`configs/skillflow.yaml` expects:

```text
data/train_v3.json
data/test_iid_v3.json
```

Each item should follow this schema:

```json
{
  "question": "task input shown to the supervisor",
  "answer": "reference answer or evaluator label",
  "task_type": "multi_hop_qa",
  "context": [],
  "extra": {
    "metric": "token_f1"
  }
}
```

Supported task types include:

```text
multi_hop_qa, factual_qa, fact_checking, math_reasoning,
strategy_qa, code_generation, science_qa, interactive_agent
```

If the configured train file is absent, `run_training.py` creates a minimal fallback dataset so the pipeline can be checked end to end.

## Start the executor service

The trainer starts the Supervisor SGLang process internally for LoRA hot swapping. The frozen Executor should be started before training. One local SGLang executor can be launched as:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /path/to/executor/model \
  --served-model-name "$SKILLFLOW_EXECUTOR_MODEL" \
  --port 8007 \
  --api-key "$SGLANG_API_KEY" \
  --context-length 32768 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
```

Then check connectivity:

```bash
python run_training.py --config configs/skillflow.yaml --test-connectivity
```

## Start training

Edit `configs/skillflow.yaml` for your GPU layout, data paths, model path, and batch size. Important fields are:

```yaml
base_model: "${SKILLFLOW_BASE_MODEL:-Qwen/Qwen3.5-9B}"
supervisor_api_base: "http://127.0.0.1:8005/v1"
executor_api_base: "http://127.0.0.1:8007/v1"
executor_model: "${SKILLFLOW_EXECUTOR_MODEL:-Qwen/Qwen3.5-9B}"
reward_mode: "outcome_only"
skill_mode: "policy_action"
ttb_edge_normalization: "per_token"
ttb_length_normalization: "steps"
```

Run a short training check:

```bash
python -u run_training.py \
  --config configs/skillflow.yaml \
  --max-steps 3 \
  --fresh
```

Run the default training configuration:

```bash
python -u run_training.py \
  --config configs/skillflow.yaml \
  --fresh
```

Resume from a checkpoint:

```bash
python -u run_training.py \
  --config configs/skillflow.yaml \
  --resume outputs/skillflow_general/checkpoint_step_XXXX
```

Run only initial skill generation:

```bash
python run_training.py \
  --config configs/skillflow.yaml \
  --genesis-only
```

## Outputs

By default, training writes to:

```text
outputs/skillflow_general/
```

Important outputs include:

```text
training_log.jsonl                 scalar logs and evolution events
trajectory_dumps/                  saved trajectories when enabled
skills/                            evolving skill workspace
checkpoint_step_XXXX/              model, LoRA, and optimizer checkpoints
```

## Method-to-code map

| Method concept | Code |
| --- | --- |
| Tempered Trajectory Balance | `training/gflownet_trainer.py`, `training/flow_metrics.py` |
| Per-token edge normalization | `edge_logprob_tilde` in `training/flow_metrics.py` |
| Backward policy credit assignment | `training/backward_policy.py` |
| Step importance `I(t)` | `edge_log_i` and flow diagnostics |
| Outcome-only reward with smoothing | `training/reward.py` |
| Skill-as-action interface | `skill_invoke` support in `training/environment.py` and `training/batch_inference.py` |
| Plateau-triggered evolution | `PlateauDetector` and `_try_evolve` |
| CGF / D-R-U skill curation | `training/skill_evolution.py`, `src/skills/skill_creator.py` |

## Citation

```bibtex
@misc{zhang2026skillflow,
  title={SkillFlow: Flow-Driven Recursive Skill Evolution for Agentic Orchestration},
  author={Mingda Zhang and Tiesunlong Shen and Haoran Luo and Wenjin Liu and Zikai Xiao and Erik Cambria and Xiaoying Tang},
  year={2026},
  eprint={2605.14089},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.14089}
}
```
