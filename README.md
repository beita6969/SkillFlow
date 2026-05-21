<div align="center">

# SkillFlow

### Flow-Driven Recursive Skill Evolution for Agentic Orchestration

SkillFlow trains a tool-using LLM supervisor with flow matching, backward credit assignment, and a self-evolving skill library.

<img src="assets/figures/skillflow_teaser.png" alt="SkillFlow teaser" width="88%">

</div>

## Overview

SkillFlow addresses three bottlenecks in agentic orchestration: strategy collapse under reward maximization, high-variance and opaque credit assignment, and unguided skill-library updates. The framework combines a trainable Supervisor, a frozen Executor, a learned backward policy, and a dynamic skill workspace. Tempered Trajectory Balance turns terminal outcome reward into trajectory-level flow supervision, while the learned backward policy exposes per-step diagnostic signals used to evolve skills.

<div align="center">
  <img src="assets/figures/skillflow_framework.png" alt="SkillFlow framework comparison" width="95%">
</div>

## Architecture

During rollout, the Supervisor selects actions such as tool calls, environment actions, skill invocation, and answer submission. The Executor performs delegated reasoning or tool execution. Training jointly updates the forward policy, backward policy, and partition head using TTB. At phase boundaries, flow diagnostics determine when to evolve the skill library, where decision gaps appear, and what skill edits should be made.

<div align="center">
  <img src="assets/figures/skillflow_architecture.png" alt="SkillFlow architecture" width="95%">
</div>

## Results and Diagnostics

SkillFlow is evaluated across question answering, mathematical reasoning, code generation, scientific QA, and interactive decision-making settings. The figures below summarize transfer behavior, pass@K/diversity, training reward curves, cost, and evolution timing from the arXiv manuscript.

<div align="center">
  <img src="assets/figures/skillflow_transfer_bars.png" alt="SkillFlow transfer performance by task type" width="95%">
</div>

<table>
<tr>
<td width="33%"><img src="assets/figures/skillflow_pass_at_k.png" alt="Pass@K and diversity"></td>
<td width="33%"><img src="assets/figures/skillflow_reward_curves.png" alt="Reward curves"></td>
<td width="33%"><img src="assets/figures/skillflow_cost_tokens.png" alt="Training and token cost"></td>
</tr>
<tr>
<td width="33%"><img src="assets/figures/skillflow_evolution_timing.png" alt="Evolution timing"></td>
<td width="33%"><img src="assets/figures/skillflow_transfer_reward_curve.png" alt="Transfer reward curve"></td>
<td width="33%"><img src="assets/figures/skillflow_transfer_loss_curve.png" alt="Transfer loss curve"></td>
</tr>
</table>

## Key Features

- Tempered Trajectory Balance for reward-proportional trajectory flow matching.
- Learned backward policy for transparent per-step credit assignment.
- Skill invocation as a first-class Supervisor action.
- Flow-driven recursive skill evolution using plateau triggers and curation signals.
- Outcome-only reward smoothing aligned with the manuscript formulation.
- LoRA-based Supervisor training with OpenAI-compatible local inference services.

## Repository Layout

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
assets/figures/                 arXiv figures used in this README
```

## Environment Setup

```bash
git clone https://github.com/beita6969/SkillFlow.git
cd SkillFlow

conda create -n skillflow python=3.10 -y
conda activate skillflow
pip install -r requirements.txt
```

Set local service variables:

```bash
export SGLANG_API_KEY=EMPTY
export SKILLFLOW_BASE_MODEL=/path/to/supervisor/base/model
export SKILLFLOW_EXECUTOR_MODEL=m_exec
```

If the Skill Creator LLM is served separately:

```bash
export SKILL_CREATOR_API_BASE=http://127.0.0.1:3456/v1/messages
export SKILL_CREATOR_MODEL=skill-creator-model
export SKILL_CREATOR_API_KEY=EMPTY
```

## Dataset

Prepared training data is hosted on Hugging Face:

```text
https://huggingface.co/datasets/beita6969/SkillFlow-Dataset
```

`configs/skillflow.yaml` expects the files at:

```text
data/train_v3.json
data/test_iid_v3.json
```

If the dataset repository is private, authenticate first with `hf auth login`. Then download the hosted files into the repository data directory:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="beita6969/SkillFlow-Dataset",
    repo_type="dataset",
    local_dir="data",
    allow_patterns=["train_v3.json", "test_iid_v3.json"],
    endpoint="https://huggingface.co",
    token=True,
)
PY
```

The paper-aligned dataset contains 3,500 training records and 798 IID validation records across code generation, WebShop, ALFWorld, mathematical reasoning, factual QA, science QA, and multi-hop QA. The train split is balanced to 500 records per IID benchmark family; SWE-bench is oversampled from 372 unique non-validation SWE-bench Verified training instances to 500 training records. The IID validation split uses 128 records for each non-AIME benchmark family and 30 official AIME 2026 records; TriviaQA validation contains 127 unique questions plus one deterministic duplicate to keep the public split balanced. The math training pool uses historical AIME problems from 1983-2025 to avoid training on the AIME 2026 validation items. The hosted files are intended for training and in-training IID validation, not the full 14-benchmark final evaluation suite.

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

Supported task types:

```text
multi_hop_qa, factual_qa, fact_checking, math_reasoning,
strategy_qa, code_generation, science_qa, interactive_agent
```

If the configured training file is absent, `run_training.py` creates a minimal fallback dataset so the pipeline can be checked end to end.

## Model Weights

Merged Supervisor weights are hosted on Hugging Face:

```text
https://huggingface.co/beita6969/SkillFlow-Model
```

The uploaded model is the merged `checkpoint_step_0110` Supervisor `theta` LoRA on top of Qwen/Qwen3.5-9B. The training-time backward policy adapter is not merged into the inference model.

## Quick Start

Start a frozen Executor service:

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

Check connectivity:

```bash
python run_training.py --config configs/skillflow.yaml --test-connectivity
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

## Important Configuration Fields

```yaml
base_model: "${SKILLFLOW_BASE_MODEL:-Qwen/Qwen3.5-9B}"
supervisor_api_base: "http://127.0.0.1:8005/v1"
executor_api_base: "http://127.0.0.1:8007/v1"
executor_model: "${SKILLFLOW_EXECUTOR_MODEL:-Qwen/Qwen3.5-9B}"
reward_mode: "outcome_only"
skill_mode: "policy_action"
ttb_edge_normalization: "per_token"
ttb_length_normalization: "steps"
plateau_window_size: 10
plateau_rho: 0.05
plateau_m_consecutive: 2
zeta_trig: 1.0
```

## Outputs

By default, training writes to:

```text
outputs/skillflow_general/
```

Important outputs:

```text
training_log.jsonl                 scalar logs and evolution events
trajectory_dumps/                  saved trajectories when enabled
skills/                            evolving skill workspace
checkpoint_step_XXXX/              model, LoRA, and optimizer checkpoints
```

## Method-to-Code Map

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
