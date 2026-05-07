# SkillFlow

> Trajectory-Balance GFlowNet for autonomous skill discovery and evolution in LLM agents.

[![Status](https://img.shields.io/badge/status-WIP-yellow)]()
[![Paper](https://img.shields.io/badge/paper-NeurIPS%202026-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🚧 Repo Placeholder

This repository will host the official codebase, training logs, and evaluation
artifacts for **SkillFlow** — a Trajectory-Balance (TTB) GFlowNet framework
that lets LLM agents propose, validate, and evolve their own skill library
during reinforcement-style training.

Code release timeline:

- [ ] **2026-Q2** — clean training entry-point + Qwen3.5-9B reference checkpoint
- [ ] **2026-Q2** — 14-dataset cross-LLM evaluation harness (TriviaQA, HotpotQA, MedQA, AIME-2026, WebShop, ALFWorld, SWE-bench, GPQA Diamond, NQ-Open, MATH-Hard, MuSiQue, HumanEval, ScienceWorld, Mind2Web)
- [ ] **2026-Q3** — skill workspace dump + per-step I(t) / F̂(s) traces

## ✨ Highlights

- **Per-step flow attribution** via I(t) = exp(log π_θ − log P_φ) — surfaces critical decisions
- **Skill evolution** driven by F̂(s) marginal flow + DAG counterfactual pairs
- **Anytime workspace mutation** (ADD / UPDATE / DELETE / SPLIT) by a Claude curator
- **TTB stabilization** for long-horizon agentic trajectories (16-100 steps)

## 📊 Headline Numbers (preview)

| Dataset                | Best baseline | SkillFlow |
|------------------------|--------------:|----------:|
| GPQA Diamond           |   93.75%      |    TBD    |
| MATH-Hard              |   96.88%      |    TBD    |
| MuSiQue Ans EM         |   87.50%      |    TBD    |
| ScienceWorld Success   |   56.25%      |    TBD    |
| Mind2Web Step Acc      |   ~50%        |    TBD    |

(Baselines from claude-sonnet-4-6 / claude-haiku-4-5 / kimi-k2-0905 / gpt-5.4-mini / glm-5.1 / deepseek-v4-pro / gemini-3-flash, 32 samples per dataset.)

## 📜 Citation

```bibtex
@article{skillflow2026,
  title  = {SkillFlow: Self-Evolving Skill Libraries for LLM Agents via Trajectory-Balance GFlowNets},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## 📬 Contact

Maintainer: [@beita6969](https://github.com/beita6969)

---

_Detailed documentation, datasets, and training scripts will land in this repo as the paper progresses through review._
