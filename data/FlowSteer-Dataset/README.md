---
license: apache-2.0
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - math
  - code
  - qa
  - evaluation
  - benchmark
size_categories:
  - 10K<n<100K
---

# FlowSteer Dataset

A comprehensive evaluation and training benchmark containing 12 evaluation datasets and 1 training dataset across 3 domains: Math, Code, and QA.

## Dataset Structure

```
├── train/                    # Training data
│   └── train_12k.jsonl      # 12,000 balanced training samples
└── eval/                     # Evaluation data
    ├── gsm8k.jsonl          # 128 samples
    ├── math.jsonl           # 128 samples
    ├── aime2025.jsonl       # 30 samples
    ├── mathqa.jsonl         # 128 samples
    ├── humaneval.jsonl      # 128 samples
    ├── mbpp.jsonl           # 128 samples
    ├── apps.jsonl           # 128 samples
    ├── ds1000.jsonl         # 128 samples
    ├── hotpotqa.jsonl       # 128 samples
    ├── squad_v2.jsonl       # 128 samples
    ├── nq.jsonl             # 128 samples
    └── triviaqa.jsonl       # 128 samples
```

## Dataset Overview

### Evaluation Datasets (1,438 samples)

| Dataset | Domain | Samples | Task Type | Description |
|---------|--------|---------|-----------|-------------|
| **GSM8K** | Math | 128 | Open-ended | Grade school math word problems |
| **MATH** | Math | 128 | Open-ended | Competition-level math problems |
| **AIME2025** | Math | 30 | Open-ended | AIME 2025 competition problems |
| **MathQA** | Math | 128 | Multiple Choice | Math word problems with 5 options |
| **HumanEval** | Code | 128 | Code Generation | Python function completion |
| **MBPP** | Code | 128 | Code Generation | Basic Python programming |
| **APPS** | Code | 128 | Code Generation | Competitive programming problems |
| **DS1000** | Code | 128 | Code Generation | Data science code completion |
| **HotpotQA** | QA | 128 | Extractive QA | Multi-hop reasoning questions |
| **SQuAD v2** | QA | 128 | Extractive QA | Reading comprehension with unanswerable |
| **NQ** | QA | 128 | Extractive QA | Natural Questions from Google |
| **TriviaQA** | QA | 128 | Extractive QA | Trivia questions with long context |

### Training Dataset (12,000 samples)

**Distribution by Source:**
- HotpotQA: 2,000 samples
- GSM8K: 2,000 samples
- MATH: 2,000 samples
- MBPP: 2,000 samples
- SQuAD v2: 2,000 samples
- HumanEval: 2,000 samples

**Distribution by Problem Type:**
- QA: 4,000 samples
- Math: 4,000 samples
- Code: 4,000 samples

## Data Format

All datasets use JSONL format with the following fields:

```json
{
  "problem": "The problem/question text",
  "problem_type": "math|code|qa|mathqa_mc",
  "source": "dataset_name",
  "ground_truth": "The expected answer",
  "meta": { ... }
}
```

### Domain-Specific Meta Fields

#### Math Tasks (GSM8K, MATH, AIME2025)
- `meta.full_solution`: Complete solution steps
- `meta.level`: Difficulty level (Level 1-5)
- `meta.type`: Math category (Algebra, Geometry, etc.)

#### Code Tasks (HumanEval, MBPP, APPS, DS1000)
- `meta.task_id`: Unique task identifier
- `meta.entry_point`: Function entry point name
- `meta.test`: Test cases for validation

#### QA Tasks (HotpotQA, SQuAD v2, NQ, TriviaQA)
- `meta.id`: Question identifier
- `meta.has_answer`: Whether the question is answerable
- `meta.all_answers`: List of acceptable answer variants

## Usage

```python
from datasets import load_dataset

# Load training dataset
train_data = load_dataset("beita6969/FlowSteer-Dataset", data_files="train/train_12k.jsonl")

# Load specific evaluation dataset
gsm8k = load_dataset("beita6969/FlowSteer-Dataset", data_files="eval/gsm8k.jsonl")

# Load all evaluation datasets
eval_datasets = {}
for name in ["gsm8k", "math", "humaneval", "mbpp", "apps", "ds1000",
             "hotpotqa", "squad_v2", "nq", "triviaqa", "mathqa", "aime2025"]:
    eval_datasets[name] = load_dataset("beita6969/FlowSteer-Dataset",
                                        data_files=f"eval/{name}.jsonl")
```

## Evaluation Metrics

| Domain | Primary Metric | Secondary Metric |
|--------|---------------|------------------|
| Math | Exact Match | Symbolic Equivalence |
| Code | Pass@1 | Test Pass Rate |
| QA | Exact Match | F1 Score |

## License

This dataset is released under Apache 2.0 license for research purposes. Individual datasets retain their original licenses.

## Citation

If you use this benchmark, please cite the original datasets:
- GSM8K: Cobbe et al., 2021
- MATH: Hendrycks et al., 2021
- HumanEval: Chen et al., 2021
- MBPP: Austin et al., 2021
- APPS: Hendrycks et al., 2021
- DS-1000: Lai et al., 2022
- HotpotQA: Yang et al., 2018
- SQuAD v2: Rajpurkar et al., 2018
- Natural Questions: Kwiatkowski et al., 2019
- TriviaQA: Joshi et al., 2017
- MathQA: Amini et al., 2019
