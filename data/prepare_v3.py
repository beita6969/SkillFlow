
import json
import re
import random
import os
import pandas as pd
import glob as _glob
from pathlib import Path
from collections import Counter

random.seed(42)
BASE = Path(os.environ.get("SKILLFLOW_DATASETS_DIR", "datasets"))
OUT = Path(os.environ.get("SKILLFLOW_DATA_OUT", "data"))

train_all = []
eval_all = []

TARGET_TRAIN = 500
TARGET_EVAL = 128


def add_dataset(name, task_type, items, metric="exact_match"):

    random.shuffle(items)


    while len(items) < TARGET_TRAIN + TARGET_EVAL:
        items = items + items
        random.shuffle(items)

    eval_items = items[:TARGET_EVAL]
    train_items = items[TARGET_EVAL:TARGET_EVAL + TARGET_TRAIN]

    for item in train_items + eval_items:
        item["extra"]["source"] = name
        item["extra"]["metric"] = metric
        item["task_type"] = task_type

    train_all.extend(train_items)
    eval_all.extend(eval_items)
    print(f"  {name}: {len(train_items)} train + {len(eval_items)} eval ({len(items)} available)")


print("1. HotpotQA...")

df = pd.read_parquet(BASE / "hotpotqa/distractor/validation-00000-of-00001.parquet")
items = []
for _, row in df.iterrows():
    ctx = row.get("context", {})
    passages = []
    if isinstance(ctx, dict):
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", [])):
            passages.append(f"[{title}] " + " ".join(sents))
    elif isinstance(ctx, list):
        for c in ctx:
            if isinstance(c, list) and len(c) >= 2:
                passages.append(f"[{c[0]}] " + (" ".join(c[1]) if isinstance(c[1], list) else str(c[1])))

    q_text = f"Based on the following passages, answer the question.\n\n"
    for i, p in enumerate(passages[:10]):
        q_text += f"[{p[:300]}]\n\n"
    q_text += f"Question: {row['question']}"

    items.append({
        "question": q_text,
        "answer": str(row["answer"]),
        "task_type": "multi_hop_qa",
        "context": passages[:10],  
        "extra": {"type": row.get("type", ""), "level": row.get("level", "")}
    })
add_dataset("HotpotQA", "multi_hop_qa", items, metric="token_f1")


print("2. AIME...")
items = []
for pf in _glob.glob(str(BASE / "aime/**/*.parquet"), recursive=True):
    df = pd.read_parquet(pf)
    for _, row in df.iterrows():
        items.append({
            "question": str(row.get("problem", "")),
            "answer": str(row.get("answer", "")),
            "task_type": "math_reasoning",
            "context": [],
            "extra": {
                "url": str(row.get("url", "")),
            }
        })

seen = set()
unique = []
for item in items:
    key = item["question"][:100]
    if key not in seen:
        seen.add(key)
        unique.append(item)
items = unique
add_dataset("AIME", "math_reasoning", items, metric="exact_match")


print("3. ALFWorld...")
items = []
task_templates = [
    ("Pick up {obj} from {loc} and put it in/on {target}.", "put"),
    ("Examine {obj} with the desk lamp.", "examine"),
    ("Clean {obj} and put it in/on {target}.", "clean"),
    ("Heat {obj} and put it in/on {target}.", "heat"),
    ("Cool {obj} and put it in/on {target}.", "cool"),
]
objects = ["apple", "book", "pen", "cup", "knife", "plate", "potato", "tomato", "egg", "mug",
           "cloth", "soap", "candle", "pillow", "remote", "watch", "key", "phone", "vase", "bowl"]
locations = ["counter", "table", "shelf", "drawer", "cabinet", "desk", "bed", "sofa", "sink", "fridge"]

for i in range(700):
    template, task = random.choice(task_templates)
    obj = random.choice(objects)
    loc = random.choice(locations)
    target = random.choice([l for l in locations if l != loc])
    question = template.format(obj=obj, loc=loc, target=target)

    items.append({
        "question": (
            f"You are in a household environment. Complete this task:\n\n{question}\n\n"
            f"Available actions: go to [location], take [object] from [location], "
            f"put [object] in/on [location], open [object], close [object], "
            f"use [object], examine [object], look"
        ),
        "answer": f"Completed: {task} {obj}",
        "task_type": "interactive_agent",
        "context": [],
        "env_type": "alfworld",
        "env_config": {"mode": "train"},
        "extra": {"task": task, "object": obj}
    })
add_dataset("ALFWorld", "interactive_agent", items, metric="exact_match")


print("4. WebShop...")
with open(BASE / "webshop/baseline_models/data/human_goals.json") as f:
    goals = json.load(f)

items = []
for idx, goal in enumerate(goals):
    if isinstance(goal, str):
        goal_text = goal
    elif isinstance(goal, dict):
        goal_text = goal.get("goal", goal.get("instruction", str(goal)))
    else:
        goal_text = str(goal)

    items.append({
        "question": (
            f"You are shopping online. Find and buy the following item:\n\n{goal_text}\n\n"
            f"Available actions: search[query], click[element], buy"
        ),
        "answer": "Purchased successfully",
        "task_type": "interactive_agent",
        "context": [],
        "env_type": "webshop",
        "env_config": {"observation_mode": "text", "seed": idx},
        "extra": {"goal": goal_text[:200]}
    })
add_dataset("WebShop", "interactive_agent", items, metric="exact_match")


print("5. TriviaQA...")
items = []
for pf in sorted(_glob.glob(str(BASE / "triviaqa/rc/validation-*.parquet"))):
    df = pd.read_parquet(pf)
    for _, row in df.iterrows():

        answer = row.get("answer", {})
        if isinstance(answer, dict):
            ans_value = answer.get("value", "")
            aliases = answer.get("aliases", [])
        else:
            ans_value = str(answer)
            aliases = []
        all_answers = [ans_value] + (aliases if isinstance(aliases, list) else [])
        ans_str = " | ".join([a for a in all_answers if a][:5])


        passages = []
        ep = row.get("entity_pages", {})
        if isinstance(ep, dict):
            wiki_ctx = ep.get("wiki_context", [])
            wiki_titles = ep.get("title", [])
            if hasattr(wiki_ctx, '__len__'):
                for j, ctx in enumerate(wiki_ctx):
                    if ctx and len(str(ctx)) > 50:
                        title = wiki_titles[j] if j < len(wiki_titles) else f"Doc {j}"

                        passages.append(f"[{title}] {str(ctx)[:1000]}")

        sr = row.get("search_results", {})
        if isinstance(sr, dict):
            search_ctx = sr.get("search_context", [])
            search_titles = sr.get("title", [])
            if hasattr(search_ctx, '__len__'):
                for j, ctx in enumerate(search_ctx):
                    if ctx and len(str(ctx)) > 50:
                        title = search_titles[j] if j < len(search_titles) else f"Search {j}"
                        passages.append(f"[{title}] {str(ctx)[:800]}")


        passages = passages[:10]

        if not ans_str:
            continue

        q_text = str(row.get("question", ""))
        if passages:
            q_text = f"Based on the following passages, answer the question.\n\n"
            for p in passages[:5]:
                q_text += f"{p[:200]}\n\n"
            q_text += f"Question: {row['question']}"

        items.append({
            "question": q_text,
            "answer": ans_str,
            "task_type": "factual_qa",
            "context": passages,
            "extra": {}
        })
add_dataset("TriviaQA", "factual_qa", items, metric="token_f1")


print("6. SWE-bench...")


_REPO_CACHE_DIR = Path(os.environ.get("SWEBENCH_REPO_CACHE", str(BASE / "swe-bench-repos")))


def _ensure_repo_cloned(repo: str) -> Path:

    repo_dir = _REPO_CACHE_DIR / repo.replace("/", "__")
    if repo_dir.exists() and (repo_dir / ".git").exists():
        return repo_dir
    repo_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    print(f"    Cloning {repo}...")
    import subprocess
    subprocess.run(
        ["git", "clone", "--depth=1", "--no-single-branch", url, str(repo_dir)],
        capture_output=True, timeout=120,
    )
    return repo_dir


def extract_files_from_repo(repo: str, base_commit: str, file_paths: list) -> dict:

    import subprocess
    code_files = {}
    repo_dir = _REPO_CACHE_DIR / repo.replace("/", "__")
    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        return code_files

    for fp in file_paths[:3]:  
        try:
            result = subprocess.run(
                ["git", "show", f"{base_commit}:{fp}"],
                capture_output=True, text=True, timeout=5, cwd=str(repo_dir),
            )
            if result.returncode == 0 and result.stdout:
                code_files[fp] = result.stdout[:50000]
            else:

                subprocess.run(
                    ["git", "fetch", "origin", base_commit],
                    capture_output=True, timeout=15, cwd=str(repo_dir),
                )
                result = subprocess.run(
                    ["git", "show", f"{base_commit}:{fp}"],
                    capture_output=True, text=True, timeout=5, cwd=str(repo_dir),
                )
                if result.returncode == 0 and result.stdout:
                    code_files[fp] = result.stdout[:50000]
        except Exception:
            pass

    return code_files


def extract_pre_patch_files(patch_text, repo="", base_commit=""):


    affected_files = re.findall(r"diff --git a/(\S+)", patch_text)


    if repo and base_commit and affected_files:
        code_files = extract_files_from_repo(repo, base_commit, affected_files)
        if code_files:
            return code_files


    code_files = {}
    current_file = None
    current_lines = []

    for line in patch_text.split('\n'):
        m = re.match(r'^diff --git a/(\S+)', line)
        if m:
            if current_file and current_lines:
                code_files[current_file] = '\n'.join(current_lines)
            current_file = m.group(1)
            current_lines = []
            continue
        if line.startswith('---') or line.startswith('+++') or line.startswith('index '):
            continue
        if line.startswith('@@'):
            current_lines.append(f'# {line}')
            continue
        if line.startswith(' '):
            current_lines.append(line[1:])
        elif line.startswith('-'):
            current_lines.append(line[1:])
        elif line.startswith('+'):
            continue
        else:
            current_lines.append(line)

    if current_file and current_lines:
        code_files[current_file] = '\n'.join(current_lines)

    return code_files


df = pd.read_parquet(BASE / "swe-bench-data/data/test-00000-of-00001.parquet")
items = []
for i, (_, row) in enumerate(df.iterrows()):
    problem = str(row.get("problem_statement", ""))
    patch = str(row.get("patch", ""))
    repo = str(row.get("repo", ""))
    base_commit = str(row.get("base_commit", ""))


    code_files = extract_pre_patch_files(patch, repo=repo, base_commit=base_commit)

    items.append({
        "question": f"Fix the following software issue:\n\n{problem[:2000]}",
        "answer": patch[:2000],
        "task_type": "code_generation",
        "context": [],
        "code_files": code_files,
        "extra": {
            "repo": repo,
            "instance_id": str(row.get("instance_id", "")),
            "base_commit": base_commit,
        }
    })
    if (i + 1) % 100 == 0:
        print(f"    Processed {i+1} SWE-bench instances...")
add_dataset("SWE-bench", "code_generation", items, metric="exact_match")


print("7. MedQA...")
items = []
for split in ["phrases_no_exclude_train.jsonl", "phrases_no_exclude_test.jsonl"]:
    fpath = BASE / "medqa" / split
    if not fpath.exists():
        continue
    with open(fpath) as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question", "")
            options = row.get("options", {})
            answer_idx = row.get("answer_idx", "")
            answer_text = row.get("answer", "")


            option_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

            items.append({
                "question": f"{q}\n\nOptions:\n{option_text}",
                "answer": f"{answer_idx}. {answer_text}" if answer_idx else answer_text,
                "task_type": "science_qa",
                "context": [],
                "extra": {
                    "correct_option": answer_idx,
                    "meta_info": str(row.get("meta_info", ""))[:100],
                }
            })


seen = set()
unique = []
for item in items:
    key = item["question"][:100]
    if key not in seen:
        seen.add(key)
        unique.append(item)
items = unique
add_dataset("MedQA", "science_qa", items, metric="exact_match")


random.shuffle(train_all)
random.shuffle(eval_all)

with open(OUT / "train_v3.json", "w") as f:
    json.dump(train_all, f, ensure_ascii=False, indent=2)

with open(OUT / "test_iid_v3.json", "w") as f:
    json.dump(eval_all, f, ensure_ascii=False, indent=2)

print(f"\n=== 完成 ===")
print(f"训练集: {len(train_all)} 样本 → data/train_v3.json")
print(f"评估集: {len(eval_all)} 样本 → data/test_iid_v3.json")


train_sources = Counter(d["extra"]["source"] for d in train_all)
eval_sources = Counter(d["extra"]["source"] for d in eval_all)
print("\n训练集分布:")
for s, c in sorted(train_sources.items()):
    print(f"  {s}: {c}")
print("\n评估集分布:")
for s, c in sorted(eval_sources.items()):
    print(f"  {s}: {c}")


print("\n字段验证:")
for d in eval_all[:1] + train_all[:1]:
    src = d['extra']['source']
    ctx = d.get('context', [])
    cf = d.get('code_files', {})
    et = d.get('env_type', '')
    print(f"  {src}: context={type(ctx).__name__}(len={len(ctx) if isinstance(ctx, list) else '?'}), "
          f"code_files={'yes' if cf else 'no'}, env_type={et or 'none'}")
