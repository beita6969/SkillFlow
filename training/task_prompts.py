"""
Per-task-type system prompts and tool configurations.

Each task_type has:
- system_prompt: tailored role description + rules
- tools: list of task tools the model should see (answer auto-added).
  In paper-aligned `skill_mode=policy_action`, environment adds skill_invoke
  dynamically when the skill library is non-empty; legacy prompt-injection mode
  keeps tips in the system prompt instead.

v7 重构: answer 是正规工具 (action_type="answer" via tool call with 'response' arg)。
       不再使用 <answer>X</answer> 文本标签。
       动态过滤: trajectory 未用过工具前, answer tool 不在列表中 (rollout 层)。

WebShop/ALFWorld use ReAct text format (react_prompts.py), not tool calling.
"""

# ═══════════════════════════════════════════════════════
# code_generation (SWE-bench) — SWE-agent ACI style
# ═══════════════════════════════════════════════════════

CODE_GENERATION = {
    "max_episode_steps": 28,
    "system_prompt": (
        "Fix a bug in this repository using the provided source tools. "
        "The evaluator submits the workspace diff at the end.\n"
        "Each tool result may include SWE_MEMORY: a compact history of previous searches, viewed source lines, and edit outcomes. Treat it only as episode memory.\n"
        "Rules: describe edits in natural language (never write replacement source or exact diff text). "
        "Cite function name or unique substring so M_exec targets the edit. "
        "If you are not fully certain about the implementation, describe the desired behavior and local anchor rather than guessing exact grouping/order/control-flow. "
        "When formatting/quoting is subtle, state the intended behavior instead of giving a literal replacement format string. "
        "Never edit test files. Make minimal changes. Never repeat the same instruction. "
        "Keep track of the current best source candidate, the evidence observed for it, and any remaining uncertainty. "
        "When a viewed source line directly matches the issue mechanism and expected behavior is clear, use that evidence for a minimal edit instead of spending the remaining budget proving unrelated base-class/language internals. "
        "Repeated searches/views are recorded in memory and add no new evidence. "
        "After edit_file reports [OK], the returned updated snippet is the current source state and the workspace diff is non-empty. "
        "Do not use synthetic reproductions or example datasets as evidence; inspect the real source files."
    ),
    "tools": [
        "search_code", "view_file", "edit_file",
        "list_files",
        # code eval uses workspace diff; answer tool is intentionally hidden.
    ],
}

# ═══════════════════════════════════════════════════════
# math_reasoning (GSM8K / MATH) — PAL style
# ═══════════════════════════════════════════════════════

MATH_REASONING = {
    "max_episode_steps": 8,
    "system_prompt": (
        "Solve math problems by describing computations. M_exec writes and runs the Python.\n"
        "Rules: describe in natural language — never write source code. "
        "If a result looks wrong, rephrase the description; don't repeat verbatim. "
        "Finalize with answer(response=<number>). AIME answers are integers 0-999."
    ),
    "tools": [
        "python_execute", "verify_answer", "self_consistency", "decompose",
        "answer",
    ],
}

# ═══════════════════════════════════════════════════════
# multi_hop_qa (HotPotQA) — ReAct search chain
# ═══════════════════════════════════════════════════════

MULTI_HOP_QA = {
    "max_episode_steps": 6,
    "system_prompt": (
        "Chain evidence across passages to answer multi-hop questions. "
        "Search for specific entity names (not the full question). "
        "If [NO_MATCH] or [REPEATED], pivot with synonyms. "
        "Answer briefly: name, date, number, or short phrase. "
        "Finalize with answer(response=<your answer>)."
    ),
    "tools": [
        "search", "lookup", "decompose", "fact_verify", "answer",
    ],
}

# ═══════════════════════════════════════════════════════
# factual_qa (TriviaQA / NQ) — RAG retrieval
# ═══════════════════════════════════════════════════════

FACTUAL_QA = {
    "max_episode_steps": 5,
    "system_prompt": (
        "Extract the answer from provided passages. Don't guess from memory. "
        "Answer concisely: name, place, number, or short phrase. "
        "Finalize with answer(response=<your answer>)."
    ),
    "tools": [
        "search", "lookup", "fact_verify", "answer",
    ],
}

# ═══════════════════════════════════════════════════════
# science_qa (MedQA / MMLU-Medical) — Medical RAG
# ═══════════════════════════════════════════════════════

SCIENCE_QA = {
    "max_episode_steps": 8,
    "system_prompt": (
        "Answer medical multiple-choice by searching the textbook database. "
        "Use medical terminology, not lay terms. "
        "For differential dx, search the most distinguishing symptom. "
        "For calculations, describe in NL via python_execute (never pass source code). "
        "Finalize with answer(response=<A|B|C|D|E>)."
    ),
    "tools": [
        "search", "lookup", "python_execute", "analyze", "answer",
    ],
}

# ═══════════════════════════════════════════════════════
# webshop / alfworld — ReAct text format (no tool calling)
# Prompts managed by training/react_prompts.py
# ═══════════════════════════════════════════════════════

WEBSHOP = {
    "max_episode_steps": 10,
    "react": True,
    "tools": [],
}

ALFWORLD = {
    "max_episode_steps": 20,
    "react": True,
    "tools": [],
}

# ═══════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════

TASK_CONFIGS = {
    "code_generation": CODE_GENERATION,
    "math_reasoning":  MATH_REASONING,
    "multi_hop_qa":    MULTI_HOP_QA,
    "factual_qa":      FACTUAL_QA,
    "science_qa":      SCIENCE_QA,
    "webshop":         WEBSHOP,
    "alfworld":        ALFWORLD,
}
