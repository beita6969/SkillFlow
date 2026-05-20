


GENERATE_SKILL_PROMPT = """You are a skilled AI agent architect. Analyze the trajectory and extract a reusable Standard Operating Procedure (SOP).

### Guiding Principles:
1. **Learn from Success and Failure**: Compare the successful and failed trajectories using only outcome/reward metadata.
   - From successful patterns: Extract the effective workflows and tool sequences.
   - From failed attempts or near-misses: Note what went wrong and why - these lessons are often more valuable.
2. **Keep It General**: Use placeholders instead of specific values from this trajectory. The skill should apply to ANY similar problem, not just this one.
   - For QA tasks: `[ENTITY]`, `[QUERY]`, `[KEYWORD]`, `[RELATIONSHIP]`
   - For math tasks: `[EQUATION]`, `[VARIABLE]`, `[EXPRESSION]`, `[CONSTRAINT]`, `[FORMULA]`
   - For code tasks: `[FUNCTION_NAME]`, `[INPUT_SPEC]`, `[ALGORITHM]`, `[DATA_STRUCTURE]`
   - NEVER include specific names, numbers, equations, or code from this trajectory.
3. **Capture Executable Knowledge**: Extract the core tool usage patterns as reusable templates. Focus on WHICH TOOLS to use in WHAT ORDER — this is more valuable than describing the problem.
4. **Brevity Matters**: Aim for ~500 words. Focus on what's actionable.

### Available Tools:
The AI Supervisor has these tools:
- **skill_invoke**: Invoke a learned strategy from the skill library (core SkillFlow mechanism)
- **think**: Quick reasoning without calling executor
- **plan**: Have the Executor create a step-by-step plan
- **decompose**: Break complex problem into 2-3 sub-questions
- **python_execute**: Have the Executor write and run Python code (math via sympy, numerical via numpy)
- **test_code**: Have the Executor write a function, run against test cases
- **analyze**: Have the Executor analyze data or reason about a specific aspect
- **search**: Search context passages by BM25+semantic hybrid matching
- **lookup**: Look up a keyword in previously retrieved documents
- **fact_verify**: Verify a claim against context passages (SUPPORTED/NOT_SUPPORTED/PARTIAL)
- **ask_llm**: Ask a powerful LLM to directly answer based on gathered evidence
- **self_consistency**: Generate 3 independent solutions, return majority answer
- **verify_answer**: Verify candidate answer by substitution/testing
- **check_answer**: Quick sanity check on answer format and plausibility
- **cross_validate**: Solve using a different method, compare with candidate
- **search_code**: Search for patterns in the code workspace (SWE tasks)
- **view_file**: View file contents in the code workspace
- **edit_file**: Edit a file by replacing content
- **run_tests**: Run test commands in the code workspace
- **act**: Execute an action in an interactive environment (ALFWorld)
- **search_product**: Search products in WebShop environment
- **click**: Click an element in WebShop environment
- **accept**: Submit final answer

### Output Structure:
```
---
name: [SkillName]
description: |
  [Clear, concise description of what this skill does and when to use it. 1-2 sentences focusing on the core purpose.]
task_type: {task_type}
---

# [Skill Title]

## When to Use
- **Task patterns**: [what kinds of questions or problems, using generic descriptions]
- **Key signals**: [what clues in the question indicate this approach]

## Strategy Overview
[1-2 sentences on the core approach]

## Workflow
1. **[Phase Name]**: [Action using specific tool and rationale]
2. **[Phase Name]**: [Action and rationale]
3. ...

## Tool Templates
(Include only if the trajectory contained useful tool usage patterns)

- **[Tool] - [Purpose]**:
  `[tool_name]("[QUERY/TARGET placeholder]")`

## Watch Out For
- [Common mistake or trap observed in the trajectory, generalized]
```

### Input:
<successful_trajectory>
{success_trajectory}
</successful_trajectory>

<failed_trajectory>
{failure_trajectory}
</failed_trajectory>

<outcome_summary>
{outcome_summary}
</outcome_summary>

Learn from BOTH trajectories: the successful one shows what works, the failed one shows what to avoid.
Use outcome/reward metadata to diagnose failure patterns (wrong format? unsupported conclusion? gave up?) without copying any training label.
Output ONLY the SKILL.md content starting with `---`. Do NOT include any specific names, dates, numbers, or answers from the trajectories — use [PLACEHOLDER] notation instead."""


MERGE_SKILL_PROMPT = """You are a knowledge architect. Your job is to maintain a single, unified skill document for {task_type} tasks that grows wiser with each new case.

### Philosophy:
Think of this as a living document. Each new skill brings potential insights — your task is to integrate them thoughtfully, not mechanically.

### Integration Strategy:
For each part of the new skill, ask:
- **Is this part better?** → Rewrite the existing version
- **Is this part redundant or too specific?** → Delete it
- **Is this part complementary?** → Merge into a more general form
- **Is this part genuinely different?** → Add as a variant workflow (but consolidate if possible)

### Quality Guidelines:
- Preserve concrete, reusable tool usage templates and patterns
- Delete overly specific examples and cases that don't apply to similar problems
- Replace any specific names, numbers, or answers with [PLACEHOLDER] notation
- Consolidate similar trigger phrases
- If workflows differ only in minor details, merge them into one with noted variations

### Length Budget:
- Target: ~800 words
- If growing too long: merge similar workflows, trim verbose explanations
- Maximum 4 variant workflows - if you have more, they likely can be consolidated

### Available Tools:
skill_invoke, think, plan, decompose, python_execute, test_code, analyze, search, lookup, fact_verify, ask_llm, self_consistency, verify_answer, check_answer, cross_validate, search_code, view_file, edit_file, run_tests, act, search_product, click, accept

### Input:
<existing_skill>
{existing_skill}
</existing_skill>

<new_skills>
{new_skills}
</new_skills>

Output ONLY the merged SKILL.md starting with `---`. No preamble. Use [PLACEHOLDER] notation for any specific values."""


SKILL_REFINE_PROMPT = """You are a skill document architect. Refine the SKILL.md to remove redundancy, generalize specific cases, and improve structure.

### Current Stats:
- Word count: {word_count}
- Task type: {task_type}

### Refinement Goals:

1. **Remove Redundancy**:
   - Merge duplicate or near-duplicate content across sections
   - Eliminate repeated explanations that appear in multiple places
   - Consolidate overlapping concepts into single, clearer statements

2. **Avoid Too Specific Cases**:
   - Replace overly specific examples with generalizable patterns
   - Convert hardcoded values to placeholders (e.g., `[TARGET]`, `[QUERY]`, `[ENTITY]`)
   - Delete task-specific details or specific cases that don't apply to similar problems
   - Remove any leaked training data (specific question text, answers, names)

3. **Logical Consolidation**:
   - Merge workflows that share substantial overlap into variants
   - Extract common preliminary steps into dedicated sections
   - Group related tool templates and patterns together
   - Consolidate similar pitfalls into broader categories

4. **Format Optimization**:
   - Ensure consistent structure: When to Use → Workflow → Tool Templates → Watch Out For
   - Make workflows easier to scan (clear steps, consistent formatting)
   - Organize content from general principles → specific techniques

5. **Content Quality**:
   - Keep description concise and focused on core purpose
   - Ensure all content is actionable and reusable
   - Remove verbose explanations that don't add value
   - Maintain the most essential and distinctive elements

### Principles:
- Prioritize generalizability over specificity
- Keep what enables reuse across similar problems
- Remove what only applies to one particular case
- Maintain clarity and actionability
- Target ~600 words after refinement

Output ONLY the refined SKILL.md starting with `---`. No preamble.

<current_skill>
{skill_content}
</current_skill>
"""


EVOLUTION_SKILL_PROMPT = """You are a skilled AI agent architect. Analyze the trajectories and extract reusable Standard Operating Procedures (SOPs).

### Guiding Principles:
1. **Learn from Success AND Failure**: Use trajectory outcomes and rewards to infer what went wrong or what worked. Do not copy dataset labels into skills.
2. **Keep It General**: Use placeholders like [TARGET], [QUERY], [ENTITY] instead of specific values. The skill should apply to similar problems, not just these ones.
3. **Capture Executable Knowledge**: Focus on WHICH TOOLS to use in WHAT ORDER. Tool templates are more valuable than descriptions.
4. **Brevity Matters**: Each skill ~200 words. Focus on what's actionable.

### Available Tools:
skill_invoke, think, plan, decompose, python_execute, test_code, analyze, search, lookup, fact_verify, ask_llm, self_consistency, verify_answer, check_answer, cross_validate, search_code, view_file, edit_file, run_tests, act, search_product, click, accept

### Failed Trajectories (need better strategies):
{failure_section}

### Successful Trajectories (patterns to learn from):
{success_section}

### Current Skills (avoid duplicating):
{existing_titles}

{experience_section}
{struggling_section}

Generate 2-4 NEW skills. Output ONLY the SKILL.md content:

---
name: [SkillName]
description: |
  [What this skill does and when to use it. 1-2 sentences.]
task_type: [task_type]
---
## When to Use
- [what kinds of problems, using generic descriptions]

## Workflow
1. **[Phase]**: [tool] → [goal with placeholders]
2. **[Phase]**: [tool] → [goal with placeholders]
3. ...

## Watch Out For
- [pitfall learned from the failures above]
---

Generate now:"""


def clean_skill_output(text: str) -> str:

    import re

    text = re.sub(r'(\d)\uFE0F\u20E3', r'\1.', text)

    text = text.replace('★', '*')
    text = text.replace('⭐', '*')

    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    return text


def _classify_question_subtype(task_type: str, question: str) -> str:

    q_lower = question.lower()

    if task_type == "math_reasoning":
        if any(w in q_lower for w in ["sequence", "series", "a_1", "a_n"]):
            return "sequence/series problem"
        if any(w in q_lower for w in ["triangle", "angle", "circle", "polygon"]):
            return "geometry problem"
        if any(w in q_lower for w in ["probability", "random", "dice", "card", "coin"]):
            return "probability/combinatorics problem"
        if any(w in q_lower for w in ["equation", "solve", "root", "factor"]):
            return "algebra/equation problem"
        if any(w in q_lower for w in ["modulo", "mod ", "prime", "divisor", "digit"]):
            return "number theory problem"
        return "mathematical reasoning problem"

    if task_type == "code_generation":
        if any(w in q_lower for w in ["tree", "node", "binary"]):
            return "tree/graph algorithm"
        if any(w in q_lower for w in ["string", "substring", "palindrom"]):
            return "string processing"
        if any(w in q_lower for w in ["sort", "search", "array", "list"]):
            return "array/sorting algorithm"
        if any(w in q_lower for w in ["dynamic", "dp ", "minimum cost", "maximum"]):
            return "dynamic programming problem"
        return "function implementation"

    if task_type == "multi_hop_qa":
        if any(w in q_lower for w in ["who ", "person", "actor", "director"]):
            return "person-entity linking question"
        if any(w in q_lower for w in ["when ", "year", "date", "founded"]):
            return "temporal/date question"
        if any(w in q_lower for w in ["where", "located", "capital", "country"]):
            return "location/geography question"
        return "multi-hop factual question"

    return f"{task_type} problem"


def format_trajectory_for_skill_generation(traj, hide_question: bool = False) -> str:

    lines = [
        f"Task type: {traj.task_type}",
        f"Question: {(traj.question or '')[:200]}",
        f"Result: {'SUCCESS' if traj.reward >= 0.5 else 'FAILURE'} (reward={traj.reward:.2f})",
        f"Total steps: {len(traj.turns)}",
        "",
    ]


    prev_instr = None
    repeat_count = 0

    for i, turn in enumerate(traj.turns[:8]):
        action = turn.action_type
        instr = (turn.instruction or "")[:120]
        obs = (turn.observation or "")[:200]


        if action == prev_instr and action not in ("skill_invoke", "accept"):
            repeat_count += 1
            if repeat_count >= 2:
                lines.append(f"  Step {i+1}: [{action}] (REPEATED same instruction × {repeat_count+1})")
                continue
        else:
            repeat_count = 0
        prev_instr = action


        imp = getattr(turn, "step_importance", 1.0)
        marker = "★ " if imp > 1.5 else ""

        lines.append(f"  {marker}Step {i+1}: [{action}] {instr}")
        if obs:
            lines.append(f"    → {obs}")
        lines.append("")


    if traj.final_answer:
        lines.append(f"Final answer: {(traj.final_answer or '')[:150]}")

    return "\n".join(lines)


def format_trajectory_steps_only(traj, max_steps: int = 5) -> str:

    steps = []
    for i, turn in enumerate(traj.turns[:max_steps]):
        action = turn.action_type
        instr_preview = (turn.instruction or "")[:50]
        imp = getattr(turn, "step_importance", 1.0)
        marker = "★" if imp > 1.5 else " "
        steps.append(f"  {marker} Step {i+1}: [{action}] {instr_preview}")
    return "\n".join(steps)
