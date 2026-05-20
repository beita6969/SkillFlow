
import os, json
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from openai import OpenAI


client = OpenAI(
    base_url=os.environ.get("SKILLFLOW_EXECUTOR_API_BASE", "http://127.0.0.1:8007/v1"),
    api_key=os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY"),
)
MODEL_NAME = os.environ.get("SKILLFLOW_EXECUTOR_MODEL", "Qwen/Qwen3.5-9B")


task_type = "code_generation"
tool_list = "search_code, view_file, edit_file, python_execute, run_tests, list_files, verify_fix"

existing_tip = """tip-code-generation-1775904986-0:
  trigger: "When editing source files, search targeted terms first"
  body: "Use search_code with specific function/class names from the issue, then view_file
  the relevant file. After 2-3 exploration steps, commit to edits."
  flow_score: -2.30 (low — newly created, not yet validated)
  usage_count: 4, success_count: 2"""


success_evidence = """τ_3 (R̃=1.48, RESOLVED, 15 steps):
  S1:  list_files          log I(t)=+13.75 ★★ EXPLORE  — understood project structure first
  S3:  view_file           log I(t)= -1.00 → NEUTRAL   — read the target file
  S4:  python_execute      log I(t)=-20.00 ◆◆ KEY STEP — reproduced bug (backward strongly approved)
  S5:  python_execute      log I(t)= +3.50 ★ RISKY     — tried another approach
  S6:  search_code         log I(t)= -1.75 ◆ CONFIRMED — searched for related code
  S8:  view_file           log I(t)=-81.00 ◆◆ KEY STEP — read full file context (critical for understanding)
  S9:  edit_file           log I(t)= +1.00 → NEUTRAL   — first edit attempt (calm, balanced)
  S10: edit_file           log I(t)= +0.75 → NEUTRAL   — second edit (fixing LINT error)
  S12: edit_file           log I(t)=+18.00 ★★ EXPLORE  — KEY FIX: bold, correct edit
  S13: edit_file           log I(t)= +4.75 ★ RISKY     — refinement edit
  S15: edit_file           log I(t)= +5.00 ★ RISKY     — final polish

  Pattern: list→view→REPRODUCE(◆◆)→search→view(◆◆)→edit→edit→edit(★★)→edit→edit"""


failure_evidence = """τ_4 (R̃=0.13, FAILED, 15 steps):
  S1:  search_code         log I(t)=+27.50 ★★ — aggressive search
  S2:  view_file           log I(t)= -7.25 ◆◆ — read file (backward approved)
  S6:  python_execute      log I(t)= -6.00 ◆◆ — tried reproduce (backward approved the attempt)
  S7:  search_code         log I(t)=+18.00 ★★ — MORE searching (already searched!)
  S9:  python_execute      log I(t)= -6.00 ◆◆ — another reproduce attempt
  S11: view_file           log I(t)= +7.75 ★★ — still viewing files at step 11!
  S12: search_code         log I(t)=+16.50 ★★ — STILL searching at step 12!
  S13: edit_file           log I(t)=-22.00 ◆◆ KEY — first edit at step 13 (too late! got LINT error)
  S14: view_file           log I(t)= +6.37 ★★ — viewing after failed edit
  S15: view_file           log I(t)= +0.00 → — ran out of steps

  Pattern: search→view→...search→search→python→search→search→edit(LINT!)→view→view
  Problem: 13 steps of exploration, only 1 edit attempt at step 13 → LINT error → no time to fix"""


critical_steps = """Flow-derived critical decision points:

1. τ_3 Step 4 (python_execute, log I=-20.00):
   - Reproduced the bug early (step 4 of 15)
   - Backward policy STRONGLY approved (P_φ >> π_θ)
   - This early bug confirmation enabled focused editing later

2. τ_3 Step 8 (view_file, log I=-81.00):
   - Read the FULL file context before editing
   - Backward policy's strongest approval in all trajectories
   - Having complete context → edits S9-S15 were all on target

3. τ_4 Step 13 (edit_file, log I=-22.00):
   - First and ONLY edit, at step 13/15
   - Backward policy approved (this edit WAS necessary)
   - But too late — got LINT error, no steps left to fix

4. τ_3 Step 12 (edit_file, log I=+18.00):
   - The key fixing edit in the successful trajectory
   - Forward policy was bold (π_θ >> P_φ)
   - This is the creative step that actually solved the bug"""


dag_comparisons = """Same question, 4 trajectories:
  τ_3 (R̃=1.48 ✅): list→view→reproduce→search→view→edit×5 (edits from step 9)
  τ_1 (R̃=0.59 ⚠️): search→view→reproduce→edit→view→search→run_tests×6 (too much testing)
  τ_2 (R̃=0.20 ❌): search→view→reproduce→view×2→edit→view×2→python×3→edit×2 (LINT errors)
  τ_4 (R̃=0.13 ❌): search→view→list→view×2→python→search→view→python×2→view→search→edit(LINT)→view×2

Key divergence:
  - τ_3 starts editing at step 9 → 6 edit attempts → succeeds
  - τ_4 starts editing at step 13 → 1 edit attempt → LINT error → fails
  - Reward gap: 1.35 (huge)

Success pattern: reproduce early (step 4) + read full context (step 8) + start editing by step 9
Failure pattern: excessive search/view cycles consuming 80%+ of step budget"""


print("=" * 90)
print("PHASE 1: LLM Curator 审视已有 tip + flow 证据")
print("=" * 90)
print()

curation_prompt = f"""You are curating the tip library for {task_type} tasks.
A "tip" is a reusable tool-calling strategy that helps an agent fix software bugs.

## Current tips for {task_type}
{existing_tip}

## Available tools
{tool_list}

## Evidence from recent training (with GFlowNet per-step credit I(t))

### Successful trajectories (high reward)
{success_evidence}

### Failed trajectories (low reward)
{failure_evidence}

### Critical decision points (backward policy I(t) analysis)
{critical_steps}

### Same-question trajectory comparisons (DAG analysis)
{dag_comparisons}

## Your task
Review the existing tip against this flow evidence. Decide: KEEP, UPDATE, or DELETE it.
Then decide if a NEW tip is needed to capture the pattern revealed by the flow analysis.

The flow signals tell you:
- log I(t) << 0 (◆◆): backward policy strongly approved this step AFTER seeing the result → KEY STEP
- log I(t) >> 0 (★★): forward policy explored boldly → CREATIVE/RISKY step
- log I(t) ≈ 0: forward and backward agreed → ROUTINE step

```yaml
verdict:
  actions:
    - action: "KEEP" or "UPDATE" or "DELETE"
      skill_id: "tip-code-generation-1775904986-0"
      reason: "brief reason"
      new_body: "only if UPDATE"
  needs_new_tip: true/false
  new_tip_focus: "what the new tip should capture"
```

Output ONLY the YAML block."""

print("Sending to model for curation verdict...")
print()

resp = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": curation_prompt}],
    max_tokens=800,
    temperature=0.3,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

curation_result = resp.choices[0].message.content
print("Curation verdict:")
print(curation_result)


print()
print("=" * 90)
print("PHASE 2: 基于 flow 信号生成新 tip")
print("=" * 90)
print()

evidence_summary = f"""
Success pattern (from τ_3, R̃=1.48):
  Step 4 python_execute (log I=-20, ◆◆ KEY): reproduce bug early
  Step 8 view_file (log I=-81, ◆◆ KEY): read FULL file context before editing
  Steps 9-15 edit_file: focused editing with balanced I(t) ≈ 0-5

Failure pattern (from τ_4, R̃=0.13):
  Steps 1-12: excessive search/view/python (consuming 80% budget)
  Step 13 edit_file (log I=-22, ◆◆ KEY but too late): only 1 edit, LINT error

Credit decomposition shows:
  - The reproduce step (log I=-20) contributes 35% of τ_3's total negative credit
  - The full-file read (log I=-81) contributes 141% (dominant single step)
  - edit_file steps contribute positive credit when successful (+18, +5, +5)

This means: the agent should invest 2-3 steps in understanding (reproduce + read full context),
then commit 5+ steps to editing. Do NOT spend more than 4 steps on search/view before editing."""

gen_prompt = f"""You are generating ONE tip for {task_type} tasks.

## Existing tips (DO NOT duplicate)
{existing_tip}

## What this new tip should capture
The flow credit analysis reveals that successful bug-fixing follows a strict "understand fast, edit extensively" pattern.
The key insight from backward policy (I(t)) is: reproducing the bug early and reading full file context are the two most
credit-worthy steps, while excessive searching is the primary failure mode.

## Evidence
{evidence_summary}

## Available tools
{tool_list}

Generate exactly ONE tip. It must describe a concrete workflow with tool names, step counts, and timing.

```yaml
tip:
  description: "when to use this pattern (trigger condition)"
  body: "concrete tool sequence with step budget allocation"
```

Rules:
- Use ACTUAL tool names: {tool_list}
- Include TIMING guidance (by which step to start editing)
- Reference the flow signals (which steps matter most)
- 50-120 words for the body
- Must be DIFFERENT from the existing tip above

Output ONLY the YAML block."""

print("Sending to model for tip generation...")
print()

resp2 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[{"role": "user", "content": gen_prompt}],
    max_tokens=500,
    temperature=0.3,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

tip_result = resp2.choices[0].message.content
print("Generated tip:")
print(tip_result)


print()
print("=" * 90)
print("COMPLETE PIPELINE: trajectory → flow analysis → per-node credit → skill evolution")
print("=" * 90)
print()
print("1. 4 trajectories collected on same SWE-bench issue (xarray #4966)")
print("2. Per-node flow F(s_t) computed via TTB: log F(s_t) = log Z + Σ log I(k)")
print("3. Step importance I(t) = π_θ/P_φ decomposed total reward into per-step credits")
print("4. Flow signals identified:")
print("   - Key steps: reproduce bug (log I=-20) + read full context (log I=-81)")
print("   - Failure mode: excessive search consuming 80% step budget")
print("5. LLM Curator reviewed existing tip against flow evidence → verdict")
print("6. New tip generated targeting the flow-identified pattern")
print()
print("This is the complete SkillFlow loop: GFlowNet → Flow Signals → Skill Evolution → Better Policy")
