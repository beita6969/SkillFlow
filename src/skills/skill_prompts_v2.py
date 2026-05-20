


GENERATE_TIP_PROMPT = """Compare these two trajectories for a {task_type} task and extract the KEY DIFFERENCE that made one succeed and the other fail.

<success>
{success_trajectory}
</success>

<failure>
{failure_trajectory}
</failure>

<outcome_summary>{outcome_summary}</outcome_summary>

Output 1-2 atomic tips. Each tip MUST be:
- Under 60 words
- Format: "When [situation], [do X] instead of [Y]." or "Pitfall: [mistake]. Fix: [correct approach]."
- About a NON-OBVIOUS decision (don't state things like "use tools" or "read the question carefully")
- Generalizable (use [ENTITY], [QUERY] placeholders, never specific names/numbers)

Output format (YAML):
```yaml
tips:
  - trigger: "[task_type] task where [specific situation]"
    tip: "[actionable advice, under 60 words]"
```

Output ONLY the YAML block. No preamble."""


GENERATE_TIP_FROM_OBSERVATIONS_PROMPT = """You are analyzing how an agent orchestrates tools on {task_type} tasks.

Available tools for {task_type}:
{tool_list}

These are CRITICAL steps where the agent made good tool choices:
{observations}

Compare with FAILED trajectories where the agent chose wrong tools or wrong order:
{failed_contrast}

Extract the correct tool orchestration pattern — which tool to call first, which next, and how to connect them.

Generate 1-2 tips. Each tip should describe a concrete tool-calling sequence like:
- "search_product with all keywords → click matching product_id → click size option → click Buy Now"
- "search for entity A → lookup detail → search for entity B → answer"
- "list_files → view_file the relevant file → edit_file to fix → run_tests to verify"

```yaml
tips:
  - description: "when to use this pattern"
    body: "the tool sequence and how to connect each step"
```

Rules:
- Use actual tool names from the list above
- Describe the ORDER: tool_A → tool_B → tool_C
- Be specific about what to pass between tools
- Do NOT use abstract placeholders like [ENTITY] or [VALUE]

Output ONLY the YAML block."""


CURATE_TIPS_PROMPT = """You are curating the tip library for {task_type} tasks.
A "tip" is a reusable tool-calling strategy that helps an agent solve problems.

## Current tips for {task_type}
{existing_tips}

## Available tools
{tool_list}

## Evidence from recent training

### Successful trajectories (high reward)
{success_evidence}

### Failed trajectories (low reward)
{failure_evidence}

### Critical decision points (backward policy analysis)
{critical_steps}

### Same-question trajectory comparisons
{dag_comparisons}

## Counterfactual Analysis Protocol (MANDATORY)

Before deciding actions, perform this analysis:

1. **Identify the divergence**: Compare the SAME step in success vs. failure trajectories.
   What tool/argument did success pick that failure missed?

2. **Extract the principle**: Generalize the divergence into a RULE that works for unseen instances.
   Example: "Success used `search_code` with exact function name; failure used vague concept phrases."

3. **Identify the fatal step in failure**: Which single decision, if flipped, would have saved the failure?
   This is the highest-leverage lesson.

4. **Check existing tip coverage**: Does the current tip already warn about this? If yes → UPDATE to strengthen;
   if no → NEW tip or UPDATE to add.

## Your task
Based on counterfactual analysis, decide what changes to make.

Principles:
- FEWER tips is better. Aim for 1-3 tips per task type maximum.
- A tip must describe a CONCRETE tool sequence using actual tool names.
- **Every UPDATE/ADD must cite the specific evidence divergence** (step index, action differences).
- DELETE tips that are wrong, misleading, or redundant with a better tip.
- UPDATE tips that are correct but could be improved with new evidence.
- KEEP tips that are still accurate and useful.
- Only set needs_new_tip to true if the evidence shows a genuinely new pattern not covered.

## Quality Bar for new_body
- Start with a ONE-LINE imperative summary ("Use X before Y; never Z.")
- Follow with a numbered workflow (max 5 steps)
- End with a "FATAL MISTAKES" list from failure evidence (with step-level citations)
- No hedging language ("try", "maybe") — use imperatives ("do", "never", "always")

```yaml
verdict:
  actions:
    - action: "KEEP"
      skill_id: "tip-xxx"
      reason: "still accurate"
    - action: "UPDATE"
      skill_id: "tip-yyy"
      new_body: "updated tool sequence (with divergence citation)"
      new_description: "updated trigger condition"
      reason: "counterfactual evidence: success used A at step N, failure used B → reward Δ=0.7"
    - action: "DELETE"
      skill_id: "tip-zzz"
      reason: "why remove"
  needs_new_tip: false
  new_tip_focus: ""
```

If there are no existing tips, just decide whether a new tip is needed based on the evidence.

Output ONLY the YAML block."""


GENERATE_SINGLE_TIP_PROMPT = """You are generating ONE tip for {task_type} tasks.

## Existing tips (DO NOT duplicate)
{existing_tips}

## What this new tip should capture
{new_tip_focus}

## Evidence (success + failure trajectories)
{evidence_summary}

## Available tools
{tool_list}

## Counterfactual Analysis (mandatory)
Before writing the tip, mentally answer:
1. At which step did success and failure DIVERGE?
2. What did success pick that failure missed (concrete tool + argument)?
3. What was the FATAL MISTAKE in failures (1-2 sentences)?

Use answers to write a tip that would have flipped the failure trajectories.

## Output format

```yaml
tip:
  description: "when to use this pattern (trigger condition, 1 line)"
  body: |
    (1 line imperative summary)
    WORKFLOW:
    1. tool_A with specific input → expected observation
    2. tool_B using result → expected observation
    3. tool_C to verify → if [X], STOP; else retry
    FATAL MISTAKES (from evidence):
    - Mistake 1 (what failures did wrong)
    - Mistake 2 (what failures did wrong)
```

Rules:
- Use ACTUAL tool names from the list above
- Describe the ORDER and data flow between tools
- Must be DIFFERENT from existing tips shown above
- 50-200 words for the body (concrete > verbose)
- Every fatal mistake must reference observable evidence (e.g., "burned 5 steps on X")
- No hedging: use "do", "never", "always"; avoid "try", "maybe", "consider"

Output ONLY the YAML block."""


DIAGNOSE_AND_CURATE_PROMPT = """You are curating the tip library for {task_type} tasks.

## Current tips for {task_type}
{existing_tips}

## Available tools
{tool_list}

## Flow Bottleneck Diagnosis

GFlowNet training identified these decision points where the agent is INCONSISTENT
(high variance in forward/backward policy agreement across episodes):

{bottleneck_diagnoses}

## Counterfactual Evidence

At each bottleneck, these are concrete examples where the SAME question led to
different outcomes based on the choice at the bottleneck step:

{counterfactual_evidence}

## Your task

For each diagnosed bottleneck, decide what edit to make:

1. Bottleneck with NO existing tip coverage -> ADD a new tip
2. Bottleneck covered by a tip with low success rate -> UPDATE the tip
3. Bottleneck covered by a tip that works well -> KEEP (agent needs more training)
4. A tip that is WRONG based on counterfactual evidence -> DELETE

Principles:
- FEWER tips is better. 1-3 tips per task type maximum.
- Each tip must describe a CONCRETE tool sequence with actual tool names.
- Use the counterfactual evidence: it shows exactly what works vs what fails.
- Reference specific bottleneck IDs and counterfactual examples in your reasoning.

```yaml
verdict:
  actions:
    - bottleneck_id: 0
      action: "ADD"
      reason: "bottleneck X shows failure at step Y, counterfactual evidence shows Z works"
      new_body: "tool_A with specific input → tool_B using result → tool_C to verify"
      new_description: "trigger condition: when to apply this tip"
    - bottleneck_id: 1
      action: "KEEP"
      skill_id: "tip-xxx"
      reason: "this tip already covers the bottleneck"
```

Rules:
- If there are NO existing tips and bottlenecks are detected, you MUST ADD at least one tip.
- For ADD: provide new_body (concrete tool sequence, 30-100 words) and new_description (trigger condition).
- For UPDATE: provide skill_id + new_body + new_description.
- For DELETE: provide skill_id + reason.
- For KEEP: provide skill_id + reason.

Output ONLY the YAML block."""


def parse_curation_verdict(text: str) -> dict:

    import re
    import yaml

    yaml_match = re.search(r'```(?:yaml)?\s*\n(.*?)```', text, re.DOTALL)
    yaml_text = yaml_match.group(1) if yaml_match else text

    default = {"actions": [], "needs_new_tip": True, "new_tip_focus": "generate a useful tool-calling strategy based on the evidence"}

    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict):
            verdict = data.get("verdict", data)
            actions = verdict.get("actions", [])

            has_add = any(
                a.get("action", "").upper() == "ADD" for a in actions if isinstance(a, dict)
            )
            needs_new = bool(verdict.get("needs_new_tip", not has_add))
            return {
                "actions": actions,
                "needs_new_tip": needs_new,
                "new_tip_focus": str(verdict.get("new_tip_focus", "") or ""),
            }
    except Exception:
        pass

    return default


def parse_single_tip(text: str):

    import re
    import yaml


    yaml_match = re.search(r'```(?:yaml)?\s*\n(.*?)```', text, re.DOTALL)
    yaml_text = yaml_match.group(1) if yaml_match else text


    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict):
            tip = data.get("tip", data)
            if isinstance(tip, dict):
                desc = str(tip.get("description", "")).strip()
                body = str(tip.get("body", "")).strip()
                if desc and body and len(body.split()) >= 8:
                    return (desc, body)
    except Exception:
        pass


    try:
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            tip = data.get("tip", data)
            if isinstance(tip, dict):
                desc = str(tip.get("description", "")).strip()
                body = str(tip.get("body", "")).strip()
                if desc and body and len(body.split()) >= 8:
                    return (desc, body)
    except Exception:
        pass


    desc_m = re.search(r'description:\s*["\']?(.+?)["\']?\s*\n', text)

    body_m = re.search(r'body:\s*\|\s*\n(.+?)(?=\n\w+:|$)', text, re.DOTALL)
    if not body_m:
        body_m = re.search(r'body:\s*["\']?(.+?)["\']?\s*(?:\n\w+:|$)', text, re.DOTALL)
    if desc_m and body_m:
        d = desc_m.group(1).strip().strip('"\'')
        b = body_m.group(1).strip().strip('"\'')
        if d and b and len(b.split()) >= 8:
            return (d, b)

    return None


def validate_tip(trigger: str, tip: str) -> bool:

    if not trigger or not tip:
        return False
    if len(tip.split()) < 5:
        return False

    generic_phrases = [
        "use tools", "read carefully", "think step by step",
        "be careful", "double check",
    ]
    tip_lower = tip.lower()
    if any(g in tip_lower for g in generic_phrases):
        return False
    return True


def parse_tips_yaml(text: str) -> list:

    import re
    import yaml

    yaml_match = re.search(r'```(?:yaml)?\s*\n(.*?)```', text, re.DOTALL)
    yaml_text = yaml_match.group(1) if yaml_match else text

    try:
        data = yaml.safe_load(yaml_text)
        if isinstance(data, dict) and 'tips' in data:
            tips = data['tips']
        elif isinstance(data, list):
            tips = data
        else:
            return []

        result = []
        for item in tips:
            if isinstance(item, dict):

                desc = str(item.get('description', item.get('trigger', '')))
                body = str(item.get('body', item.get('tip', '')))
                if validate_tip(desc, body):
                    result.append((desc, body))
        return result
    except Exception:

        triggers = re.findall(r'(?:trigger|description):\s*["\']?(.*?)["\']?\s*\n', text)
        tips = re.findall(r'(?:tip|body):\s*["\']?(.*?)["\']?\s*\n', text)
        result = []
        for t, p in zip(triggers, tips):
            if validate_tip(t, p):
                result.append((t, p))
        return result
