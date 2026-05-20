

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.executor.m_exec import MExec
from training.reward import compute_full_reward, EPSILON_MIN
from training.trajectory import Trajectory, Turn

logger = logging.getLogger(__name__)


_VALID_ACTION_TYPES = {

    "plan", "decompose",

    "python_execute", "test_code", "analyze",

    "search", "lookup", "fact_verify",

    "ask_llm", "self_consistency",

    "verify_answer", "check_answer", "cross_validate",

    "bash", "str_replace_editor", "verify_fix",

    "list_files", "search_code", "view_file", "edit_file", "run_tests",

    "act", "search_product", "click",

    "answer",

    "skill_invoke", "passage_search", "direct_act", "reflect", "think",
    "parse_error",  
}


_tool_call_counter = 0


def _next_tool_call_id() -> str:

    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_{_tool_call_counter}"


def _coerce_int(value, default: int = 0) -> int:

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(text)
    except Exception:
        m = re.search(r"-?\d+", text)
        return int(m.group(0)) if m else default


_SYSTEM_PROMPT = (
    "You are a task-solving agent that uses tools to solve problems.\n\n"

    "# System\n"
    "- You have access to tools defined in the tool list. Each tool has a name, description, and parameters.\n"
    "- Tool results may contain useful information. Read them carefully before deciding your next action.\n"
    "- You can call one tool per step. Choose the most appropriate tool based on the task.\n\n"

    "# Doing tasks\n"
    "- Use tools to gather evidence and solve problems. Do not answer from memory alone.\n"
    "- Go straight to the point. Try the simplest approach first.\n"
    "- If a tool returns an error, [NO_MATCH], [FAILED], or empty results, try a different tool or query.\n"
    "- Do not repeat the same tool call with the same arguments. If you tried something that didn't work, change your approach.\n"
    "- If you are stuck after 3 attempts, provide your best answer based on what you have gathered so far.\n\n"

    "# Using your tools\n"
    "- Each tool serves a specific purpose as described in its definition. Choose based on the task:\n"
    "  - Retrieval tools (search, lookup): find information in provided passages or context.\n"
    "  - Computation tools (python_execute, test_code): run code for calculations or verification.\n"
    "  - Code tools (list_files, view_file, search_code, edit_file, run_tests): navigate and modify code.\n"
    "  - Reasoning tools (plan, decompose, analyze, ask_llm): break down complex problems.\n"
    "  - Verification tools (verify_answer, check_answer, fact_verify): validate your answer.\n"
    "  - Environment tools (act, search_product, click): interact with external environments.\n"
    "- If an 'Available Learned Skills' section appears, call skill_invoke with a listed skill_id before using that skill. "
    "If a legacy 'Learned Strategy' section appears, follow it directly with the regular tools.\n\n"

    "# Answer format\n"
    "- When you have the answer, wrap it in <answer> tags: <answer>YOUR_ANSWER</answer>\n"
    "- Be concise — just the answer, no explanation.\n"
    "- Examples: <answer>Paris</answer>  <answer>42</answer>  <answer>B</answer>\n"
)


class GenericTaskEnvironment:


    def __init__(
        self,
        m_exec: MExec,
        max_episode_steps: int = 8,
        epsilon_min: float = EPSILON_MIN,
        skill_workspace=None,    
        experience_store=None,   
        context_manager=None,    
        max_obs_chars: int = 0,          
        max_context_chars: int = 30000,   
        reward_mode: str = "outcome_only",
        skill_mode: str = "policy_action",  
    ):
        self.m_exec = m_exec
        self.max_episode_steps = max_episode_steps
        self.epsilon_min = epsilon_min
        self.workspace = skill_workspace
        self._experience_store = experience_store
        self.context_manager = context_manager
        self.max_obs_chars = max_obs_chars
        self.max_context_chars = max_context_chars
        self.reward_mode = reward_mode
        self.skill_mode = skill_mode


        self._recent_tool_calls: Dict[str, str] = {}  


        self._code_workspace: Dict[str, str] = {}

        self._repo_path: Optional[str] = None
        self._cached_diff: Optional[str] = None  
        self._edit_history: Dict[str, str] = {}  
        self._repeat_count: Dict[str, int] = {}  
        self._python_execute_count: int = 0  
        self._consecutive_repeats: int = 0  


        self._ragen_adapter = None
        try:
            from src.ragen_adapter import RAGENAdapter
            self._ragen_adapter = RAGENAdapter()
        except ImportError:
            pass

    def reset(
        self,
        question: Dict[str, Any],
        episode_goal: str = "",
        retrieved_experience: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], "Trajectory"]:

        self._question = question
        self._gold = str(question.get("answer", ""))
        self._task_type = str(question.get("task_type", "factual_qa"))
        self._extra = question.get("extra", {})

        if question.get("code_files") and "code_files" not in self._extra:
            self._extra["code_files"] = question["code_files"]

        raw_ctx = question.get("context", [])
        if isinstance(raw_ctx, str):
            try:
                import json as _json
                raw_ctx = _json.loads(raw_ctx)
            except (ValueError, TypeError):
                raw_ctx = [raw_ctx] if raw_ctx.strip() else []
        self._context = raw_ctx if isinstance(raw_ctx, list) else []
        self._history: List[Dict] = []

        self._retrieved_passage_ids: set = set()
        self._step = 0
        self._episode_goal = episode_goal
        self._retrieved_experience = retrieved_experience or []


        self._code_workspace = {}
        self._code_workspace_original = {}
        self._repo_path = None
        self._cached_diff = None
        if self._task_type == "code_generation" and self._extra.get("instance_id"):
            self._repo_path = self._setup_swe_repo()
        if not self._repo_path:

            code_files = question.get("code_files", {})
            if isinstance(code_files, dict):
                self._code_workspace = dict(code_files)
                self._code_workspace_original = dict(code_files)


        self._ragen_initial_obs = ""
        self._env_done = False
        self._env_reward = 0.0
        if self._ragen_adapter and self._task_type in ("webshop", "alfworld", "interactive_agent"):
            env_type = question.get("env_type", "")
            env_config = question.get("env_config", {})
            try:
                self._ragen_initial_obs = self._ragen_adapter.reset(
                    env_type, env_config,
                    question=str(question.get("question", "")),
                    extra=question.get("extra", {}),
                )
            except Exception as e:
                logger.warning(f"[RAGEN] Failed to reset env: {e}")


        self._messages: List[Dict] = []


        from training.task_prompts import TASK_CONFIGS
        _task_cfg = TASK_CONFIGS.get(self._task_type, {})
        _sys_prompt = _task_cfg.get("system_prompt", _SYSTEM_PROMPT)


        if self.skill_mode == "policy_action":
            self._injected_skill_ids = []
            _skill_addendum = self._skill_catalog_for_policy_action(question)
        else:
            _skill_addendum = self._inject_skill_tip_to_system(question)
        if _skill_addendum:
            _sys_prompt = _sys_prompt + _skill_addendum

        self._messages.append({"role": "system", "content": _sys_prompt})


        user_content = self._build_user_message(question)
        self._messages.append({"role": "user", "content": user_content})


        from training.batch_inference import SUPERVISOR_TOOLS
        self._tools = self._filter_tools_for_task(self._task_type, SUPERVISOR_TOOLS)


        traj = Trajectory(
            question=str(question.get("question", "")),
            gold_answer=self._gold,
            task_type=self._task_type,
        )


        for sid in getattr(self, '_injected_skill_ids', []):
            traj.add_turn(Turn(
                supervisor_input="",
                supervisor_output=f"skill_invoke({sid})",
                action_type="skill_invoke",
                skill_id=sid,
                instruction="auto_inject",
                observation=f"Skill {sid} injected as learned tip.",
            ))

        return self._messages, traj

    def _filter_tools_for_task(self, task_type: str, all_tools: list) -> list:

        from training.task_prompts import TASK_CONFIGS
        cfg = TASK_CONFIGS.get(task_type, {})
        allowed = set(cfg.get("tools", []))
        if self.skill_mode == "policy_action" and self.workspace and self.workspace.size > 0:
            allowed.add("skill_invoke")
        if not allowed:

            return all_tools
        return [t for t in all_tools if t["function"]["name"] in allowed]

    @staticmethod
    def _canonical_tool_call_text(tool_name: str, tool_args: Optional[Dict]) -> str:

        return "<tool_call>\n" + json.dumps(
            {"name": tool_name, "arguments": tool_args or {}},
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n</tool_call>"

    def step(
        self,
        content: str,
        tool_name: Optional[str],
        tool_args: Optional[Dict],
        traj: Trajectory,
    ) -> Tuple[float, bool, Dict[str, Any]]:

        self._step += 1


        import copy
        _messages_snapshot = copy.deepcopy(self._messages)

        _supervisor_input_text = "\n".join(
            f"[{m['role']}] {str(m.get('content', ''))}"
            for m in self._messages[-6:]
        )


        if tool_name is None:
            allowed_tool_names = {
                t.get("function", {}).get("name")
                for t in getattr(self, "_tools", []) or []
                if t.get("function", {}).get("name")
            }
            if self._task_type == "code_generation" and "answer" not in allowed_tool_names:


                workspace_diff = self._generate_workspace_diff()
                if workspace_diff.strip():
                    logger.info(
                        "[Env] code_generation: no tool emitted after source edit; "
                        "submitting existing workspace diff"
                    )
                    return self._force_terminate(traj)
            if self._task_type == "code_generation" and "answer" not in allowed_tool_names:
                no_tool_obs = (
                    "[PARSE_ERROR] No tool call detected. For code_generation, the final "
                    "submission is the repository diff and the 'answer' tool is not available. "
                    "If the source fix is complete, avoid extra edits; otherwise use one of "
                    f"the available tools: {', '.join(sorted(allowed_tool_names))}."
                )
            else:
                no_tool_obs = (
                    "[PARSE_ERROR] No tool call detected. You must call a tool. "
                    "To submit your final answer, use the 'answer' tool with response=<your answer>."
                )

            parse_err_turn = Turn(
                supervisor_input=_supervisor_input_text,
                supervisor_output=content if content else "",
                action_type="parse_error",
                answer="",
                parse_error=True,
                messages_snapshot=_messages_snapshot,
                observation=no_tool_obs,
                instruction="",
            )
            traj.add_turn(parse_err_turn)

            self._messages.append({"role": "assistant", "content": content if content else ""})
            self._messages.append({
                "role": "user",
                "content": parse_err_turn.observation,
            })
            if self._step >= self.max_episode_steps:
                return self._force_terminate(traj)
            return 0.0, False, {"observation": parse_err_turn.observation, "parse_error": True}

        _supervisor_output_text = content or self._canonical_tool_call_text(tool_name, tool_args or {})


        if tool_name == "answer":
            allowed_tool_names = {
                t.get("function", {}).get("name")
                for t in getattr(self, "_tools", []) or []
                if t.get("function", {}).get("name")
            }
            if allowed_tool_names and "answer" not in allowed_tool_names:
                reject_obs = (
                    f"[TOOL_UNAVAILABLE] Tool 'answer' is not available for "
                    f"task_type={self._task_type}. Available tools: "
                    f"{', '.join(sorted(allowed_tool_names))}. Choose one of the available tools."
                )
                parse_err_turn = Turn(
                    supervisor_input=_supervisor_input_text,
                    supervisor_output=_supervisor_output_text,
                    action_type="parse_error",
                    answer="",
                    parse_error=True,
                    messages_snapshot=_messages_snapshot,
                    observation=reject_obs,
                    instruction="",
                )
                traj.add_turn(parse_err_turn)
                self._messages.append({"role": "assistant", "content": content if content else ""})
                self._messages.append({"role": "user", "content": reject_obs})
                if self._step >= self.max_episode_steps:
                    return self._force_terminate(traj)
                return 0.0, False, {"observation": reject_obs, "parse_error": True}


            real_tool_turns = [
                t for t in traj.turns
                if getattr(t, 'action_type', '') not in ('answer', 'skill_invoke', '', 'parse_error')
            ]
            if not real_tool_turns:

                reject_obs = (
                    "[INVALID] The 'answer' tool is not available yet. "
                    "You MUST call another tool (e.g., search, python_execute, view_file) "
                    "to gather evidence BEFORE submitting an answer."
                )
                parse_err_turn = Turn(
                    supervisor_input=_supervisor_input_text,
                    supervisor_output=_supervisor_output_text,
                    action_type="parse_error",
                    answer="",
                    parse_error=True,
                    messages_snapshot=_messages_snapshot,
                    observation=reject_obs,
                    instruction="",
                )
                traj.add_turn(parse_err_turn)

                self._messages.append({"role": "assistant", "content": content if content else ""})
                self._messages.append({"role": "user", "content": reject_obs})
                if self._step >= self.max_episode_steps:
                    return self._force_terminate(traj)
                return 0.0, False, {"observation": reject_obs, "parse_error": True}

            answer = ""
            if tool_args:
                answer = str(tool_args.get("response", "") or "").strip()


            workspace_diff = ""
            if self._task_type == "code_generation":
                workspace_diff = self._generate_workspace_diff()
                if not workspace_diff.strip():
                    reject_obs = (
                        "[INVALID] code_generation is evaluated from the repository diff, "
                        "but no source-code edit has been made yet, so the current submitted "
                        "diff would be empty. "
                        "Do not submit a prose answer."
                    )
                    parse_err_turn = Turn(
                        supervisor_input=_supervisor_input_text,
                        supervisor_output=_supervisor_output_text,
                        action_type="parse_error",
                        answer="",
                        parse_error=True,
                        messages_snapshot=_messages_snapshot,
                        observation=reject_obs,
                        instruction="",
                    )
                    traj.add_turn(parse_err_turn)
                    self._messages.append({"role": "assistant", "content": content if content else ""})
                    self._messages.append({"role": "user", "content": reject_obs})
                    if self._step >= self.max_episode_steps:
                        return self._force_terminate(traj)
                    return 0.0, False, {"observation": reject_obs, "parse_error": True}


            turn = Turn(
                supervisor_input=_supervisor_input_text,
                supervisor_output=_supervisor_output_text,
                action_type="answer",
                answer=answer,
                parse_error=False,
                messages_snapshot=_messages_snapshot,
            )


            eval_pred = answer
            if self._task_type == "code_generation":
                if workspace_diff:
                    eval_pred = workspace_diff

            if self._task_type in ("webshop", "alfworld", "interactive_agent") and self._env_done and self._env_reward > 0:

                r_answer = min(float(self._env_reward), 1.0)
                if str(self.reward_mode).lower() in {"outcome_only", "paper", "outcome"}:
                    r_process = 0.0
                    r_total = r_answer
                else:
                    r_process = 0.1
                    r_total = r_answer + r_process
                r_tilde = max(r_total + self.epsilon_min, self.epsilon_min)
                r_skill = 0.0
            else:
                r_total, r_answer, r_process, r_skill, r_tilde = compute_full_reward(
                    pred=eval_pred,
                    gold=self._gold,
                    task_type=self._task_type,
                    turns=traj.turns + [turn],
                    extra=self._extra,
                    epsilon_min=self.epsilon_min,
                    experience_store=self._experience_store,
                    reward_mode=self.reward_mode,
                )

            final_answer = eval_pred if self._task_type == "code_generation" and workspace_diff else answer
            turn.answer = final_answer
            traj.add_turn(turn)
            traj.final_answer = final_answer
            traj.reward = r_total
            traj.answer_reward = r_answer
            traj.skill_reward = r_skill
            traj.r_tilde = r_tilde
            traj.completed = True

            logger.debug(
                f"[Env] Episode done (direct answer) | task={self._task_type} | "
                f"R={r_total:.3f} (ans={r_answer:.3f}) | steps={self._step}"
            )
            return r_total, True, {
                "final_answer": final_answer,
                "r_answer": r_answer,
                "r_process": r_process,
                "r_skill": r_skill,
            }


        args = tool_args or {}


        if self._task_type == "code_generation":
            terminal_names = {"submit", "finish", "final", "done", "accept"}
            if str(tool_name or "").strip().lower() in terminal_names:
                workspace_diff = self._generate_workspace_diff()
                if workspace_diff.strip():
                    logger.info(
                        "[Env] code_generation: terminal tool %r emitted after source edit; "
                        "submitting existing workspace diff",
                        tool_name,
                    )
                    return self._force_terminate(traj)


        if tool_name not in _VALID_ACTION_TYPES:

            if re.match(r"^dyn_\d+$", tool_name) or re.match(r"^[a-z_]+_\d{3}$", tool_name):
                args["skill_id"] = tool_name
                tool_name = "skill_invoke"
            else:
                logger.warning(f"[Env] Unknown tool: {tool_name}, treating as error")
                tool_name = "analyze"
                args = {"instruction": content[:500]}


        allowed_tool_names = {
            t.get("function", {}).get("name")
            for t in getattr(self, "_tools", []) or []
            if t.get("function", {}).get("name")
        }
        if allowed_tool_names and tool_name not in allowed_tool_names:
            allowed_s = ", ".join(sorted(allowed_tool_names))
            reject_obs = (
                f"[TOOL_UNAVAILABLE] Tool '{tool_name}' is not available for "
                f"task_type={self._task_type}. Available tools: {allowed_s}. "
                "Choose one of the available tools."
            )
            if self._task_type == "code_generation":
                memory = self._swe_memory_summary(
                    traj=traj,
                    current_action="parse_error",
                    current_args={},
                    current_instruction=str(tool_name or ""),
                    current_observation=reject_obs,
                )
                if memory:
                    reject_obs = self._cap_swe_observation(
                        reject_obs.rstrip() + "\n\n" + memory
                    )
            parse_err_turn = Turn(
                supervisor_input=_supervisor_input_text,
                supervisor_output=_supervisor_output_text,
                action_type="parse_error",
                answer="",
                parse_error=True,
                messages_snapshot=_messages_snapshot,
                observation=reject_obs,
                instruction="",
            )
            traj.add_turn(parse_err_turn)
            self._messages.append({"role": "assistant", "content": content if content else ""})
            self._messages.append({"role": "user", "content": reject_obs})
            if self._step >= self.max_episode_steps:
                return self._force_terminate(traj)
            return 0.0, False, {"observation": reject_obs, "parse_error": True}


        instruction = (
            args.get("instruction") or args.get("query") or args.get("claim")
            or args.get("thought") or args.get("goal") or args.get("problem")
            or args.get("question") or args.get("action") or args.get("keyword")
            or args.get("path") or args.get("test_cmd") or args.get("element")
            or args.get("answer") or ""
        )[:200]

        turn = Turn(
            supervisor_input=_supervisor_input_text,
            supervisor_output=_supervisor_output_text,
            action_type=tool_name,
            skill_id=args.get("skill_id"),
            instruction=instruction,
            parse_error=False,
            messages_snapshot=_messages_snapshot,
            raw_action={
                "action_type": tool_name,
                "skill_id": args.get("skill_id"),
                "instruction": instruction,
                "tool_args": args,
            },
        )


        import hashlib
        _call_key = hashlib.md5(f"{tool_name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()
        if _call_key in self._recent_tool_calls and tool_name not in ("edit_file", "run_tests", "str_replace_editor", "verify_fix"):
            prev = self._recent_tool_calls[_call_key]


            repeat_i = self._repeat_count.get(_call_key, 0) + 1
            self._repeat_count[_call_key] = repeat_i
            self._repeat_count[tool_name] = repeat_i  
            self._consecutive_repeats += 1
            _dedup_hints = {
                "bash": "The cached command result is below; this exact call adds no new evidence.",
                "search_code": "The cached search result is below; this exact search adds no new evidence.",
                "view_file": "The same source window is below; this exact view adds no new evidence.",
                "list_files": "The same file listing is below; this exact listing adds no new evidence.",
                "python_execute": "The cached execution result is below; this exact execution adds no new evidence.",
            }
            hint = _dedup_hints.get(tool_name, "This exact repeated call will not add new information.")
            evidence_note = ""
            if tool_name == "search_code":


                if "[NO_MATCH]" in str(prev):
                    evidence_note = (
                        "\nCached search status: NO_MATCH. Any relaxed locations in the cached "
                        "result are broad source hints, not exact matches for this query."
                    )
                else:
                    cached_hits: List[Tuple[str, int]] = []
                    seen_cached_hits = set()


                    for loc in re.finditer(
                        r"(?m)(?:^|\n)(?:\./)?([^:\n]+\.py)[:\-](\d+)(?=[:\-])",
                        str(prev),
                    ):
                        sug_path = loc.group(1).lstrip("./")
                        sug_line = _coerce_int(loc.group(2), 1)
                        key_hit = (sug_path, sug_line)
                        if key_hit in seen_cached_hits:
                            continue
                        cached_hits.append(key_hit)
                        seen_cached_hits.add(key_hit)
                        if len(cached_hits) >= 12:
                            break
                    if cached_hits:
                        sug_path, sug_line = cached_hits[0]
                        if repeat_i <= 3:
                            evidence_note = (
                                "\nFirst cached Python hit: "
                                f"{sug_path}:{sug_line} "
                                f"(near lines {max(1, sug_line - 20)}-{sug_line + 40})."
                            )
                            if len(cached_hits) > 1:
                                evidence_note += (
                                    "\nCached Python hit list from the same result: "
                                    + ", ".join(f"{p}:{ln}" for p, ln in cached_hits[:8])
                                )
                        else:
                            evidence_note = (
                                "\nCached search status: OK with previously returned Python hit(s). "
                                "Repeated hit/source replay is suppressed after multiple exact repeats; "
                                "SWE_MEMORY retains the earlier source evidence."
                            )
                        expanded = ""
                        context_hits: List[Tuple[str, int]] = []
                        if repeat_i <= 1:
                            context_hits = [cached_hits[0]]
                        elif repeat_i <= 3:


                            start_i = min(max(repeat_i - 1, 0), len(cached_hits) - 1)
                            context_hits = cached_hits[start_i:start_i + 2]
                        if context_hits:
                            pieces = []
                            for ctx_path, ctx_line in context_hits:
                                ctx = self._source_context_for_location(
                                    ctx_path, ctx_line, before=6, after=12
                                )
                                if ctx:
                                    pieces.append(ctx)
                            expanded = "\n\n".join(pieces)
                        if expanded:
                            evidence_note += (
                                "\nExpanded source context from cached hit location(s):\n"
                                f"{expanded}"
                            )
            elif tool_name == "view_file":
                evidence_note = (
                    "\nThis is the same source evidence as the previous view_file call."
                )
            prev_block = ""
            if repeat_i <= 1:
                prev_block = f"\nCached result excerpt:\n{prev[:700]}"
            else:


                prev_clean = self._strip_swe_memory(prev)
                if "[NO_MATCH]" in prev_clean:
                    compact_prev = "previous cached observation was NO_MATCH"
                elif "[ERROR]" in prev_clean:
                    compact_prev = "previous cached observation was ERROR"
                elif "[OK]" in prev_clean:
                    compact_prev = "previous cached observation was OK"
                else:
                    compact_prev = self._shorten_one_line(prev_clean, 120)
                prev_block = (
                    f"\nCached result summary: {compact_prev}; repeated source content "
                    "is not replayed again here. Earlier tool output and SWE_MEMORY "
                    "retain the evidence already returned."
                )
            repeat_state_tag = f"[REPEATED x{repeat_i}]"
            loop_no_evidence = (
                self._task_type == "code_generation"
                and tool_name == "search_code"
                and repeat_i >= 4
                and "[NO_MATCH]" in str(prev)
                and not self._workspace_diff_is_nonempty()
            )
            if loop_no_evidence:
                repeat_state_tag += " [LOOP_NO_NEW_EVIDENCE]"
            observation = (
                f"[{tool_name}] {repeat_state_tag} "
                f"You already called this exact tool+args; it will not return new information. "
                f"{hint if repeat_i <= 1 else 'The earlier observation remains in SWE_MEMORY; repeated source content is summarized instead of replayed.'}"
                f"{evidence_note}"
                f"{prev_block}"
            )
            if self._task_type == "code_generation":
                remaining_budget = max(
                    0,
                    int(getattr(self, "max_episode_steps", 0) or 0) - self._step,
                )
                diff_state = (
                    "non-empty current workspace diff"
                    if self._workspace_diff_is_nonempty()
                    else "empty current workspace diff"
                )
                observation += (
                    "\nRepeated-action state: this exact tool call returned cached evidence; "
                    "no repository source or workspace diff changed during this repeated call. "
                    f"Current state: {diff_state}; tool-call budget used "
                    f"{self._step}/{getattr(self, 'max_episode_steps', '?')} "
                    f"(remaining {remaining_budget})."
                )
                if tool_name == "search_code" and "[NO_MATCH]" in str(prev):
                    observation += (
                        "\nRepeated NO_MATCH state: this exact query has already "
                        "returned no exact source hit; later repeats are cached "
                        "state, not new search evidence."
                    )
                    if loop_no_evidence:
                        observation += (
                            "\nLoop state: this repeated exact NO_MATCH search has consumed "
                            "multiple source-tool calls while the workspace diff is still empty; "
                            "the current observation adds no new repository evidence beyond SWE_MEMORY."
                        )
                if tool_name == "search_code":
                    member_mentions = self._swe_issue_member_mentions()
                    if member_mentions:
                        q_l = str(args.get("query") or instruction or "").lower()
                        not_in_query = [
                            f"{cls}.{member}"
                            for cls, member in member_mentions
                            if member.lower() not in q_l
                            and f"{cls}.{member}".lower() not in q_l
                        ]
                        if not_in_query:
                            observation += (
                                "\nRepeated-query/member alignment state: "
                                "issue-visible member(s) "
                                + ", ".join(not_in_query[:4])
                                + " are not named in this repeated query."
                            )
        else:


            observation = self._dispatch_tool(tool_name, args, traj)
            self._recent_tool_calls[_call_key] = observation
            if tool_name == "edit_file" and (
                "[OK]" in str(observation)
                or "[WARN]" in str(observation)
                or "[NO_CHANGE_NET]" in str(observation)
            ):


                self._recent_tool_calls.clear()
            self._repeat_count[_call_key] = 0
            self._repeat_count[tool_name] = 0
            self._consecutive_repeats = 0  

        if self.max_obs_chars > 0:
            observation = observation[:self.max_obs_chars]


        if self._task_type == "code_generation":
            memory = self._swe_memory_summary(
                traj=traj,
                current_action=tool_name,
                current_args=args,
                current_instruction=instruction,
                current_observation=observation,
            )
            if memory:
                observation += "\n\n" + memory
            observation = self._cap_swe_observation(observation)
        turn.observation = observation


        self._history.append({
            "step": self._step,
            "action_type": tool_name,
            "skill_id": args.get("skill_id"),
            "instruction": instruction,
            "observation": observation,
        })

        traj.add_turn(turn)


        tc_id = _next_tool_call_id()
        self._messages.append({
            "role": "assistant",
            "content": content if content else None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })

        self._messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": observation,
        })


        if tool_name == "verify_fix" and "[PASS]" in observation:
            logger.info(f"[Env] verify_fix PASS → early termination at step {self._step}")
            return self._force_terminate(traj)


        if self._step >= self.max_episode_steps:
            return self._force_terminate(traj)

        return 0.0, False, {"observation": observation}


    def _inject_skill_tip_to_system(self, question: Dict[str, Any]) -> str:

        self._injected_skill_ids = []
        _ws = self.workspace
        _ws_size = _ws.size if _ws else 0
        task_type = self._task_type
        q_text = str(question.get("question", ""))

        if _ws and _ws_size > 0:
            candidates = _ws.retrieve(q_text, task_type=task_type, top_k=1)
            if candidates:
                tip = candidates[0]
                tip_types = getattr(tip.meta, 'task_types', []) if tip.meta else []
                if task_type in tip_types:
                    tip_text = tip.plan or ""
                    if tip_text:
                        self._injected_skill_ids.append(tip.meta.skill_id)
                        logger.info(
                            f"[Skill] Injected tip for {task_type}: "
                            f"[{tip.meta.skill_id}] \"{tip_text[:60]}\" ({len(tip_text.split())}w)"
                        )

                        return (
                            "\n\n# Learned Strategy (follow this priority)\n"
                            f"{tip_text.strip()}\n"
                        )
                else:
                    logger.info(
                        f"[Skill] Type mismatch: task={task_type} tip_types={tip_types} "
                        f"sid={tip.meta.skill_id}"
                    )
            else:
                logger.info(f"[Skill] No candidates for task_type={task_type}, ws_size={_ws_size}")
        elif _ws_size == 0 and _ws is not None:
            if not hasattr(self, '_ws_empty_warned'):
                self._ws_empty_warned = True
                logger.warning(f"[Skill] Workspace exists but empty (size=0)")
        return ""

    def _skill_catalog_for_policy_action(self, question: Dict[str, Any]) -> str:

        self._injected_skill_ids = []
        _ws = self.workspace
        if not _ws or _ws.size <= 0:
            return ""

        task_type = self._task_type
        q_text = str(question.get("question", ""))
        try:
            candidates = _ws.retrieve(q_text, task_type=task_type, top_k=4)
        except Exception as e:
            logger.warning(f"[Skill] Could not build policy-action catalog: {e}")
            return ""

        lines = []
        for tip in candidates:
            sid = getattr(tip.meta, "skill_id", "")
            tip_types = getattr(tip.meta, "task_types", []) if tip.meta else []
            if not sid or (tip_types and task_type not in tip_types):
                continue
            name = (getattr(tip, "name", "") or sid).strip()
            lines.append(f"- {sid}: {name}")

        if not lines:
            return ""

        return (
            "\n\n# Available Learned Skills\n"
            "You may call skill_invoke(skill_id=<id>) as an action when a listed "
            "skill is relevant. The tool will return the full strategy; do not "
            "assume the strategy before invoking it.\n"
            + "\n".join(lines)
            + "\n"
        )

    def _react_skill_catalog_for_policy_action(self, task_description: str) -> str:

        _ws = self.workspace
        if not _ws or _ws.size <= 0:
            return ""
        try:
            candidates = _ws.retrieve(task_description, task_type=self._task_type, top_k=4)
        except Exception as e:
            logger.warning(f"[Skill] Could not build ReAct skill catalog: {e}")
            return ""
        lines = []
        for tip in candidates:
            sid = getattr(tip.meta, "skill_id", "")
            tip_types = getattr(tip.meta, "task_types", []) if tip.meta else []
            if not sid or (tip_types and self._task_type not in tip_types):
                continue
            name = (getattr(tip, "name", "") or sid).strip()
            lines.append(f"- {sid}: {name}")
        if not lines:
            return ""
        return (
            "\n# Available Learned Skills\n"
            "You may choose the ReAct action skill_invoke[<id>] when a listed "
            "skill is relevant. The next observation will contain the strategy; "
            "then continue with the environment actions.\n"
            + "\n".join(lines)
            + "\n"
        )

    def _build_user_message(self, question: Dict[str, Any]) -> str:

        q_text = str(question.get("question", ""))
        task_type = self._task_type


        if task_type in ("multi_hop_qa", "factual_qa") and "\nQuestion:" in q_text:
            q_text = q_text.split("\nQuestion:")[-1].strip()
            q_text = q_text.strip("? \n") + "?"
        task_section = f"## Task\nType: {task_type}\n\nQuestion: {q_text}"


        context_section = ""
        if self._context and self._task_type not in ("multi_hop_qa", "factual_qa"):
            ctx_text = self._format_context()
            if ctx_text:
                context_section = f"\n\n## Context\n{ctx_text}"
        elif self._context:
            context_section = (
                f"\n\n## Context\n"
                f"{len(self._context)} passages are available. "
                f"Use search to find relevant information. "
                f"You must search for evidence before answering."
            )


        if self._ragen_initial_obs and "ENV_UNAVAILABLE" not in self._ragen_initial_obs:
            env_type = question.get("env_type", "")
            if env_type == "alfworld":
                context_section += (
                    "\n\n## Environment\n"
                    "You are in a household environment. ALWAYS choose actions from the Admissible actions list.\n\n"
                    "Example workflow for 'put spraybottle on toilet':\n"
                    "  1. act: go to cabinet 1 → see cloth, soapbar\n"
                    "  2. act: go to cabinet 2 → cabinet is closed\n"
                    "  3. act: open cabinet 2 → see candle, spraybottle 2\n"
                    "  4. act: take spraybottle 2 from cabinet 2 → picked up\n"
                    "  5. act: go to toilet 1 → arrived\n"
                    "  6. act: move spraybottle 2 to toilet 1 → done!\n\n"
                    f"Current observation:\n{self._ragen_initial_obs}"
                )
            elif env_type == "webshop":
                context_section += (
                    "\n\n## Environment\n"
                    "You are in a WebShop. Follow this exact workflow:\n\n"
                    "Example for 'buy 3oz citrus deodorant under $50':\n"
                    "  1. search_product: 3 ounce bright citrus deodorant\n"
                    "  2. click: B078GWRC1J (pick best matching product from results)\n"
                    "  3. click: bright citrus (select scent option)\n"
                    "  4. click: 3 ounce (pack of 1) (select size option)\n"
                    "  5. click: Buy Now → done!\n\n"
                    "Rules: Search once, click a product, select options, click Buy Now.\n"
                    "Pick the FIRST reasonable match. Don't search more than twice.\n\n"
                    f"Current observation:\n{self._ragen_initial_obs}"
                )
            else:
                context_section += f"\n\n## Environment Observation\n{self._ragen_initial_obs}"


        if self._task_type == "code_generation" and self._repo_path:
            context_section += (
                f"\n\n## Repository\n"
                f"You have access to the full repository at {self._repo_path}.\n"
                f"Use search_code to find relevant code, view_file to read files, "
                f"and edit_file to make changes.\n"
                f"Start by searching for key function/class names from the bug description."
            )
        elif self._task_type == "code_generation" and self._code_workspace:
            files_info = []
            for fp, content in sorted(self._code_workspace.items()):
                n_lines = content.count('\n') + 1
                files_info.append(f"  - {fp} ({n_lines} lines)")
            context_section += (
                f"\n\n## Code Workspace\n"
                f"Files available:\n" + "\n".join(files_info) + "\n"
            )


        user_content = (
            task_section
            + context_section
        )


        if self.max_context_chars > 0 and len(user_content) > self.max_context_chars:
            user_content = user_content[:self.max_context_chars] + "\n[CONTEXT TRUNCATED]"

        return user_content


    def _dispatch_tool(self, tool_name: str, args: Dict, traj: "Trajectory") -> str:


        if tool_name == "think":
            return self._handle_think(args)
        if tool_name == "plan":
            return self._handle_plan(args)
        if tool_name == "decompose":
            return self._handle_decompose(args, traj)


        if tool_name == "python_execute":
            return self._handle_python_execute(args, traj)
        if tool_name == "test_code":
            return self._handle_test_code(args, traj)
        if tool_name == "analyze":
            return self._handle_analyze(args)


        if tool_name in ("search", "passage_search"):
            return self._handle_search(args)
        if tool_name == "lookup":
            return self._handle_lookup(args)
        if tool_name == "fact_verify":
            return self._verify_fact(args.get("claim", ""))


        if tool_name == "ask_llm":
            return self._handle_ask_llm(args)
        if tool_name == "self_consistency":
            return self._handle_self_consistency(args)


        if tool_name == "verify_answer":
            return self._handle_verify_answer(args)
        if tool_name == "check_answer":
            return self._handle_check_answer(args)
        if tool_name == "cross_validate":
            return self._handle_cross_validate(args)


        if tool_name == "verify_fix":
            return self._handle_verify_fix(args)


        if tool_name == "bash":
            return self._handle_bash(args)
        if tool_name == "str_replace_editor":
            return self._handle_str_replace_editor(args)


        if tool_name == "list_files":
            return self._handle_list_files()
        if tool_name == "search_code":
            return self._handle_search_code(args)
        if tool_name == "view_file":
            return self._handle_view_file(args)
        if tool_name == "edit_file":
            return self._handle_edit_file(args, traj)
        if tool_name == "run_tests":
            return self._handle_run_tests(args)


        if tool_name == "act":
            return self._handle_act(args)
        if tool_name == "search_product":
            return self._handle_search_product(args)
        if tool_name == "click":
            return self._handle_click(args)


        if tool_name == "skill_invoke":
            return self._handle_skill_invoke(args)
        if tool_name == "reflect":
            return self._handle_think({"thought": args.get("instruction", "")})
        if tool_name == "direct_act":
            return self._handle_analyze({"instruction": args.get("instruction", "")})

        return f"[ERROR] Unknown tool: {tool_name}"


    def _handle_think(self, args: Dict) -> str:

        thought = args.get("thought", args.get("instruction", ""))
        return f"[Thought] {thought}"

    def _handle_plan(self, args: Dict) -> str:

        goal = args.get("goal", "").strip()
        if not goal:
            return "[plan] [ERROR] No goal provided. Pass a concrete objective as 'goal' arg."
        plan_result = self.m_exec.execute(
            instruction=f"Create a step-by-step plan to achieve this goal:\n\n{goal}",
            context=self._format_context(),
            task_type=self._task_type,
        )
        return f"[Plan]\n{plan_result}"

    def _handle_decompose(self, args: Dict, traj: "Trajectory") -> str:

        problem = args.get("problem", args.get("instruction", "")).strip()
        if not problem:
            return "[decompose] [ERROR] No problem provided. Pass the problem text as 'problem' arg."
        if self._task_type == "multi_hop_qa":
            decomp_prompt = (
                "Answer the following question by reasoning step-by-step.\n"
                "Decompose it into 2-3 sub-questions. For each sub-question, "
                "state what you need to find.\n"
                "Format:\n"
                "Q1: [first thing to find]\n"
                "Q2: [second thing to find, may depend on Q1's answer]\n"
                "Q3: Now we can answer the original question: [original question]\n\n"
                f"Question: {problem}"
            )
        elif self._task_type == "math_reasoning":
            decomp_prompt = (
                "Break this math problem into smaller, solvable steps.\n"
                "For each step, describe what to compute.\n"
                "Format:\n"
                "Q1: [first computation needed]\n"
                "Q2: [next computation, using result from Q1]\n"
                "Q3: [combine results to get final answer]\n\n"
                f"Problem: {problem}"
            )
        else:
            decomp_prompt = (
                "Decompose this problem into 2-3 concrete sub-questions.\n"
                "Format: Q1: ...\nQ2: ...\nQ3: ...\n\n"
                f"Problem: {problem}"
            )
        return self.m_exec.execute(
            instruction=decomp_prompt,
            context=self._format_context(),
            task_type=self._task_type,
        )


    _python_execute_count: int = 0

    def _handle_python_execute(self, args: Dict, traj: "Trajectory") -> str:

        instruction = args.get("instruction", "")


        import hashlib
        instr_normalized = " ".join(instruction.split())  
        instr_hash = hashlib.md5(instr_normalized.encode()).hexdigest()[:16]
        cache_key = f"python_execute::{instr_hash}"
        if hasattr(self, '_recent_tool_calls') and cache_key in self._recent_tool_calls:
            prev = self._recent_tool_calls[cache_key]
            return (
                f"[python_execute] [REPEATED] Same instruction already executed. Previous result:\n"
                f"{prev[:600]}\n\n"
                f"Hint: vary instruction substantively or proceed to answer."
            )


        if self._task_type == "code_generation":
            self._python_execute_count += 1
            if self._python_execute_count > 1:
                return (
                    "[python_execute] [LIMIT] For SWE tasks, use python_execute at most once. "
                    "No source edit has been recorded yet; the current repository diff is empty."
                )


        _first_lines = instruction.strip().split('\n')[:5]
        _code_starters = ("import ", "from ", "def ", "class ", "print(", "#!", "try:", "with ")
        instruction_is_code = any(
            line.strip().startswith(_code_starters) for line in _first_lines
        ) or "```python" in instruction or "```\n" in instruction
        if instruction_is_code:
            return (
                "[python_execute] [ERROR] instruction 含代码. 请用自然语言描述要计算什么 — "
                "M_exec 会为你写代码并运行. "
                "例: instruction='Compute the number of lattice paths from (0,0) to (8,8) "
                "using 8 right and 8 up moves with exactly 4 direction changes.' "
                "不要传 Python 源代码."
            )


        exec_prompt = self._build_code_gen_prompt("python_execute", instruction, traj)
        tip_ctx = self._get_injected_tip()
        raw_code = self.m_exec.execute(
            instruction=exec_prompt, context=tip_ctx, task_type=self._task_type,
            max_tokens=4096,
        )
        logger.info(
            f"[DEBUG python_execute] === RAW RESPONSE ({len(raw_code)} chars) ===\n"
            f"{raw_code[:500]}\n=== END ==="
        )
        code = self._extract_code_block(raw_code)
        unreliable_warning = ""
        if self._task_type == "code_generation":
            code_l = code.lower()
            unreliable_markers = (
                "load_dataset(", "fetch_", "download", "requests.", "urllib.",
                "simulate", "simulation", "let's assume", "we don't have the file",
                "since we don't have", "synthetic data", "mock",
            )
            if any(m in code_l for m in unreliable_markers):
                unreliable_warning = (
                    "[python_execute] [WARN] The generated script appears to use "
                    "external/example data or simulated repository behavior. Treat "
                    "the result as weak evidence; the real checked-out source is "
                    "more reliable for SWE fixes.\n\n"
                )
        exec_result = self._execute_python(code, timeout=30)
        if "[ERROR]" in exec_result:
            result_msg = (
                f"[python_execute] [ERROR]\n"
                f"Code:\n{code}\n\n"
                f"Error:\n{exec_result}\n\n"
                f"Hint: Check for syntax errors, undefined variables, or wrong imports. "
                f"For symbolic math, use 'import sympy as sp'."
            )
        else:
            result_lines = [l.strip() for l in exec_result.strip().split('\n') if l.strip()]
            final_value = result_lines[-1] if result_lines else exec_result
            result_msg = (
                f"[python_execute] [OK]\n"
                f"Code:\n{code}\n\n"
                f"Output:\n{exec_result}\n\n"
                f"Final result: {final_value}"
            )
        if unreliable_warning:
            result_msg = unreliable_warning + result_msg

        if hasattr(self, '_recent_tool_calls'):
            self._recent_tool_calls[cache_key] = result_msg
        return result_msg

    def _m_exec_generate_verify_script(self, description: str) -> str:

        prompt = (
            "Write a Python script that verifies whether a specific bug fix works. "
            "The script exits with 0 on PASS (fix works) and 1 on FAIL (bug still present).\n\n"
            f"Verification goal: {description}\n\n"
            "Requirements:\n"
            "- Output ONLY the Python code (no markdown fence, no prose)\n"
            "- Import all modules at the top\n"
            "- If using Django/Flask/etc, configure settings before importing models\n"
            "- Use try/except to catch the expected bug symptom\n"
            "- sys.exit(0) on success (fix works), sys.exit(1) on failure (bug still present)\n"
            "- Keep it minimal — just enough to test the claim\n"
        )
        raw = self.m_exec.execute(
            instruction=prompt, context="", task_type="code_generation",
        )
        code = self._extract_code_block(raw) if "```" in raw else raw
        return code.strip()

    def _handle_verify_fix(self, args: Dict) -> str:

        description = args.get("description", "").strip()
        if not description:

            if args.get("script", "").strip():
                return (
                    "[verify_fix] [ERROR] This tool now takes `description` (natural language). "
                    "Describe what to verify — M_exec will write the script. "
                    "Example: description=\"confirm that xr.DataArray.weighted works when weights contain bool dtype with False values\"."
                )
            return "[verify_fix] [ERROR] `description` is required (describe what to verify)."

        try:
            script = self._m_exec_generate_verify_script(description)
        except Exception as e:
            return f"[verify_fix] [ERROR] M_exec failed to write script: {type(e).__name__}: {str(e)[:300]}"
        if not script:
            return f"[verify_fix] [ERROR] M_exec returned empty script for: {description[:200]}"


        import hashlib

        script_normalized = "\n".join(line.strip() for line in script.split("\n") if line.strip())
        script_hash = hashlib.md5(script_normalized.encode()).hexdigest()[:16]
        cache_key = f"verify_fix::{script_hash}"
        if hasattr(self, '_recent_tool_calls') and cache_key in self._recent_tool_calls:
            prev = self._recent_tool_calls[cache_key]
            return (
                f"[verify_fix] [REPEATED] This script (hash {script_hash}) has been run before "
                f"with same outcome. Previous result:\n{prev[:400]}\n\n"
                f"Edit source code first before re-verifying."
            )

        import subprocess, tempfile


        python_cmd = sys.executable
        cwd = self._repo_path or "/tmp"
        instance_id = self._extra.get("instance_id", "")
        if instance_id and self._repo_path:
            try:
                from training.swe_bench_eval import _load_verified_dataset, _verified_cache, _env_python
                _load_verified_dataset()
                verified = _verified_cache.get(instance_id)
                if verified:
                    env_py = _env_python(verified["repo"], verified["version"])
                    if env_py:
                        python_cmd = str(env_py)
            except Exception:
                pass

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
                f.write(script)
                tmp_path = f.name


            py_path_parts = [cwd] if cwd else []
            if os.environ.get("PYTHONPATH"):
                py_path_parts.append(os.environ["PYTHONPATH"])
            result = subprocess.run(
                [python_cmd, tmp_path],
                capture_output=True, text=True,
                timeout=30, cwd=cwd,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(py_path_parts),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )

            stdout = result.stdout.strip()[-1000:]
            stderr = result.stderr.strip()[-500:]
            exit_code = result.returncode

            if exit_code == 0:
                result_msg = (
                    f"[verify_fix] [PASS] Test passed (exit 0).\n"
                    f"Output: {stdout}\n"
                    f"Your fix appears to be working correctly."
                )
            else:
                result_msg = (
                    f"[verify_fix] [FAIL] Test failed (exit {exit_code}).\n"
                    f"Output: {stdout}\n"
                    f"Error: {stderr}\n"
                    f"Your fix is not correct yet. Read the error above and adjust your edit_file."
                )

            if hasattr(self, '_recent_tool_calls'):
                self._recent_tool_calls[cache_key] = result_msg
            return result_msg

        except subprocess.TimeoutExpired:
            return "[verify_fix] [FAIL] Script timed out (30s). Simplify your test."
        except Exception as e:
            return f"[verify_fix] [ERROR] {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _handle_test_code(self, args: Dict, traj: "Trajectory") -> str:

        instruction = args.get("instruction", "").strip()
        if not instruction:
            return "[test_code] [ERROR] No instruction provided."
        try:
            exec_prompt = self._build_code_gen_prompt("test_code", instruction, traj)
            raw_code = self.m_exec.execute(
                instruction=exec_prompt, context="", task_type="code_generation",
            )
            code = self._extract_code_block(raw_code)
            if not code:
                return f"[test_code] [ERROR] M_exec returned no valid code. Raw (truncated): {raw_code[:300]}"
            test_result = self._test_code(code, "")
            status = "[OK]" if "[ERROR]" not in test_result and "FAIL" not in test_result else "[FAILED]"
            return f"[test_code] {status}\nCode:\n{code}\n\nTest Result:\n{test_result}"
        except Exception as e:
            return f"[test_code] [ERROR] Exception during test_code: {type(e).__name__}: {e}"

    def _handle_analyze(self, args: Dict) -> str:

        instruction = args.get("instruction", "")
        data = args.get("data", "")
        context = data if data else self._format_context()
        result = self.m_exec.execute(
            instruction=instruction, context=context, task_type=self._task_type,
        )
        return f"[Analysis] {result}"


    def _handle_search(self, args: Dict) -> str:

        query = args.get("query", "")
        if not query.strip():
            return "[search] [ERROR] Empty query provided."
        observation = self._search_passages(query)
        logger.info(
            f"[Tool] search: task={self._task_type} query={query[:60]!r}"
        )
        return observation

    def _handle_lookup(self, args: Dict) -> str:

        import re
        keyword = args.get("keyword", "").strip()
        if not keyword:
            return "[lookup] [ERROR] No keyword provided."

        keyword_lower = keyword.lower()
        tokens = keyword_lower.split()

        exact_matches: List[str] = []
        fuzzy_matches: List[str] = []

        for s in self._history:
            obs = s.get("observation", "") or ""
            obs_lower = obs.lower()
            if keyword_lower in obs_lower:

                idx = obs_lower.find(keyword_lower)
                start = max(0, idx - 200)
                end = min(len(obs), idx + len(keyword_lower) + 200)
                snippet = obs[start:end]
                exact_matches.append(f"[Step {s['step']}] ...{snippet}...")
            elif len(tokens) > 1 and all(t in obs_lower for t in tokens):


                idx = obs_lower.find(tokens[0])
                start = max(0, idx - 150)
                end = min(len(obs), idx + 400)
                snippet = obs[start:end]
                fuzzy_matches.append(f"[Step {s['step']}] [fuzzy, all tokens present] ...{snippet}...")

        if exact_matches:
            header = f"[lookup] [OK] Found '{keyword}' in {len(exact_matches)} step(s):"
            return header + "\n" + "\n".join(exact_matches[:5])

        if fuzzy_matches:
            header = (
                f"[lookup] [FUZZY_OK] '{keyword}' not found as exact phrase, but all tokens "
                f"({len(tokens)}) present in {len(fuzzy_matches)} step(s):"
            )
            return header + "\n" + "\n".join(fuzzy_matches[:3])


        from collections import Counter
        all_tokens = []
        for s in self._history:
            obs = s.get("observation", "") or ""

            all_tokens.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', obs))
        top = [w for w, _ in Counter(all_tokens).most_common(10)]
        suggestion = f" Try one of: {', '.join(top[:8])}" if top else ""
        return (
            f"[lookup] [NO_MATCH] '{keyword}' not found in previous observations."
            f"{suggestion} Or try fewer words / different capitalization."
        )


    def _handle_ask_llm(self, args: Dict) -> str:

        question_text = args.get("question", args.get("instruction", ""))
        original_q = str(self._question.get("question", ""))
        evidence = []
        for h in self._history:
            obs = h.get("observation", "")
            if obs:  
                evidence.append(f"[{h['action_type']}] {obs}")
        evidence_text = "\n".join(evidence[-5:]) if evidence else "(no evidence collected yet)"
        ask_prompt = (
            f"Based on all the evidence below, answer the question directly.\n\n"
            f"## Original Question:\n{original_q}\n\n"
            f"## Evidence Gathered:\n{evidence_text}\n\n"
            f"## Additional Context:\n{question_text}\n\n"
            f"Give ONLY the final answer, no explanation. Be concise."
        )
        result = self.m_exec.execute(ask_prompt, task_type=self._task_type)
        return f"[LLM Answer] {result}"

    def _handle_self_consistency(self, args: Dict) -> str:

        instruction = args.get("instruction", "")
        question = str(self._question.get("question", ""))[:500]
        sc_prompt = (
            f"Solve this problem step by step and give ONLY the final answer on the last line.\n\n"
            f"Problem: {question}\n"
            f"Additional context: {instruction}\n"
        )
        answers = []
        for _ in range(3):
            raw = self.m_exec.execute(sc_prompt, task_type=self._task_type, temperature=0.7)
            lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
            answers.append(lines[-1] if lines else "")
        from collections import Counter

        non_empty = [a for a in answers if a]
        if not non_empty:
            return (
                f"[Self-Consistency] [ERROR] All 3 attempts returned empty. "
                f"M_exec may be unavailable or problem too complex. Try a simpler approach."
            )
        vote = Counter(non_empty)
        majority, count = vote.most_common(1)[0]
        confidence = count / len(answers)
        return (
            f"[Self-Consistency] {len(answers)} attempts, majority answer: {majority}\n"
            f"Confidence: {confidence:.0%} ({count}/{len(answers)} agree)\n"
            f"All answers: {answers}"
        )


    def _handle_verify_answer(self, args: Dict) -> str:

        candidate = args.get("answer", args.get("instruction", ""))
        method = args.get("method", "substitute")
        question = str(self._question.get("question", ""))[:500]
        if method == "test" and self._task_type == "code_generation":
            return self._test_code(candidate, "")
        verify_prompt = (
            f"Verify this answer by substituting it back into the original problem.\n\n"
            f"Problem: {question}\n"
            f"Candidate answer: {candidate}\n\n"
            f"Write Python code that checks if this answer is correct. "
            f"Print 'VERIFIED' if correct, 'WRONG' with explanation if not."
        )
        raw = self.m_exec.execute(verify_prompt, task_type="math_reasoning")
        code = self._extract_code_block(raw)
        if code:
            return self._execute_python(code)

        return f"[verify_answer] [ERROR] Could not generate verification code. Raw response (truncated): {raw[:500]}"

    def _handle_check_answer(self, args: Dict) -> str:

        answer = args.get("answer", "").strip()
        if not answer:
            return "[check_answer] [ERROR] No answer provided."
        question = str(self._question.get("question", ""))[:500]
        check_prompt = (
            f"Check this answer for format correctness and plausibility.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Is the format correct? Is it plausible? Reply with PASS or FAIL with reason."
        )
        result = self.m_exec.execute(check_prompt, task_type=self._task_type)
        result_truncated = result[:1500]
        rl = result_truncated.lower()
        if "pass" in rl[:100]:
            status = "[PASS]"
        elif "fail" in rl[:100]:
            status = "[FAIL]"
        else:
            status = "[WARN]"
        return f"[Check] {status}\n{result_truncated}"

    def _handle_cross_validate(self, args: Dict) -> str:

        answer = args.get("answer", "").strip()
        if not answer:
            return "[cross_validate] [ERROR] No answer provided."
        question = str(self._question.get("question", ""))[:500]
        cv_prompt = (
            f"Solve this problem using a DIFFERENT method than before, "
            f"then compare with the candidate answer.\n\n"
            f"Problem: {question}\n"
            f"Candidate answer: {answer}\n\n"
            f"Use an alternative approach. State your answer and whether it matches."
        )
        result = self.m_exec.execute(cv_prompt, task_type=self._task_type)
        result_truncated = result[:1500]
        rl = result_truncated.lower()
        if "match" in rl and "mismatch" not in rl and "does not match" not in rl:
            indicator = "[MATCH]"
        elif "mismatch" in rl or "does not match" in rl or "differ" in rl:
            indicator = "[DIFF]"
        else:
            indicator = "[UNCLEAR]"
        return f"[Cross-Validation] {indicator}\n{result_truncated}"


    _SWE_BENCH_REPOS = os.environ.get("SWE_BENCH_ENVS", "swe_bench_envs")

    def _setup_swe_repo(self) -> Optional[str]:

        import subprocess, tempfile
        try:
            from training.swe_bench_eval import (
                _load_verified_dataset, _verified_cache, _repo_dir,
            )
            _load_verified_dataset()
            instance_id = self._extra.get("instance_id", "")
            verified = _verified_cache.get(instance_id)
            if not verified:
                return None
            repo = verified["repo"]
            base_commit = verified["base_commit"]
            repo_path = _repo_dir(repo)
            if not repo_path or not repo_path.exists():
                return None


            worktree_dir = tempfile.mkdtemp(prefix=f"swe_{instance_id.replace('/', '_')[:30]}_")
            result = subprocess.run(
                ["git", "worktree", "add", worktree_dir, base_commit, "--detach", "-q"],
                cwd=str(repo_path), capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:

                repo_str = str(repo_path)
                subprocess.run(["git", "checkout", base_commit, "-q"], cwd=repo_str, capture_output=True, timeout=30)
                subprocess.run(["git", "checkout", ".", "-q"], cwd=repo_str, capture_output=True, timeout=10)
                subprocess.run(["git", "clean", "-fd", "-q"], cwd=repo_str, capture_output=True, timeout=10)
                logger.info(f"[SWE] Fallback checkout {repo} @ {base_commit[:10]} at {repo_str}")
                return repo_str

            self._worktree_dir = worktree_dir  
            self._worktree_repo = str(repo_path)
            logger.info(f"[SWE] Worktree {repo} @ {base_commit[:10]} at {worktree_dir}")
            return worktree_dir
        except Exception as e:
            logger.warning(f"[SWE] Failed to setup repo: {e}")
            return None

    def cleanup(self):

        import subprocess, shutil
        wt = getattr(self, '_worktree_dir', None)
        repo = getattr(self, '_worktree_repo', None)
        if wt and repo:
            try:
                subprocess.run(["git", "worktree", "remove", wt, "--force"],
                              cwd=repo, capture_output=True, timeout=10)
            except Exception:
                pass
            try:
                shutil.rmtree(wt, ignore_errors=True)
            except Exception:
                pass


    _MAX_BASH_OUTPUT = 10000
    _MAX_RESPONSE_LEN = 16000

    def _handle_bash(self, args: Dict) -> str:

        import subprocess
        command = args.get("command", "")
        if not command.strip():
            return "No command provided."
        cwd = self._repo_path or "/tmp"

        dangerous = ["rm -rf /", "mkfs", "dd if=", "> /dev/"]
        if any(d in command for d in dangerous):
            return "Command rejected for safety."
        try:

            env_prefix = ""
            instance_id = self._extra.get("instance_id", "")
            if instance_id and self._repo_path:
                try:
                    from training.swe_bench_eval import _load_verified_dataset, _verified_cache, _env_python
                    _load_verified_dataset()
                    verified = _verified_cache.get(instance_id)
                    if verified:
                        env_py = _env_python(verified["repo"], verified["version"])
                        if env_py:
                            conda_env = str(env_py).split("/envs/")[1].split("/")[0] if "/envs/" in str(env_py) else ""
                            if conda_env:
                                env_prefix = f"conda run -n {conda_env} --no-capture-output "
                except Exception:
                    pass

            full_cmd = f"{env_prefix}{command}" if env_prefix else command
            result = subprocess.run(
                full_cmd, shell=True,
                capture_output=True, text=True,
                timeout=60, cwd=cwd,
                env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"},
            )
            output = result.stdout + result.stderr
            if len(output) > self._MAX_BASH_OUTPUT:
                half = self._MAX_BASH_OUTPUT // 2
                output = (
                    output[:half]
                    + f"\n\n... ({len(output) - self._MAX_BASH_OUTPUT} chars truncated) ...\n\n"
                    + output[-half:]
                )
            if not output.strip():
                if result.returncode != 0:
                    return f"[FAILED] Command exited with code {result.returncode} (no output)."
                return "Command ran successfully with no output."
            if result.returncode != 0:
                return f"[EXIT {result.returncode}]\n{output.strip()}"
            return output.strip()
        except subprocess.TimeoutExpired:
            return "Command timed out (60s). Try a simpler command."
        except Exception as e:
            return f"Error: {e}"

    def _handle_str_replace_editor(self, args: Dict) -> str:

        command = args.get("command", "")
        path = args.get("path", "")
        logger.info(f"[SRE] command={command}, path={path[:80]}")

        if not path:
            return "Error: path is required."


        if self._repo_path and not os.path.isabs(path):
            full_path = os.path.join(self._repo_path, path.lstrip("./"))
        else:
            full_path = path

        if command == "view":
            return self._sre_view(full_path, path, args.get("view_range"))
        elif command == "create":
            return self._sre_create(full_path, path, args.get("file_text", ""))
        elif command == "str_replace":
            return self._sre_str_replace(full_path, path, args.get("old_str", ""), args.get("new_str", ""))
        elif command == "insert":
            return self._sre_insert(full_path, path, args.get("insert_line"), args.get("new_str", ""))
        elif command == "undo_edit":
            return self._sre_undo(full_path, path)
        else:
            return f"Error: unknown command '{command}'. Use: view, create, str_replace, insert, undo_edit."

    def _sre_view(self, full_path: str, display_path: str, view_range) -> str:

        if os.path.isdir(full_path):
            import subprocess
            result = subprocess.run(
                ["find", ".", "-maxdepth", "2", "-not", "-path", "./.git/*"],
                cwd=full_path, capture_output=True, text=True, timeout=5,
            )
            entries = sorted(result.stdout.strip().split('\n'))[:100]
            return f"Here's the files and directories in {display_path}:\n" + "\n".join(entries)

        if not os.path.isfile(full_path):
            return f"Error: {display_path} not found."
        try:
            with open(full_path, 'r', errors='replace') as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading {display_path}: {e}"

        total = len(lines)
        start, end = 1, total
        if view_range:
            if isinstance(view_range, str):
                nums = re.findall(r"-?\d+", view_range)
                view_range = [_coerce_int(n, -1) for n in nums]
            if view_range:
                start = max(1, _coerce_int(view_range[0], 1))
                end_arg = -1 if len(view_range) < 2 else _coerce_int(view_range[1], -1)
                end = total if end_arg == -1 else min(total, end_arg)
                if end < start:
                    end = min(total, start)


        if end - start + 1 > 300:
            end = start + 299

        numbered = "".join(f"{i:6d}\t{lines[i-1]}" for i in range(start, end + 1))
        if len(numbered) > self._MAX_RESPONSE_LEN:
            numbered = numbered[:self._MAX_RESPONSE_LEN] + "\n<response clipped>"
        result = f"Here's the result of running `cat -n` on {display_path}:\n{numbered}"
        if end < total:
            result += f"\n({total - end} more lines below)"
        return result

    def _sre_create(self, full_path: str, display_path: str, file_text: str) -> str:

        if os.path.exists(full_path):
            return f"Error: {display_path} already exists. Use str_replace to edit it."
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(file_text)
            return f"File created at {display_path}."
        except Exception as e:
            return f"Error creating file: {e}"

    def _sre_str_replace(self, full_path: str, display_path: str, old_str: str, new_str: str) -> str:

        if not os.path.isfile(full_path):
            return f"Error: {display_path} not found."
        try:
            with open(full_path, 'r', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading {display_path}: {e}"

        if old_str not in content:
            return (
                f"Error: `old_str` not found in {display_path}. "
                f"Make sure it matches EXACTLY, including whitespace and indentation."
            )
        if content.count(old_str) > 1:
            return (
                f"Error: `old_str` appears {content.count(old_str)} times in {display_path}. "
                f"Include more context to make it unique."
            )


        new_content = content.replace(old_str, new_str, 1)
        if full_path.endswith('.py'):
            try:
                compile(new_content, display_path, 'exec')
            except SyntaxError as e:
                return (
                    f"Syntax error in edit: {e.msg} at line {e.lineno}. "
                    f"File was NOT modified. Fix the syntax and retry."
                )


        self._edit_history[full_path] = content
        with open(full_path, 'w') as f:
            f.write(new_content)


        edit_pos = content.index(old_str)
        line_num = content[:edit_pos].count('\n') + 1
        new_lines = new_content.split('\n')
        n_new = new_str.count('\n') + 1
        ctx_start = max(1, line_num - 3)
        ctx_end = min(len(new_lines), line_num + n_new + 3)
        snippet = "".join(f"{i:6d}\t{new_lines[i-1]}\n" for i in range(ctx_start, ctx_end + 1))
        return f"The file {display_path} has been edited. Here's the result of running `cat -n` on a snippet:\n{snippet}"

    def _sre_insert(self, full_path: str, display_path: str, insert_line, new_str: str) -> str:

        if not os.path.isfile(full_path):
            return f"Error: {display_path} not found."
        if insert_line is None:
            return "Error: insert_line is required."
        try:
            with open(full_path, 'r', errors='replace') as f:
                lines = f.readlines()
            self._edit_history[full_path] = "".join(lines)
            insert_line = _coerce_int(insert_line, 0)
            insert_line = max(0, min(len(lines), insert_line))
            new_lines = new_str.split('\n')
            for i, nl in enumerate(new_lines):
                lines.insert(insert_line + i, nl + '\n')
            with open(full_path, 'w') as f:
                f.writelines(lines)
            return f"The file {display_path} has been edited. Line(s) inserted after line {insert_line}."
        except Exception as e:
            return f"Error: {e}"

    def _sre_undo(self, full_path: str, display_path: str) -> str:

        if full_path not in self._edit_history:
            return f"No edit history for {display_path}."
        try:
            with open(full_path, 'w') as f:
                f.write(self._edit_history.pop(full_path))
            return f"Last edit to {display_path} has been undone."
        except Exception as e:
            return f"Error: {e}"


    def _handle_list_files(self) -> str:

        if self._repo_path:
            import subprocess
            try:

                result = subprocess.run(
                    ["find", ".", "-maxdepth", "3", "-type", "f", "-name", "*.py",
                     "-not", "-path", "./.git/*", "-not", "-path", "*/test*/*",
                     "-not", "-path", "*/docs/*"],
                    cwd=self._repo_path, capture_output=True, text=True, timeout=10,
                )
                files = sorted(result.stdout.strip().split('\n'))

                dirs: Dict[str, list] = {}
                for f in files:
                    if not f.strip():
                        continue
                    parts = f.split('/')
                    if len(parts) >= 2:
                        d = '/'.join(parts[:2])
                    else:
                        d = '.'
                    dirs.setdefault(d, []).append(f)
                output_lines = []
                for d in sorted(dirs.keys())[:30]:
                    flist = dirs[d]
                    if len(flist) <= 5:
                        for f in flist:
                            output_lines.append(f"  {f}")
                    else:
                        output_lines.append(f"  {d}/ ({len(flist)} files)")
                        for f in flist[:3]:
                            output_lines.append(f"    {f}")
                        output_lines.append(f"    ... and {len(flist)-3} more")
                return (
                    f"[list_files] [OK] Source files (excluding tests/docs):\n"
                    + "\n".join(output_lines[:60])
                    + "\n\nUse search_code to find specific code, or view_file to read a file."
                )
            except Exception as e:
                return f"[list_files] [ERROR] {e}"
        if not self._code_workspace:
            return "[list_files] [ERROR] No code workspace loaded for this task."
        files_info = []
        for path, content in sorted(self._code_workspace.items()):
            lines = content.count('\n') + 1
            chars = len(content)
            files_info.append(f"  {path} ({lines} lines, {chars} chars)")
        return f"[list_files] [OK] {len(self._code_workspace)} file(s) in workspace:\n" + "\n".join(files_info)

    def _resolve_repo_path(self, path: str) -> str:

        clean = str(path or "").strip().rstrip("/")
        if not self._repo_path:
            return clean
        if clean.startswith(self._repo_path):
            return clean
        if not os.path.isabs(clean):
            return os.path.join(self._repo_path, clean.lstrip("./"))
        if os.path.exists(clean):
            return clean
        parts = [p for p in clean.split(os.sep) if p]


        for i in range(len(parts)):
            rel = os.path.join(*parts[i:])
            cand = os.path.join(self._repo_path, rel)
            if os.path.exists(cand):
                return cand
        return clean

    def _handle_search_code(self, args: Dict) -> str:

        query = args.get("query", "")
        file_pattern = args.get("file_pattern", "")


        if self._repo_path:
            import subprocess


            search_query = query
            if len(query) > 60:

                tokens = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z_]{4,}|[A-Z]{2,}', query)
                if tokens:

                    tokens = sorted(set(tokens), key=len, reverse=True)[:3]
                    search_query = tokens[0]  

            try:
                _exclude = ["--exclude-dir=tests", "--exclude-dir=test",
                            "--exclude-dir=testing", "--exclude-dir=docs",
                            "--exclude-dir=doc", "--exclude-dir=examples",
                            "--exclude-dir=.git"]

                def _short_grep_lines(text: str, limit: int = 12) -> str:
                    return "\n".join(l[:150] for l in text.strip().split("\n")[:limit] if l.strip())

                def _relaxed_source_hint(pattern_files=None) -> str:

                    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(search_query or ""))
                    stop = {
                        "class", "def", "return", "self", "from", "import",
                        "true", "false", "none", "and", "or", "not",
                    }
                    candidates = []
                    for tok in raw_tokens:
                        base = tok.strip("_")
                        base = re.sub(r"\d+$", "", base)
                        parts = [base]
                        if "_" in base:
                            parts.extend(p for p in base.split("_") if p)
                        for p in parts:
                            pl = p.lower()
                            if len(p) >= 4 and pl not in stop:
                                candidates.append(p)
                    for token in sorted(set(candidates), key=len, reverse=True)[:4]:
                        try:
                            if pattern_files:
                                rcmd = ["grep", "-nH", "-F", "-I", "-C", "1", "-m", "10", token] + list(pattern_files)
                            else:
                                rcmd = ["grep", "-rn", "-F", "-I", "--include=*.py", "-C", "1"] + _exclude
                                rcmd.extend(["-m", "10", token, "."])
                            rr = subprocess.run(
                                rcmd, cwd=self._repo_path,
                                capture_output=True, text=True, timeout=10,
                            )
                        except Exception:
                            continue
                        if rr.stdout.strip():
                            return (
                                f"\nRelaxed source search for token '{token}' found possible locations:\n"
                                f"{_short_grep_lines(rr.stdout, 10)}\n"
                            "These are source locations returned by a relaxed search, not new hidden information."
                            )
                    return ""

                def _issue_member_query_note() -> str:

                    mentions = self._swe_issue_member_mentions()
                    if not mentions:
                        return ""
                    q_l = str(search_query or "").lower()
                    missing = []
                    for cls, member in mentions:
                        full = f"{cls}.{member}"
                        if member.lower() in q_l or full.lower() in q_l:
                            continue
                        missing.append(full)
                    if not missing:
                        return ""
                    return (
                        "\n[SWE_NOTE] Issue-visible member(s) not named in this "
                        "NO_MATCH query: " + ", ".join(missing[:4]) + "."
                    )

                target_files = None
                if file_pattern:


                    fp = file_pattern.strip().lstrip("./")
                    fp_abs = os.path.join(self._repo_path, fp)
                    if "/" in fp and os.path.isfile(fp_abs):
                        target_files = [fp]
                    elif "/" in fp and os.path.isdir(fp_abs):
                        find_result = subprocess.run(
                            ["find", fp, "-type", "f", "-name", "*.py", "-not", "-path", "*/.git/*"],
                            cwd=self._repo_path,
                            capture_output=True, text=True, timeout=5,
                        )
                        target_files = sorted(
                            p for p in find_result.stdout.strip().split("\n")
                            if p.strip()
                        )
                    else:


                        find_args = ["find", ".", "-type", "f", "-not", "-path", "./.git/*"]
                        if any(ch in fp for ch in "*?[]"):
                            if "/" in fp:
                                find_args.extend(["-path", f"*{fp}"])
                            else:
                                find_args.extend(["-name", fp])
                        else:
                            if "/" in fp:
                                find_args.extend(["-path", f"*{fp}*"])
                            else:


                                find_args.extend(["-name", fp if "." in fp else f"*{fp}*"])
                        find_result = subprocess.run(
                            find_args, cwd=self._repo_path,
                            capture_output=True, text=True, timeout=5,
                        )
                        all_target_files = sorted(
                            p for p in find_result.stdout.strip().split("\n")
                            if p.strip()
                        )


                        max_grep_files = _coerce_int(os.environ.get("SWE_SEARCH_MAX_FILES"), 5000)
                        target_files = all_target_files[:max_grep_files]
                    if target_files and len(target_files) > _coerce_int(os.environ.get("SWE_SEARCH_MAX_FILES"), 5000):
                        target_files = target_files[:_coerce_int(os.environ.get("SWE_SEARCH_MAX_FILES"), 5000)]
                    if not target_files:
                        return (
                            f"[search_code] [NO_MATCH] No files matched file_pattern='{file_pattern}'. "
                            f"Use list_files or a broader file_pattern."
                        )
                    cmd = ["grep", "-nH", "-E", "-I", "-C", "2", "-m", "20", search_query] + target_files
                else:
                    cmd = ["grep", "-rn", "-E", "-I", "--include=*.py", "-C", "2"] + _exclude
                    cmd.extend(["-m", "20", search_query, "."])
                result = subprocess.run(
                    cmd, cwd=self._repo_path,
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode == 2:

                    if file_pattern and target_files:
                        cmd = ["grep", "-nH", "-F", "-I", "-C", "2", "-m", "20", search_query] + target_files
                    else:
                        cmd = ["grep", "-rn", "-F", "-I", "--include=*.py", "-C", "2"] + _exclude
                        cmd.extend(["-m", "20", search_query, "."])
                    result = subprocess.run(
                        cmd, cwd=self._repo_path,
                        capture_output=True, text=True, timeout=15,
                    )
                output = result.stdout.strip()
                if not output:
                    if file_pattern:
                        sample_files = ", ".join((target_files or [])[:5])
                        more = "" if not target_files or len(target_files) <= 5 else f", ... ({len(target_files)} files)"
                        global_hint = ""


                        cmd_global = ["grep", "-rn", "-E", "-I", "--include=*.py", "-C", "1"] + _exclude
                        cmd_global.extend(["-m", "10", search_query, "."])
                        g = subprocess.run(
                            cmd_global, cwd=self._repo_path,
                            capture_output=True, text=True, timeout=10,
                        )
                        if g.returncode == 2:
                            cmd_global = ["grep", "-rn", "-F", "-I", "--include=*.py", "-C", "1"] + _exclude
                            cmd_global.extend(["-m", "10", search_query, "."])
                            g = subprocess.run(
                                cmd_global, cwd=self._repo_path,
                                capture_output=True, text=True, timeout=10,
                            )
                        if g.stdout.strip():
                            gh_lines = [l[:150] for l in g.stdout.strip().split("\n")[:12]]
                            global_hint = (
                                "\nSame query has matches outside that file_pattern; "
                                "same-query matches outside the requested file_pattern:\n"
                                + "\n".join(gh_lines)
                            )
                        definition_absence_note = ""
                        m_def_query = re.match(
                            r"\s*(?:async\s+def\s+|def\s+)?([A-Za-z_][A-Za-z0-9_]*)\b",
                            str(search_query or ""),
                        )
                        if m_def_query and re.search(r"\bdef\b", str(search_query or "")):
                            definition_absence_note = (
                                "\n[SWE_NOTE] In the requested file_pattern, this definition query "
                                "returned no source definition. Any same-named matches outside the "
                                "file_pattern are separate source evidence, not evidence that the "
                                "definition exists inside the requested file."
                            )
                        relaxed_hint = _relaxed_source_hint(target_files)
                        issue_member_note = _issue_member_query_note()
                        return (
                            f"[search_code] [NO_MATCH] No matches for '{search_query}' "
                            f"in file_pattern='{file_pattern}' ({sample_files}{more}). "
                            f"No source hit was returned inside this file_pattern."
                            f"{global_hint}"
                            f"{definition_absence_note}"
                            f"{issue_member_note}"
                            f"{relaxed_hint}"
                        )

                    cmd2 = ["grep", "-rn", "-E", "-I", "--include=*.py",
                            "--exclude-dir=.git", "-C", "2",
                            "-m", "15", search_query, "."]
                    result = subprocess.run(
                        cmd2, cwd=self._repo_path,
                        capture_output=True, text=True, timeout=15,
                    )
                    if result.returncode == 2:
                        cmd2 = ["grep", "-rn", "-F", "-I", "--include=*.py",
                                "--exclude-dir=.git", "-C", "2",
                                "-m", "15", search_query, "."]
                        result = subprocess.run(
                            cmd2, cwd=self._repo_path,
                            capture_output=True, text=True, timeout=15,
                        )
                    output = result.stdout.strip()
                if not output:

                    find_result = subprocess.run(
                        ["find", ".", "-name", f"*{search_query}*", "-type", "f",
                         "-not", "-path", "./.git/*"],
                        cwd=self._repo_path, capture_output=True, text=True, timeout=5,
                    )
                    found_files = find_result.stdout.strip()
                    hint = ""
                    if found_files:
                        files = found_files.split('\n')[:10]
                        hint = f" Found {len(files)} file(s) with similar names:\n" + "\n".join(f"  {f}" for f in files)
                    relaxed_hint = _relaxed_source_hint()
                    issue_member_note = _issue_member_query_note()
                    return (
                        f"[search_code] [NO_MATCH] No matches for '{search_query}' in source code.{hint}\n"
                        f"No source hit was returned for this query."
                        f"{issue_member_note}"
                        f"{relaxed_hint}"
                    )
                lines = output.split('\n')[:50]
                lines = [l[:150] for l in lines]
                hint = ""
                if "raise NotImplementedError" in output:
                    placeholder = ""
                    m_ph = re.search(
                        r"(?m)^(.+?):(\d+):\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\b[\s\S]{0,300}?raise NotImplementedError",
                        output,
                    )
                    query_token = re.sub(r"\W+", "", str(search_query or "")).lower()
                    if m_ph and query_token and query_token == m_ph.group(3).lower():
                        ph_path = m_ph.group(1).lstrip("./")
                        ph_line = m_ph.group(2)
                        ph_name = m_ph.group(3)
                        placeholder = (
                            f" Matching placeholder location: {ph_path}:{ph_line} "
                            f"(near lines {max(1, _coerce_int(ph_line, 1) - 5)}-"
                            f"{_coerce_int(ph_line, 1) + 10}) for `{ph_name}`."
                        )
                    if placeholder:


                        hint = (
                            "[SWE_NOTE] Existing placeholder/NotImplemented source location "
                            f"matching the query name.{placeholder}\n"
                        )
                return f"[search_code] [OK] {hint}Results for '{search_query}':\n" + "\n".join(lines)
            except subprocess.TimeoutExpired:
                return f"[search_code] Search timed out; no source evidence was returned for this query."
            except Exception as e:
                return f"[search_code] [ERROR] {e}"


        if not self._code_workspace:
            return f"[search_code] [ERROR] No code workspace loaded."
        import fnmatch
        use_regex = True
        try:
            re.compile(query)
        except re.error:
            use_regex = False
        keywords = [k.strip() for k in query.split() if len(k.strip()) > 1]
        matches = []

        def _workspace_path_matches(path: str, pattern: str) -> bool:

            if not pattern:
                return True
            fp = str(pattern).strip().lstrip("./")
            norm_path = str(path).strip().lstrip("./")
            base = os.path.basename(norm_path)
            if fnmatch.fnmatch(norm_path, fp) or fnmatch.fnmatch("./" + norm_path, fp):
                return True
            if "/" not in fp:
                return fnmatch.fnmatch(base, fp) or base == fp or ("." not in fp and fp in base)
            return fnmatch.fnmatch(norm_path, f"*{fp}*")

        for path, content in self._code_workspace.items():
            if file_pattern and not _workspace_path_matches(path, file_pattern):
                continue
            for i, line in enumerate(content.split('\n'), 1):
                matched = False
                if use_regex:
                    try:
                        if re.search(query, line, re.IGNORECASE):
                            matched = True
                    except re.error:
                        pass
                if not matched and keywords:
                    if any(kw.lower() in line.lower() for kw in keywords):
                        matched = True
                if matched:
                    matches.append(f"{path}:{i}: {line.rstrip()[:150]}")
        if not matches:
            avail = list(self._code_workspace.keys())
            return (
                f"[search_code] [NO_MATCH] No matches for '{query}'. "
                f"Available files: {avail}. No source hit was returned for this query."
            )
        hint = ""
        if any("raise NotImplementedError" in m for m in matches):
            query_token = re.sub(r"\W+", "", str(query or "")).lower()
            if query_token and any(re.search(rf"\bdef\s+{re.escape(query_token)}\b", m, re.I) for m in matches):
                hint = (
                    "[SWE_NOTE] Existing placeholder/NotImplemented source location "
                    "matching the query name.\n"
                )
        return f"[search_code] [OK] {hint}{len(matches)} match(es):\n" + "\n".join(matches[:30])

    def _handle_view_file(self, args: Dict) -> str:

        path = args.get("path", "")
        if self._task_type == "code_generation" and not str(path or "").strip():
            return (
                "[view_file] [ERROR] `path` is required for source inspection. "
                "No file content was returned because the requested path was empty."
            )


        if self._repo_path:
            clean_path = path.strip()
            full_path = self._resolve_repo_path(clean_path)
            if not os.path.exists(full_path):

                full_path = os.path.join(self._repo_path, clean_path.lstrip("./").rstrip("/"))
                if not os.path.exists(full_path):
                    return f"[view_file] [ERROR] File '{path}' not found. Use search_code or list_files to find files."

            if os.path.isdir(full_path):
                pass  
            else:
                try:
                    with open(full_path, 'r', errors='replace') as f:
                        content = f.read()
                except Exception as e:
                    return f"[view_file] [ERROR] Cannot read '{path}': {e}"
        elif path in self._code_workspace:
            content = self._code_workspace[path]
        else:
            avail = list(self._code_workspace.keys())[:20]
            return f"[view_file] [ERROR] File '{path}' not found. Available files: {avail}"


        actual_path = locals().get('full_path') if self._repo_path else None
        if actual_path and os.path.isdir(actual_path):
            import subprocess
            result = subprocess.run(
                ["find", ".", "-maxdepth", "2", "-not", "-path", "./.git/*"],
                cwd=actual_path, capture_output=True, text=True, timeout=5,
            )
            entries = sorted(result.stdout.strip().split('\n'))[:80]
            return f"[view_file] [OK] Directory {path}:\n" + "\n".join(f"  {e}" for e in entries)

        lines = content.split('\n')
        total = len(lines)
        has_start = "start_line" in args
        has_end = "end_line" in args
        has_range = has_start or has_end
        view_max_lines = max(20, _coerce_int(os.environ.get("SWE_VIEW_MAX_LINES", 140), 140))
        if has_end and not has_start:
            end_hint = _coerce_int(args.get("end_line", total), total)
            start_val = max(1, end_hint - view_max_lines + 1)
        else:
            start_val = _coerce_int(args.get("start_line", 1), 1)
        if has_start and not has_end:


            end_val = start_val + view_max_lines - 1
        else:
            end_val = _coerce_int(args.get("end_line", total), total)
        start = max(1, start_val) - 1
        end = min(total, end_val)
        if end <= start:
            end = min(total, start + 1)
        range_note = ""


        if not has_range and total > 500 and path.endswith('.py'):
            filemap = self._generate_filemap(content, path)
            if filemap:
                return (
                    f"[view_file] [OK] {path} ({total} lines) — file structure:\n"
                    f"{filemap}\n\n"
                    f"No full-file content is shown in this observation."
            )

            end = min(total, 200)
        elif has_range and (end - start) > view_max_lines:
            requested_start, requested_end = start + 1, end
            end = min(total, start + view_max_lines)
            range_note = (
                f"\n[SWE_NOTE] Requested view range {requested_start}-{requested_end} "
                f"was truncated to {start + 1}-{end} ({view_max_lines} line window) "
                "to keep the model context within budget. Request the next adjacent "
                "window explicitly if more source is needed."
            )

        numbered = "\n".join(f"{i+1:4d} | {lines[i]}" for i in range(start, end))
        above = f"({start} more lines above)\n" if start > 0 else ""
        below = f"\n({total - end} more lines below)" if end < total else ""
        return f"[view_file] [OK] {path} (lines {start+1}-{end}, total {total} lines):\n{above}{numbered}{below}{range_note}"

    @staticmethod
    def _generate_filemap(content: str, path: str) -> Optional[str]:

        import ast
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        lines = content.split('\n')
        result = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno
                end = node.end_lineno or start
                body_len = end - start

                indent = ""

                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in parent.body:
                            if child is node:
                                indent = "    "
                                break

                if isinstance(node, ast.ClassDef):
                    sig = lines[start - 1].rstrip()
                    result.append((start, f"{start:4d} | {sig}"))
                    if body_len > 5:
                        result.append((start + 0.5, f"     | ...({body_len} lines)"))
                else:
                    sig = lines[start - 1].rstrip()
                    result.append((start, f"{start:4d} | {sig}"))
                    if body_len > 3:
                        result.append((start + 0.5, f"     | ...({body_len} lines)"))


        imports = []
        for i, line in enumerate(lines[:15]):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append(f"{i+1:4d} | {line.rstrip()}")

        if not result and not imports:
            return None

        result.sort(key=lambda x: x[0])
        output = "\n".join(imports[:5])
        if imports:
            output += "\n     | ..."
        output += "\n" + "\n".join(r[1] for r in result)
        return output

    def _literal_replacement_candidates_from_instruction(
        self,
        instruction: str,
    ) -> List[Tuple[str, str]]:

        import re as _re

        text = " ".join(str(instruction or "").strip().split())
        if not text:
            return []
        candidates: List[Tuple[str, str]] = []
        q = r"['\"`]"


        for m in _re.finditer(
            rf"\b(?:replace|change)\s+(?:the\s+text\s+)?(?P<q1>{q})(?P<old>.+?)(?P=q1)\s+(?:with|to)\s+(?P<q2>{q})(?P<new>.+?)(?P=q2)",
            text,
            flags=_re.I,
        ):
            candidates.append((m.group("old"), m.group("new")))


        for m in _re.finditer(
            rf"\bfrom\s+(?P<q1>{q})(?P<old>.+?)(?P=q1)\s+to\s+(?P<q2>{q})(?P<new>.+?)(?P=q2)",
            text,
            flags=_re.I,
        ):
            candidates.append((m.group("old"), m.group("new")))


        code_atom = (
            r"[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
            r"(?:\[[^\]\n]{1,80}\])?"
            r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
        )
        for pat in (
            rf"\bfrom\s+(?P<old>{code_atom})\s+to\s+(?P<new>{code_atom})\b",
            rf"\b(?:replace|change)\s+(?:line\s+\d+\s+from\s+)?(?P<old>{code_atom})\s+(?:with|to)\s+(?P<new>{code_atom})\b",
        ):
            for m in _re.finditer(pat, text, flags=_re.I):
                candidates.append((m.group("old"), m.group("new")))

        out: List[Tuple[str, str]] = []
        seen_pairs = set()
        for old, new in candidates:
            old = (old or "").strip()
            new = (new or "").strip()
            if not old or not new or old == new:
                continue
            key = (old, new)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out.append(key)
        return out

    def _literal_replacement_from_instruction(
        self,
        instruction: str,
        file_content: str,
        edit_context: str = "",
    ) -> Optional[Tuple[str, str]]:

        for old, new in self._literal_replacement_candidates_from_instruction(instruction):
            if old not in file_content:
                continue


            if file_content.count(old) == 1:
                return old, new


            for line in file_content.splitlines(keepends=True):
                if old not in line:
                    continue
                line_no_eol = line.rstrip("\n")
                if edit_context and line_no_eol not in edit_context:
                    continue
                if file_content.count(line) == 1:
                    return line, line.replace(old, new, 1)
        return None

    def _m_exec_generate_edit(
        self,
        path: str,
        instruction: str,
        file_content: str,
        extra_context: str = "",
        strict_note: str = "",
    ) -> Tuple[str, str]:

        import json as _json
        import re as _re
        import ast as _ast
        max_chars = 12000
        if len(file_content) > max_chars:
            half = max_chars // 2
            content_for_prompt = (
                file_content[:half]
                + f"\n\n... [TRUNCATED {len(file_content) - max_chars} chars of middle] ...\n\n"
                + file_content[-half:]
            )
        else:
            content_for_prompt = file_content
        issue_text = str(getattr(self, "_question", {}).get("question", "") or "")
        if len(issue_text) > 2500:
            issue_text = issue_text[:2500] + "\n[ISSUE TRUNCATED]"
        extra_context_block = ""
        if extra_context:
            extra_context_block = (
                "Additional recent source/search context (read-only; NOT the edit target; "
                "do not copy old_content from here):\n"
                f"```\n{extra_context[:2000]}\n```\n\n"
            )

        prompt = (
            "You are a precise source-code editor. Produce one minimal exact string replacement.\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            '{"old_content": "...", "new_content": "..."}\n\n'
            "Source of truth: the issue text and the target file/excerpt below. Do not use tests, gold patches, or hidden feedback.\n\n"
            f"Issue:\n{issue_text}\n\n"
            f"{extra_context_block}"
            f"Target file: {path}\n"
            f"Target file content/excerpt:\n```\n{content_for_prompt}\n```\n\n"
            f"Requested change:\n{instruction}\n\n"
            "Rules:\n"
            "1. old_content must be copied verbatim from the target file/excerpt; do not include displayed line numbers.\n"
            "2. Keep old_content short and unique, usually 1-12 lines. For an insertion, replace a small anchor block with anchor+inserted code.\n"
            "3. new_content is replacement text, not a diff. Preserve indentation, imports, aliases, and local API style.\n"
            "4. Make the smallest source-code change that fixes the issue; do not edit tests/docs/examples or unrelated behavior.\n"
            "5. If a safe exact edit is not possible in this target file, return {\"old_content\":\"\",\"new_content\":\"\"}.\n"
            "6. The JSON must be complete and parseable; escape newlines as \\n and quotes as \\\".\n"
        )
        if strict_note:
            prompt += (
                "\nRetry note:\n"
                f"{strict_note}\n"
                "Return only one short, complete JSON object. Use a smaller exact anchor if needed.\n"
            )
        raw = self.m_exec.execute(
            instruction=prompt, context="", task_type="code_generation",
            max_tokens=4096,
        )


        def _looks_like_edit_object(text: str) -> bool:
            return (
                ('"old_content"' in text or "'old_content'" in text)
                and ('"new_content"' in text or "'new_content'" in text)
            )

        def _balanced_json_candidates(text: str) -> List[str]:
            out: List[str] = []
            start: Optional[int] = None
            depth = 0
            in_str = False
            quote = ""
            esc = False
            for i, ch in enumerate(text):
                if start is None:
                    if ch == "{":
                        start = i
                        depth = 1
                        in_str = False
                        quote = ""
                        esc = False
                    continue
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == quote:
                        in_str = False
                    continue
                if ch in ("'", '"'):
                    in_str = True
                    quote = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = text[start:i + 1]
                        if _looks_like_edit_object(cand):
                            out.append(cand)
                        start = None
            return out

        candidates: List[str] = []
        for m in _re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", raw):
            fenced = m.group(1).strip()
            if fenced:
                candidates.extend(_balanced_json_candidates(fenced))
                if fenced.startswith("{") and _looks_like_edit_object(fenced):
                    candidates.append(fenced)
        candidates.extend(_balanced_json_candidates(raw))
        m = _re.search(r'\{[\s\S]*(?:"old_content"|\'old_content\')[\s\S]*(?:"new_content"|\'new_content\')[\s\S]*\}', raw)
        if m:
            candidates.append(m.group(0))

        seen_json = set()
        candidates = [c for c in candidates if not (c in seen_json or seen_json.add(c))]

        def _repair_missing_final_string_quote(text: str) -> Optional[Dict[str, str]]:

            s = (text or "").strip()
            brace = s.find("{")
            if brace > 0:
                s = s[brace:].strip()
            if not (s.startswith("{") and "old_content" in s and "new_content" in s):
                return None
            dec = _json.JSONDecoder()
            m_old = _re.search(r'"old_content"\s*:\s*', s)
            if not m_old:
                return None
            try:
                old_val, old_end = dec.raw_decode(s, m_old.end())
            except Exception:
                return None
            m_new = _re.search(r'"new_content"\s*:\s*', s[old_end:])
            if not m_new:
                return None
            new_start = old_end + m_new.end()
            try:
                new_val, _ = dec.raw_decode(s, new_start)
                if isinstance(old_val, str) and isinstance(new_val, str):
                    return {"old_content": old_val, "new_content": new_val}
            except Exception:
                pass


            if new_start >= len(s) or s[new_start] != '"':
                return None
            tail = s[new_start + 1:].rstrip()
            if not tail.endswith("}"):
                return None
            tail = tail[:-1].rstrip()
            if not tail:
                return None
            try:
                new_val = _json.loads('"' + tail + '"')
            except Exception:
                try:
                    new_val = _json.loads('"' + tail.replace("\n", "\\n") + '"')
                except Exception:
                    return None
            if isinstance(old_val, str) and isinstance(new_val, str):
                return {"old_content": old_val, "new_content": new_val}
            return None

        if not candidates:
            repaired = _repair_missing_final_string_quote(raw)
            if repaired is not None:
                return repaired.get("old_content", ""), repaired.get("new_content", "")
            raw_s = raw.strip()
            if raw_s.startswith("{") and "old_content" in raw_s and "new_content" in raw_s:


                raise ValueError(f"partial JSON object in M_exec output. Raw first 400: {raw[:400]}")
            raise ValueError(f"no JSON found in M_exec output. Raw first 400: {raw[:400]}")
        last_error = None
        obj = None
        for json_str in candidates:
            try:
                obj = _json.loads(json_str)
                break
            except _json.JSONDecodeError as e:
                last_error = e
                try:
                    obj = _json.loads(json_str.replace("\r", ""))
                    break
                except _json.JSONDecodeError as e2:
                    last_error = e2
                    try:
                        lit = _ast.literal_eval(json_str)
                        if isinstance(lit, dict):
                            obj = lit
                            break
                    except Exception as e3:
                        last_error = e3
                    continue
        if obj is None:
            for json_str in candidates + [raw]:
                obj = _repair_missing_final_string_quote(json_str)
                if obj is not None:
                    break
        if obj is None:
            e = last_error
            raise ValueError(f"invalid JSON: {e}. Raw: {candidates[0][:400]}")
        return obj.get("old_content", ""), obj.get("new_content", "")

    def _select_edit_context(
        self,
        path: str,
        instruction: str,
        file_content: str,
        traj: "Trajectory",
        window_before: int = 60,
        window_after: int = 90,
    ) -> str:

        lines = file_content.splitlines()
        total = len(lines)
        if total == 0:
            return file_content


        full_file_max = _coerce_int(os.environ.get("SWE_EDIT_FULL_FILE_MAX_LINES", 180), 180)
        if total <= full_file_max:
            return file_content

        def _window(
            center_start: int,
            center_end: Optional[int] = None,
            before: Optional[int] = None,
            after: Optional[int] = None,
        ) -> str:
            center_end = center_end or center_start
            before = window_before if before is None else before
            after = window_after if after is None else after
            start = max(1, _coerce_int(center_start, 1) - before)
            end = min(total, _coerce_int(center_end, _coerce_int(center_start, 1)) + after)
            excerpt = "\n".join(lines[start - 1:end])
            return (
                f"[Excerpt from {path}, original lines {start}-{end} of {total}; "
                f"line numbers are NOT part of the file]\n"
                f"{excerpt}"
            )

        def _line_for_class_method(class_name: str, method_name: str) -> Optional[int]:

            try:
                import ast as _ast
                tree = _ast.parse(file_content)
            except SyntaxError:
                tree = None
            if tree is not None:
                stack = list(getattr(tree, "body", []) or [])
                while stack:
                    node = stack.pop(0)
                    if isinstance(node, _ast.ClassDef):
                        if node.name == class_name:
                            for child in node.body:
                                if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and child.name == method_name:
                                    return int(child.lineno)
                        stack[0:0] = [c for c in node.body if isinstance(c, _ast.ClassDef)]

            m_cls = re.search(rf"(?m)^([ \t]*)class\s+{re.escape(class_name)}\b", file_content)
            if not m_cls:
                return None
            cls_indent = len(m_cls.group(1).replace("\t", "    "))
            cls_start_line = file_content[:m_cls.start()].count("\n") + 1
            lines_local = file_content.splitlines()
            for i in range(cls_start_line, len(lines_local) + 1):
                line = lines_local[i - 1]
                stripped = line.strip()
                if i > cls_start_line and stripped and not line.startswith((" ", "\t")):
                    break
                indent = len(line) - len(line.lstrip(" \t"))
                if indent > cls_indent and re.match(
                    rf"(?:async\s+def|def)\s+{re.escape(method_name)}\b", stripped
                ):
                    return i
            return None

        def _recent_same_file_window() -> Optional[str]:

            norm_path = path.strip().lstrip("./")
            for t in reversed(getattr(traj, "turns", []) or []):
                if getattr(t, "action_type", "") != "view_file":
                    continue
                raw = getattr(t, "raw_action", {}) or {}
                args = raw.get("tool_args", {}) or {}
                viewed = str(args.get("path", "")).strip().lstrip("./")
                if viewed != norm_path:
                    continue
                obs = self._strip_swe_memory(str(getattr(t, "observation", "") or ""))
                m_obs = re.search(r"\(lines\s+(\d+)\s*-\s*(\d+),\s*total\s+\d+\s+lines\)", obs)
                if m_obs:
                    return _window(int(m_obs.group(1)), int(m_obs.group(2)), before=8, after=20)
                if "start_line" in args or "end_line" in args:
                    try:
                        start = _coerce_int(args.get("start_line", 1), 1)
                        end = _coerce_int(args.get("end_line", start), start)
                        return _window(start, end, before=8, after=20)
                    except Exception:
                        continue
            return None

        def _anchor_window_from_instruction() -> Optional[str]:

            anchors = []
            anchors.extend(re.findall(r"`([^`]{4,120})`", instruction))
            anchors.extend(re.findall(r"'([^']{4,120})'", instruction))
            anchors.extend(re.findall(r'"([^"]{4,120})"', instruction))
            anchors.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", instruction))
            anchors.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b", instruction))

            def _anchor_rank(a: str) -> Tuple[int, int, str]:
                return (
                    1 if ("_" in a or "." in a or re.search(r"[A-Z]", a)) else 0,
                    len(a),
                    a,
                )
            for anchor in sorted(set(anchors), key=_anchor_rank, reverse=True):
                if not anchor or anchor.startswith("["):
                    continue
                pos = file_content.find(anchor)
                if pos >= 0:
                    line_no = file_content[:pos].count("\n") + 1
                    return _window(line_no)
            return None


        m = re.search(
            r"(?:after|before|around|at)?\s*lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?",
            instruction,
            flags=re.I,
        )
        if m:
            try:
                start_line = int(m.group(1))
                end_line = int(m.group(2) or m.group(1))
                return _window(start_line, end_line, before=12, after=35)
            except Exception:
                pass


        dotted_targets = re.findall(
            r"\b([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            instruction,
        )
        seen_dotted = set()
        for cls_name, fn_name in dotted_targets:
            if (cls_name, fn_name) in seen_dotted:
                continue
            seen_dotted.add((cls_name, fn_name))
            line_no = _line_for_class_method(cls_name, fn_name)
            if line_no:
                return _window(line_no)


        anchor_window = _anchor_window_from_instruction()
        if anchor_window:
            return anchor_window


        recent_window = _recent_same_file_window()
        if recent_window:
            return recent_window


        class_names = []
        class_names.extend(re.findall(r"\bclass\s+([A-Z][A-Za-z0-9_]*)\b", instruction))
        class_names.extend(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s+class\b", instruction))

        class_names.extend(re.findall(r"\bin\s+([A-Z][A-Za-z0-9_]*)\b", instruction))
        func_names_for_pair = []
        func_names_for_pair.extend(re.findall(r"\b(?:function|method|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", instruction, flags=re.I))
        func_names_for_pair.extend(re.findall(r"\bin\s+[A-Z][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\b", instruction))
        func_names_for_pair.extend(re.findall(r"\b([a-z_][A-Za-z0-9_]*)\s+(?:function|method)\b", instruction, flags=re.I))
        for cls_name in sorted(set(class_names), key=len, reverse=True):
            for fn_name in sorted(set(func_names_for_pair), key=len, reverse=True):
                line_no = _line_for_class_method(cls_name, fn_name)
                if line_no:
                    return _window(line_no)
        for cls_name in sorted(set(class_names), key=len, reverse=True):
            m_cls = re.search(rf"(?m)^([ \t]*)class\s+{re.escape(cls_name)}\b", file_content)
            if m_cls:
                line_no = file_content[:m_cls.start()].count("\n") + 1
                return _window(line_no)

        func_names = []
        func_names.extend(re.findall(r"\b(?:function|method|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", instruction, flags=re.I))
        func_names.extend(re.findall(r"\bin\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:function|method)\b", instruction, flags=re.I))
        for fn_name in sorted(set(func_names), key=len, reverse=True):
            m_fn = re.search(rf"(?m)^([ \t]*)(?:async\s+def|def)\s+{re.escape(fn_name)}\b", file_content)
            if m_fn:
                line_no = file_content[:m_fn.start()].count("\n") + 1
                return _window(line_no)


        anchor_window = _anchor_window_from_instruction()
        if anchor_window:
            return anchor_window


        recent_window = _recent_same_file_window()
        if recent_window:
            return recent_window

        return file_content

    def _recent_view_context(self, path: str, traj: "Trajectory", max_chars: int = 3500) -> str:

        norm_path = str(path or "").strip().lstrip("./")
        snippets = []
        for t in reversed(getattr(traj, "turns", []) or []):
            action = getattr(t, "action_type", "")
            if action not in {"view_file", "search_code"}:
                continue
            raw = getattr(t, "raw_action", {}) or {}
            args = raw.get("tool_args", {}) or {}
            obs = str(getattr(t, "observation", "") or "")
            if not obs:
                continue
            obs = self._strip_swe_memory(obs)
            if action == "view_file":
                viewed = str(args.get("path", "")).strip().lstrip("./")


                if viewed == norm_path:
                    continue
                snippets.append(f"[view_file {viewed}]\n{obs[:1200]}")
            else:
                query = self._shorten_one_line(args.get("query") or "", 90)
                pattern = self._shorten_one_line(args.get("file_pattern") or "", 70)
                label = f"[search_code query={query!r}" + (f" file_pattern={pattern!r}" if pattern else "") + "]"


                snippets.append(f"{label}\n{obs[:900]}")
            if sum(len(s) for s in snippets) >= max_chars:
                break
        return "\n\n".join(reversed(snippets))[:max_chars]

    def _recent_view_locations(self, traj: "Trajectory", limit: int = 4) -> List[str]:

        locations: List[str] = []
        seen = set()
        for t in reversed(getattr(traj, "turns", []) or []):
            if getattr(t, "action_type", "") != "view_file":
                continue
            raw = getattr(t, "raw_action", {}) or {}
            args = raw.get("tool_args", {}) or {}
            path = str(args.get("path", "")).strip().lstrip("./")
            if not path:
                continue
            start = _coerce_int(args.get("start_line", 1), 1)
            end = _coerce_int(args.get("end_line", start), start)
            loc = f"{path}:{start}-{end}"
            if loc in seen:
                continue
            locations.append(loc)
            seen.add(loc)
            if len(locations) >= limit:
                break
        return list(reversed(locations))

    @staticmethod
    def _strip_swe_memory(observation: str) -> str:

        text = str(observation or "")
        for marker in ("\n\n[SWE_MEMORY]", "\n[SWE_MEMORY]"):
            if marker in text:
                return text.split(marker, 1)[0].rstrip()
        return text

    @staticmethod
    def _cap_swe_observation(observation: str, limit: Optional[int] = None) -> str:

        if limit is None:
            limit = _coerce_int(os.environ.get("SWE_OBS_MAX_CHARS", 3000), 3000)
        text = str(observation or "")
        if limit <= 0 or len(text) <= limit:
            return text
        marker = "\n\n[SWE_MEMORY]"
        memory = ""
        base = text
        if marker in text:
            base, mem = text.split(marker, 1)
            memory = marker + mem
        remaining = max(1200, limit - len(memory) - 180)
        if len(base) <= remaining:
            clipped_base = base
        else:
            head = max(700, int(remaining * 0.68))
            tail = max(350, remaining - head)
            omitted = len(base) - head - tail
            clipped_base = (
                base[:head].rstrip()
                + f"\n\n[OBSERVATION_TRUNCATED: omitted {omitted} chars from the middle of this tool result; "
                "SWE_MEMORY below preserves the episode history.]\n\n"
                + base[-tail:].lstrip()
            )
        clipped = clipped_base + memory
        if len(clipped) > limit + 400:

            clipped = clipped[: limit - 1].rstrip() + "…"
        return clipped

    @staticmethod
    def _shorten_one_line(text: str, limit: int = 140) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _summarize_search_observation(self, observation: str, max_hits: int = 3) -> str:

        obs = self._strip_swe_memory(observation)
        if "[NO_MATCH]" in obs:
            status = "NO_MATCH"
        elif "[ERROR]" in obs:
            status = "ERROR"
        elif "[REPEATED" in obs:
            status = "REPEATED"
        else:
            status = "OK"
        hits = []
        for m in re.finditer(r"(?m)(?:^|\n)\s*(?:\./)?([^:\n]+?\.py)[:\-](\d+)[:\-]\s*(.*)", obs):
            path = m.group(1).lstrip("./")
            line = m.group(2)
            code = self._shorten_one_line(m.group(3), 90)
            hit = f"{path}:{line}"
            if code:
                hit += f" `{code}`"
            if hit not in hits:
                hits.append(hit)
            if len(hits) >= max_hits:
                break
        if hits:
            return f"{status}; hits: " + "; ".join(hits)
        m = re.search(r"No matches for '([^']+)'", obs)
        if m:
            return f"{status}; no returned source hit for `{self._shorten_one_line(m.group(1), 80)}`"
        return self._shorten_one_line(obs, 180) or status

    def _summarize_view_observation(self, observation: str, max_lines: int = 6) -> str:

        obs = self._strip_swe_memory(observation)
        if "[ERROR]" in obs:
            return self._shorten_one_line(obs, 180)
        issue_terms = [k.lower() for k in self._swe_issue_keywords(limit=32) if len(k) >= 3]
        scored: List[Tuple[int, int, str]] = []
        fallback: List[Tuple[int, str]] = []
        for line in obs.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s?(.*)", line)
            if not m:
                continue
            line_no_s, code = m.group(1), m.group(2).rstrip()
            line_no = _coerce_int(line_no_s, 0)
            stripped = code.strip()
            if not stripped:
                continue
            item = f"L{line_no_s}: {self._shorten_one_line(stripped, 105)}"
            if (
                len(fallback) < max_lines
                and not stripped.startswith("#")
            ):
                fallback.append((line_no, item))
            code_l = stripped.lower()
            score = 0


            for term in issue_terms:
                if term in code_l:
                    score += 4
            if re.search(r"\b(raise|return|except|if|elif|else|for|while|try|with)\b", stripped):
                score += 2
            if re.search(r"\w+\s*=", stripped) or re.search(r"\w+\(", stripped):
                score += 1
            if "self." in stripped:
                score += 1
            if stripped.startswith("#"):
                score -= 2
            if score > 0:
                scored.append((-score, line_no, item))
        selected = []
        if scored:
            top = sorted(scored)[:max_lines]
            selected = [item for _, _, item in sorted(top, key=lambda x: x[1])]
        if not selected:
            selected = [item for _, item in fallback[:max_lines]]
        if selected:
            return "observed " + "; ".join(selected)
        first = obs.splitlines()[0] if obs.splitlines() else ""
        return self._shorten_one_line(first, 180)

    def _summarize_edit_observation(self, observation: str) -> str:

        obs = self._strip_swe_memory(observation)
        if "[NO_CHANGE_NET]" in obs:
            lines = []
            for line in obs.splitlines():
                m = re.match(r"\s*(\d+)\s*\|\s?(.*)", line)
                if m:
                    code = m.group(2).strip()
                    if code and not code.startswith("#"):
                        lines.append(f"L{m.group(1)}: {self._shorten_one_line(code, 100)}")
                    if len(lines) >= 3:
                        break
            return (
                "NO_CHANGE_NET; current workspace diff empty after edit; updated source: "
                + ("; ".join(lines) if lines else "snippet returned")
            )
        if "[OK]" in obs or "[WARN]" in obs:
            lines = []
            for line in obs.splitlines():
                m = re.match(r"\s*(\d+)\s*\|\s?(.*)", line)
                if m:
                    code = m.group(2).strip()
                    if code and not code.startswith("#"):
                        lines.append(f"L{m.group(1)}: {self._shorten_one_line(code, 100)}")
                    if len(lines) >= 3:
                        break
            status = "WARN" if "[WARN]" in obs else "OK"
            warn = ""
            if "[WARN]" in obs:
                m = re.search(r"Post-edit source consistency note:\s*(.*)", obs, flags=re.S)
                if m:
                    warn = "; warning: " + self._shorten_one_line(m.group(1), 180)
            return f"{status}; updated source: " + ("; ".join(lines) if lines else "snippet returned") + warn
        if "[LINT_ERROR]" in obs:
            m = re.search(r"(?:syntax error:\n\s*)?([^\n]+ at line \d+)", obs, flags=re.I)
            detail = m.group(1).strip() if m else self._shorten_one_line(obs, 140)
            return f"LINT_ERROR; {detail}; file not modified"
        if "[ERROR]" in obs:
            return "ERROR; " + self._shorten_one_line(obs, 180)
        return self._shorten_one_line(obs, 180)

    def _compress_memory_items(self, items: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

        compressed: List[Tuple[str, str]] = []
        i = 0
        while i < len(items):
            action, item = items[i]
            j = i + 1
            while j < len(items) and items[j] == (action, item):
                j += 1
            count = j - i
            if count > 1:
                compressed.append((action, f"{item} (same exact observation repeated x{count})"))
            else:
                compressed.append((action, item))
            i = j
        return compressed

    def _swe_issue_keywords(self, limit: int = 24) -> List[str]:

        issue = str(getattr(self, "_question", {}).get("question", "") or "")
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue)
        stop = {
            "the", "and", "for", "with", "this", "that", "from", "when", "where",
            "which", "should", "would", "could", "have", "has", "are", "was",
            "were", "not", "but", "issue", "problem", "description", "error",
            "traceback", "expected", "actual", "python", "pytest", "test",
            "tests", "class", "function", "method", "line", "file", "code",
        }
        counts: Dict[str, int] = {}
        original: Dict[str, str] = {}
        for tok in tokens:
            low = tok.lower()
            if low in stop or len(low) < 4:
                continue
            counts[low] = counts.get(low, 0) + 1
            original.setdefault(low, tok)

        ranked = sorted(
            counts,
            key=lambda k: (
                -(counts[k] * 3 + (2 if "_" in k else 0) + (2 if any(c.isupper() for c in original[k]) else 0)),
                k,
            ),
        )
        return [original[k] for k in ranked[:limit]]

    def _swe_issue_source_frames(self, limit: int = 6) -> List[str]:

        issue = str(getattr(self, "_question", {}).get("question", "") or "")
        frames: List[str] = []
        seen = set()
        for m in re.finditer(r"(?<![\w/.-])([A-Za-z0-9_./-]+\.py):(\d+)", issue):
            path = m.group(1).strip().lstrip("./")
            line = _coerce_int(m.group(2), 0)
            if not path or not line:
                continue
            low = path.lower()


            if "/tests/" in low or low.startswith("tests/") or "/test_" in low or low.endswith("_test.py"):
                continue
            loc = f"{path}:{line}"
            if loc in seen:
                continue
            frames.append(loc)
            seen.add(loc)
            if len(frames) >= limit:
                break
        return frames

    def _swe_issue_member_mentions(self, limit: int = 6) -> List[Tuple[str, str]]:

        issue = str(getattr(self, "_question", {}).get("question", "") or "")
        mentions: List[Tuple[str, str]] = []
        seen = set()
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]{1,80})\.([A-Za-z_][A-Za-z0-9_]{1,80})\b", issue):
            cls, member = m.group(1), m.group(2)
            key = (cls, member)
            if key in seen:
                continue
            mentions.append(key)
            seen.add(key)
            if len(mentions) >= limit:
                break
        return mentions

    def _swe_issue_member_state_note(self, memory_items: List[Tuple[str, str]]) -> str:

        mentions = self._swe_issue_member_mentions()
        if not mentions or not memory_items:
            return ""
        blob = "\n".join(item for _action, item in memory_items)
        notes: List[str] = []
        for cls, member in mentions:
            cls_seen = re.search(rf"\bclass\s+{re.escape(cls)}\b|\b{re.escape(cls)}\b", blob)
            member_seen = re.search(rf"\b{re.escape(member)}\b", blob)
            def_no_match = re.search(
                rf"search `(?:def\s+)?{re.escape(member)}`[^`]*(?:in `[^`]+`)? -> NO_MATCH",
                blob,
                flags=re.I,
            )
            placeholder_seen = (
                "placeholder/NotImplemented" in blob
                and re.search(rf"\b{re.escape(member)}\b", blob)
            )
            base_m = re.search(rf"\bclass\s+{re.escape(cls)}\s*\(([^)]{{1,160}})\)", blob)
            class_view_paths = []
            for hit in re.finditer(
                rf"view `([^`]+?\.py)(?::[^`]*)?` -> observed [^`\n]*\bclass\s+{re.escape(cls)}\b",
                blob,
            ):
                loc = hit.group(1).strip().lstrip("./")
                if loc and loc not in class_view_paths:
                    class_view_paths.append(loc)
                if len(class_view_paths) >= 3:
                    break
            member_def_hits = []
            for hit in re.finditer(
                rf"([A-Za-z0-9_./-]+\.py):(\d+)\s+`[^`]*\bdef\s+{re.escape(member)}\b",
                blob,
            ):
                loc = f"{hit.group(1).lstrip('./')}:{hit.group(2)}"
                if loc not in member_def_hits:
                    member_def_hits.append(loc)
                if len(member_def_hits) >= 3:
                    break
            if not (cls_seen and member_seen):
                continue
            pieces = [f"`{cls}.{member}` mentioned in the issue"]
            if def_no_match:
                pieces.append("a same-named definition search in an inspected/narrowed file returned NO_MATCH")
            if placeholder_seen:
                pieces.append("a same-named placeholder/NotImplemented location was visible elsewhere")
            if base_m:
                bases = self._shorten_one_line(base_m.group(1), 120)
                pieces.append(f"visible class bases: {bases}")
            if member_def_hits:
                pieces.append("same-named def hit(s) already visible at " + ", ".join(member_def_hits))
                if class_view_paths and not any(
                    hit_loc.split(":", 1)[0] in class_view_paths
                    for hit_loc in member_def_hits
                ):
                    pieces.append(
                        "visible same-named def hit(s) are outside the viewed issue-mentioned class file/window"
                    )
            if len(pieces) > 1:
                notes.append("; ".join(pieces))
            if len(notes) >= 3:
                break
        if not notes:
            return ""
        return "Issue-mentioned member state from visible evidence: " + " | ".join(notes) + "."

    def _swe_issue_source_overlap_note(
        self,
        texts: List[str],
        limit: int = 8,
    ) -> str:

        issue_keywords = self._swe_issue_keywords(limit=32)
        if not issue_keywords or not texts:
            return ""
        blob_l = "\n".join(str(t or "") for t in texts).lower()
        overlaps: List[str] = []
        for kw in issue_keywords:
            low = kw.lower()
            if len(low) < 4:
                continue
            if re.search(rf"\b{re.escape(low)}\b", blob_l):
                if kw not in overlaps:
                    overlaps.append(kw)
            if len(overlaps) >= limit:
                break
        if len(overlaps) < 2:
            return ""
        return (
            "Issue/source overlap already visible in tool evidence: "
            + ", ".join(overlaps)
            + "."
        )

    def _score_memory_item_for_issue(self, item: str, issue_keywords: List[str]) -> int:
        low = item.lower()
        score = 0
        for kw in issue_keywords:
            k = kw.lower()
            if k and k in low:
                score += 3 + min(3, len(k) // 6)
        if "-> OK; hits:" in item:
            score += 1
        if "view `" in item and "observed " in item:
            score += 2
        if "edit `" in item:
            score += 4
        if "NO_MATCH" in item or "REPEATED" in item:
            score -= 1
        return score

    def _swe_memory_summary(
        self,
        traj: "Trajectory",
        current_action: str,
        current_args: Dict[str, Any],
        current_instruction: str,
        current_observation: str,
        max_turns: int = 6,
        max_evidence: int = 6,
        max_chars: int = 1900,
    ) -> str:

        max_turns = _coerce_int(os.environ.get("SWE_MEMORY_MAX_TURNS"), max_turns)
        max_evidence = _coerce_int(os.environ.get("SWE_MEMORY_MAX_EVIDENCE"), max_evidence)
        max_chars = _coerce_int(os.environ.get("SWE_MEMORY_MAX_CHARS"), max_chars)
        memory_items: List[Tuple[str, str]] = []

        def _turn_item(action: str, args: Dict[str, Any], instr: str, obs: str) -> Optional[str]:
            action = str(action or "")
            if action not in {"search_code", "view_file", "edit_file", "list_files", "str_replace_editor"}:
                return None
            args = args or {}
            obs = self._strip_swe_memory(obs)
            if action == "search_code":
                query = self._shorten_one_line(args.get("query") or instr, 70)
                pattern = self._shorten_one_line(args.get("file_pattern") or "", 45)
                prefix = f"search `{query}`"
                if pattern:
                    prefix += f" in `{pattern}`"
                return f"{prefix} -> {self._summarize_search_observation(obs)}"
            if action == "view_file":
                path = self._shorten_one_line(args.get("path") or instr, 90)
                start = args.get("start_line")
                end = args.get("end_line")
                loc = path
                if start or end:
                    loc += f":{start or '?'}-{end or '?'}"
                return f"view `{loc}` -> {self._summarize_view_observation(obs)}"
            if action == "edit_file":
                path = self._shorten_one_line(args.get("path") or "", 80)
                edit_instr = self._shorten_one_line(args.get("instruction") or instr, 120)
                return f"edit `{path}` request `{edit_instr}` -> {self._summarize_edit_observation(obs)}"
            if action == "list_files":
                return f"list_files -> {self._shorten_one_line(obs, 180)}"
            return f"{action} -> {self._shorten_one_line(obs, 180)}"

        for t in getattr(traj, "turns", []) or []:
            raw = getattr(t, "raw_action", {}) or {}
            args = raw.get("tool_args", {}) or {}
            item = _turn_item(
                getattr(t, "action_type", ""),
                args,
                getattr(t, "instruction", ""),
                getattr(t, "observation", ""),
            )
            if item:
                memory_items.append((str(getattr(t, "action_type", "") or ""), item))

        current_item = _turn_item(
            current_action,
            current_args or {},
            current_instruction or "",
            current_observation or "",
        )
        if current_item:
            memory_items.append((str(current_action or ""), current_item))

        if not memory_items:
            return ""


        evidence_pairs: List[Tuple[str, str]] = []
        seen_evidence = set()

        def _canonical_evidence_key(item: str) -> str:
            key = re.sub(r"\[REPEATED x\d+\]", "[REPEATED]", item)
            key = re.sub(r"\(same exact observation repeated x\d+\)", "", key)


            key = re.sub(r"Cached result (?:excerpt|summary):.*", "Cached result", key)
            return self._shorten_one_line(key, 240)

        for action, item in memory_items:
            if action not in {"search_code", "view_file", "edit_file", "str_replace_editor"}:
                continue
            evidence_key = _canonical_evidence_key(item)
            if evidence_key in seen_evidence:
                continue
            evidence_pairs.append((action, item))
            seen_evidence.add(evidence_key)
        evidence_omitted = 0
        if len(evidence_pairs) > max_evidence:
            issue_keywords = self._swe_issue_keywords()
            scored = [
                (self._score_memory_item_for_issue(item, issue_keywords), idx, action, item)
                for idx, (action, item) in enumerate(evidence_pairs)
            ]
            keep_indices = set()

            for idx in range(min(2, len(evidence_pairs))):
                keep_indices.add(idx)
            for idx in range(max(0, len(evidence_pairs) - 2), len(evidence_pairs)):
                keep_indices.add(idx)
            remaining_slots = max_evidence - len(keep_indices)
            for _score, idx, _action, _item in sorted(scored, key=lambda x: (-x[0], x[1])):
                if remaining_slots <= 0:
                    break
                if idx in keep_indices:
                    continue
                keep_indices.add(idx)
                remaining_slots -= 1
            evidence_omitted = len(evidence_pairs) - len(keep_indices)
            evidence_pairs = [pair for idx, pair in enumerate(evidence_pairs) if idx in keep_indices]

        recent = self._compress_memory_items(memory_items)[-max_turns:]
        recent_omitted = max(0, len(memory_items) - len(recent))

        has_edit = any(
            "edit `" in item and "-> OK;" in item
            for _, item in memory_items
        )
        repeated_tail_count = 1
        repeated_tail_summary = ""
        repeated_prior_evidence: List[str] = []
        if memory_items:
            tail_key = (
                memory_items[-1][0],
                _canonical_evidence_key(memory_items[-1][1]),
            )
            for prev_action, prev_item in reversed(memory_items[:-1]):
                if (prev_action, _canonical_evidence_key(prev_item)) != tail_key:
                    break
                repeated_tail_count += 1
            if repeated_tail_count >= 3:
                repeated_tail_summary = self._shorten_one_line(memory_items[-1][1], 180)


                for prev_action, prev_item in reversed(memory_items[:-repeated_tail_count]):
                    if prev_action not in {"search_code", "view_file", "edit_file", "str_replace_editor"}:
                        continue
                    key = (prev_action, _canonical_evidence_key(prev_item))
                    if key == tail_key:
                        continue
                    item_s = self._shorten_one_line(prev_item, 190)
                    if item_s not in repeated_prior_evidence:
                        repeated_prior_evidence.append(item_s)
                    if len(repeated_prior_evidence) >= 3:
                        break
        remaining = max(0, int(getattr(self, "max_episode_steps", 0) or 0) - self._step)
        actual_diff_nonempty = self._workspace_diff_is_nonempty()
        if actual_diff_nonempty:
            diff_state = "non-empty current workspace diff"
        elif has_edit:
            diff_state = "empty current workspace diff (prior edit(s) left no net source change)"
        else:
            diff_state = "empty so far"
        issue_frames = self._swe_issue_source_frames()
        overlap_note = self._swe_issue_source_overlap_note(
            [item for _action, item in evidence_pairs[-8:]]
            + [item for _action, item in recent[-4:]]
            + [self._strip_swe_memory(current_observation or "")]
        )
        member_state_note = self._swe_issue_member_state_note(memory_items)
        source_tool_calls = sum(
            1 for action, _item in memory_items
            if action in {"search_code", "view_file", "edit_file", "str_replace_editor"}
        )
        search_focus_note = ""
        recent_search_norms: List[str] = []
        for action, item in memory_items[-10:]:
            if action != "search_code":
                continue
            m_q = re.search(r"search `([^`]+)`", item)
            if not m_q:
                continue
            q_norm = re.sub(r"^\s*(?:async\s+def|def)\s+", "", m_q.group(1).strip(), flags=re.I)
            q_norm = re.sub(r"\W+", "_", q_norm.lower()).strip("_")
            if q_norm:
                recent_search_norms.append(q_norm)
        if len(recent_search_norms) >= 4:
            counts: Dict[str, int] = {}
            for q_norm in recent_search_norms:
                counts[q_norm] = counts.get(q_norm, 0) + 1
            focus, focus_count = max(counts.items(), key=lambda kv: kv[1])
            if focus_count >= 4 and not actual_diff_nonempty:
                member_mentions = self._swe_issue_member_mentions()
                misaligned = []
                for cls, member in member_mentions:
                    member_l = member.lower()
                    full_l = f"{cls}.{member}".lower()
                    if member_l not in focus and full_l not in focus:
                        misaligned.append(f"{cls}.{member}")
                if misaligned:
                    search_focus_note = (
                        "Search-focus state: recent source searches repeatedly focus on "
                        f"`{focus}` ({focus_count}/{len(recent_search_norms)} recent search calls) "
                        "while issue-visible member(s) "
                        + ", ".join(misaligned[:4])
                        + " remain in memory; workspace diff is still empty."
                    )

        scroll_scan_note = ""
        tail_view_path = ""
        tail_view_count = 0
        tail_view_lines: List[int] = []


        for action, item in reversed(memory_items):
            if action != "view_file":
                break
            m_loc = re.search(r"view `([^`]+)`", item)
            loc = (m_loc.group(1) if m_loc else "").strip()
            m_path = re.match(r"(.+?\.py)(?::.*)?$", loc)
            path_norm = (m_path.group(1) if m_path else loc).strip()
            if not path_norm:
                break
            if not tail_view_path:
                tail_view_path = path_norm
            elif path_norm != tail_view_path:
                break
            tail_view_count += 1
            for m_line in re.finditer(r"\bL(\d+)\b", item):
                tail_view_lines.append(_coerce_int(m_line.group(1), 0))
        if (
            tail_view_count >= 4
            and tail_view_path
            and not actual_diff_nonempty
            and not has_edit
        ):
            line_note = ""
            good_lines = [ln for ln in tail_view_lines if ln > 0]
            if good_lines:
                line_note = (
                    f"; observed tail line numbers span roughly "
                    f"{min(good_lines)}-{max(good_lines)}"
                )
            scroll_scan_note = (
                "[LOOP_NO_NEW_EVIDENCE] Scroll-scan state: the last "
                f"{tail_view_count} source-tool calls are view_file observations "
                f"within `{tail_view_path}` while the workspace diff is still empty "
                f"and no source edit has been recorded{line_note}. These broad "
                "browsing observations do not erase the earlier issue-linked "
                "evidence kept in SWE_MEMORY."
            )

        edit_oscillation_note = ""
        recent_edit_states: List[Tuple[str, str]] = []
        for action, item in memory_items[-12:]:
            if action != "edit_file":
                continue
            m_loc = re.search(r"edit `([^`]+)`", item)
            edit_path = (m_loc.group(1) if m_loc else "").strip()
            if "NO_CHANGE_NET" in item:
                state = "NO_CHANGE_NET"
            elif "-> OK" in item:
                state = "OK_DIFF"
            elif "-> WARN" in item:
                state = "WARN_DIFF"
            elif "ERROR" in item or "LINT_ERROR" in item:
                state = "ERROR"
            else:
                state = "EDIT"
            recent_edit_states.append((edit_path, state))
        if len(recent_edit_states) >= 4:
            paths = [p for p, _s in recent_edit_states if p]
            common_path = paths[-1] if paths else ""
            if common_path and all((not p or p == common_path) for p, _s in recent_edit_states[-4:]):
                tail_states = [s for _p, s in recent_edit_states[-6:]]
                if "NO_CHANGE_NET" in tail_states and any(s in {"OK_DIFF", "WARN_DIFF"} for s in tail_states):
                    edit_oscillation_note = (
                        "Edit-oscillation state: recent edit_file calls on "
                        f"`{common_path}` have alternated between a pending non-empty diff "
                        "and NO_CHANGE_NET (empty net diff). Current workspace diff state is "
                        f"{diff_state}."
                    )

        lines = [
            "[SWE_MEMORY]",
            (
                f"Tool-call budget: used {self._step}/{getattr(self, 'max_episode_steps', '?')}, "
                f"remaining {remaining}. Workspace diff state: {diff_state}."
            ),
            "Persistent evidence ledger, oldest to newest"
            + (f" (omitted {evidence_omitted} middle item(s))" if evidence_omitted else "")
            + ":",
        ]
        if repeated_tail_summary:
            lines.insert(
                2,
                (
                    f"Repeated-action state: last {repeated_tail_count} source-tool observations "
                    f"have the same tool/query summary: {repeated_tail_summary}"
                ),
            )
            member_mentions = self._swe_issue_member_mentions()
            if member_mentions:
                repeated_l = repeated_tail_summary.lower()
                not_in_tail = [
                    f"{cls}.{member}"
                    for cls, member in member_mentions
                    if member.lower() not in repeated_l and f"{cls}.{member}".lower() not in repeated_l
                ]
                if not_in_tail:
                    lines.insert(
                        3,
                        "Repeated-query/member alignment state: issue-mentioned member(s) "
                        + ", ".join(not_in_tail[:3])
                        + " are not named in the repeated query summary."
                    )
            if repeated_prior_evidence:
                lines.insert(
                    3,
                    "Distinct source evidence before this repeated tail: "
                    + " | ".join(reversed(repeated_prior_evidence)),
                )
        if issue_frames:
            lines.insert(
                2,
                "Issue-visible source frame(s) from the problem text: "
                + ", ".join(issue_frames),
            )
        if overlap_note:
            lines.insert(2, overlap_note)
        if member_state_note:
            lines.insert(2, member_state_note)
        if search_focus_note:
            lines.insert(2, search_focus_note)
        if scroll_scan_note:
            lines.insert(2, scroll_scan_note)
        if edit_oscillation_note:
            lines.insert(2, edit_oscillation_note)
        if not actual_diff_nonempty and not has_edit and source_tool_calls >= 6:
            lines.insert(
                2,
                (
                    f"No source edit has produced a workspace diff after {source_tool_calls} "
                    "source-navigation/tool call(s) in this episode."
                ),
            )
        if not actual_diff_nonempty and remaining <= 3:
            lines.insert(
                2,
                (
                    "Near-budget workspace state: the current workspace diff is empty, "
                    "so no source patch will be submitted if the episode ends now."
                ),
            )
        lines.extend(f"- {item}" for _, item in evidence_pairs)
        lines.append(
            "Recent action log, oldest to newest"
            + (f" (omitted {recent_omitted} older item(s))" if recent_omitted else "")
            + ":"
        )
        lines.extend(f"- {item}" for _, item in recent)
        text = "\n".join(lines)
        if len(text) > max_chars:


            head = max(700, int(max_chars * 0.58))
            tail = max(500, max_chars - head - 120)
            omitted = len(text) - head - tail
            if omitted > 0:
                text = (
                    text[:head].rstrip()
                    + f"\n... [SWE_MEMORY omitted {omitted} chars from middle; recent actions preserved below] ...\n"
                    + text[-tail:].lstrip()
                )
            if len(text) > max_chars + 200:
                text = text[: max_chars - 1].rstrip() + "…"
        return text

    @staticmethod
    def _format_numbered_excerpt_from_content(content: str, center_line: int, radius: int = 5) -> str:

        lines = content.splitlines()
        if not lines:
            return ""
        center = max(1, _coerce_int(center_line, 1))
        start = max(1, center - radius)
        end = min(len(lines), center + radius)
        return "\n".join(
            f"{i:4d} | {lines[i - 1]}"
            for i in range(start, end + 1)
        )

    def _source_context_for_location(
        self,
        path: str,
        line_no: int,
        before: int = 8,
        after: int = 18,
    ) -> str:

        norm = str(path or "").strip().lstrip("./")
        content = ""
        if self._repo_path:
            try:
                full = self._resolve_repo_path(norm)
                if os.path.isfile(full):
                    with open(full, "r", errors="replace") as f:
                        content = f.read()
            except Exception:
                content = ""
        if not content and norm in self._code_workspace:
            content = self._code_workspace.get(norm, "")
        if not content:
            return ""
        lines = content.splitlines()
        total = len(lines)
        center = max(1, _coerce_int(line_no, 1))
        start = max(1, center - before)
        end = min(total, center + after)
        body = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1))
        return f"{norm} (lines {start}-{end} of {total}):\n{body}"

    @staticmethod
    def _python_syntax_error(content: str, path: str) -> Optional[SyntaxError]:

        try:
            compile(content, path, "exec")
            return None
        except SyntaxError as e:
            return e

    @staticmethod
    def _is_same_preexisting_syntax_error(
        before: Optional[SyntaxError],
        after: Optional[SyntaxError],
        original_content: str,
        edit_pos: int,
        old_content: str,
        new_content: str,
    ) -> bool:

        if before is None or after is None:
            return False
        before_text = (before.text or "").strip()
        after_text = (after.text or "").strip()
        before_msg = re.sub(r"\bon line \d+\b", "on line N", before.msg or "")
        after_msg = re.sub(r"\bon line \d+\b", "on line N", after.msg or "")
        if before_msg != after_msg:
            return False
        if before_text and after_text and before_text != after_text:
            return False
        if not before.lineno or not after.lineno:
            return True

        edit_start_line = original_content[:edit_pos].count("\n") + 1
        old_end_line = edit_start_line + old_content.count("\n")
        line_delta = new_content.count("\n") - old_content.count("\n")
        expected_after_line = before.lineno
        if before.lineno > old_end_line:
            expected_after_line += line_delta

        return abs(after.lineno - expected_after_line) <= 2

    def _current_source_diff_summary(self, path: str) -> str:

        try:
            if self._repo_path:
                import subprocess
                full = self._resolve_repo_path(path)
                rel = os.path.relpath(full, self._repo_path) if os.path.isabs(full) else path.lstrip("./")
                num = subprocess.run(
                    ["git", "diff", "--numstat", "--", rel],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                if not num:
                    return "Current diff summary for this file: no git diff for this file."
                parts = num.splitlines()[0].split("\t")
                added = parts[0] if len(parts) >= 2 else "?"
                deleted = parts[1] if len(parts) >= 2 else "?"
                udiff = subprocess.run(
                    ["git", "diff", "--unified=0", "--", rel],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                hunks = len(re.findall(r"^@@", udiff, flags=re.M))
                return f"Current diff summary for this file: +{added} -{deleted} across {hunks or 1} hunk(s)."
            norm = str(path or "").strip()
            original = self._code_workspace_original.get(norm, "")
            current = self._code_workspace.get(norm, "")
            if not original or original == current:
                return "Current diff summary for this file: no workspace diff for this file."
            import difflib
            diff = list(difflib.unified_diff(original.splitlines(), current.splitlines(), n=0))
            added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            deleted = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
            hunks = sum(1 for l in diff if l.startswith("@@"))
            return f"Current diff summary for this file: +{added} -{deleted} across {hunks or 1} hunk(s)."
        except Exception:
            return ""

    def _current_source_diff_excerpt(self, path: str, max_chars: int = 1400) -> str:

        try:
            if self._repo_path:
                import subprocess
                full = self._resolve_repo_path(path)
                rel = os.path.relpath(full, self._repo_path) if os.path.isabs(full) else path.lstrip("./")
                diff = subprocess.run(
                    ["git", "diff", "--unified=2", "--", rel],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            else:
                import difflib
                norm = str(path or "").strip()
                original = self._code_workspace_original.get(norm, "")
                current = self._code_workspace.get(norm, "")
                if not original or original == current:
                    diff = ""
                else:
                    diff = "".join(difflib.unified_diff(
                        original.splitlines(keepends=True),
                        current.splitlines(keepends=True),
                        fromfile=f"a/{norm}",
                        tofile=f"b/{norm}",
                        n=2,
                    )).strip()
            if not diff:
                return ""
            if len(diff) > max_chars:
                head = max(500, int(max_chars * 0.62))
                tail = max(300, max_chars - head - 120)
                omitted = len(diff) - head - tail
                diff = (
                    diff[:head].rstrip()
                    + f"\n... [diff excerpt omitted {omitted} chars] ...\n"
                    + diff[-tail:].lstrip()
                )
            return diff
        except Exception:
            return ""

    def _post_edit_requested_change_note(
        self,
        instruction: str,
        before_content: str,
        after_content: str,
    ) -> str:

        notes = []
        for old, new in self._literal_replacement_candidates_from_instruction(instruction)[:3]:
            if old not in before_content:
                continue
            old_after = old in after_content
            new_after = new in after_content
            if old_after and not new_after:
                notes.append(
                    "requested replacement not reflected: "
                    f"`{self._shorten_one_line(old, 90)}` is still present and "
                    f"`{self._shorten_one_line(new, 90)}` is not visible in the updated source"
                )
            elif not new_after:
                notes.append(
                    "requested new text is not visible after edit: "
                    f"`{self._shorten_one_line(new, 90)}`"
                )
        if not notes:
            return ""
        return " Post-edit requested-change note: " + "; ".join(notes) + "."

    def _workspace_diff_is_nonempty(self) -> bool:

        try:
            if self._repo_path:
                import subprocess
                result = subprocess.run(
                    ["git", "diff", "--quiet", "--", ".", ":(exclude)tests/", ":(exclude)*/tests/",
                     ":(exclude)test_*", ":(exclude)*/test_*"],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 1
            for path, current in (self._code_workspace or {}).items():
                if self._code_workspace_original.get(path, "") != current:
                    return True
            return False
        except Exception:


            return False

    @staticmethod
    def _normalize_membership_literal_order_for_note(text: str) -> str:

        def repl(m: re.Match) -> str:
            vals = re.findall(r"(['\"])(.*?)\1", m.group(1))
            if len(vals) < 2:
                return m.group(0)
            quote = vals[0][0]
            sorted_vals = sorted(v for _q, v in vals)
            return "in [" + ", ".join(f"{quote}{v}{quote}" for v in sorted_vals) + "]"
        return re.sub(r"\bin\s*\[([^\]\n]+)\]", repl, str(text or ""))

    def _net_membership_order_noop_note(self, path: str, current_content: str) -> str:

        try:
            original = ""
            norm_path = str(path or "").strip().lstrip("./")
            if self._repo_path:
                import subprocess
                rel = norm_path
                show = subprocess.run(
                    ["git", "show", f"HEAD:{rel}"],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if show.returncode == 0:
                    original = show.stdout
            else:
                original = self._code_workspace_original.get(norm_path, "")
            if not original or original == current_content:
                return ""
            norm_original = self._normalize_membership_literal_order_for_note(original)
            norm_current = self._normalize_membership_literal_order_for_note(current_content)
            if norm_original == norm_current:
                return (
                    "net workspace diff appears to only reorder quoted literals inside Python "
                    "`in [...]` membership checks compared with the original source; membership "
                    "order is normally a semantic no-op"
                )
        except Exception:
            return ""
        return ""

    def _post_edit_option_swap_state_note(self, path: str, current_content: str) -> str:

        try:
            issue_text = str(getattr(self, "_question", {}).get("question", "") or "")
            issue_l = issue_text.lower()
            if not re.search(r"\b(revers(?:ed|e)?|swapp?ed|opposite|inverted)\b", issue_l):
                return ""
            norm_path = str(path or "").strip().lstrip("./")
            original = ""
            if self._repo_path:
                import subprocess
                show = subprocess.run(
                    ["git", "show", f"HEAD:{norm_path}"],
                    cwd=self._repo_path,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if show.returncode == 0:
                    original = show.stdout
            else:
                original = self._code_workspace_original.get(norm_path, "")
            if not original or not current_content or original == current_content:
                return ""

            def _option_expr_map(text: str) -> Dict[str, str]:
                out: Dict[str, str] = {}
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if not re.search(r"\b(?:if|elif)\b", line):
                        continue
                    if not re.search(
                        r"\b(align|option|mode|style|format|kind|type|orientation)\b",
                        line,
                        flags=re.I,
                    ):
                        continue
                    lits = tuple(
                        m.group(2)
                        for m in re.finditer(r"(['\"])([^'\"]{1,40})\1", line)
                    )
                    if not lits:
                        continue
                    body = "\n".join(lines[i + 1:i + 5])
                    m_assign = re.search(
                        r"(?m)^\s*(offsets|result|value|ret|return_value)\s*=\s*(.+)$",
                        body,
                    )
                    if not m_assign:
                        continue
                    expr = self._shorten_one_line(m_assign.group(0).strip(), 120)
                    for lit in lits:
                        out.setdefault(lit, expr)
                return out

            old_map = _option_expr_map(original)
            cur_map = _option_expr_map(current_content)
            issue_lits = [
                lit for lit in sorted(set(old_map) & set(cur_map))
                if re.search(rf"\b{re.escape(lit.lower())}\b", issue_l)
            ]
            if len(issue_lits) != 2:
                return ""
            lit_a, lit_b = issue_lits
            old_a, old_b = old_map[lit_a], old_map[lit_b]
            cur_a, cur_b = cur_map[lit_a], cur_map[lit_b]
            if cur_a == old_b and cur_b == old_a and (cur_a != old_a or cur_b != old_b):
                return (
                    " Post-edit source state: compared with the original checked-out source, "
                    "the issue-visible option branch mapping is now a direct swap: "
                    f"`{lit_a}` now uses `{cur_a}` (originally `{old_a}`), and "
                    f"`{lit_b}` now uses `{cur_b}` (originally `{old_b}`)."
                )
        except Exception:
            return ""
        return ""

    def _post_edit_source_sanity_note(
        self,
        old_content: str,
        new_content: str,
        traj: "Trajectory",
        full_old_content: Optional[str] = None,
        full_new_content: Optional[str] = None,
        path: Optional[str] = None,
    ) -> str:

        notes: List[str] = []
        try:
            import difflib
            diff_lines = list(difflib.ndiff(old_content.splitlines(), new_content.splitlines()))
            added_lines = [l[2:] for l in diff_lines if l.startswith("+ ")]
            removed_lines = [l[2:] for l in diff_lines if l.startswith("- ")]
            added_text = "\n".join(added_lines)
            removed_text = "\n".join(removed_lines)
        except Exception:
            added_lines = []
            removed_lines = []
            added_text = ""
            removed_text = ""

        if added_text:
            cumulative_source_text = full_new_content or new_content
            pre_full_source_text = full_old_content or old_content
            recent_source_text_parts: List[str] = [old_content, new_content]
            pre_edit_source_text_parts: List[str] = [old_content]
            pre_edit_visible_source_parts: List[str] = [old_content]
            for t in reversed(getattr(traj, "turns", []) or []):
                if getattr(t, "action_type", "") not in {"search_code", "view_file"}:
                    continue
                obs = self._strip_swe_memory(str(getattr(t, "observation", "") or ""))
                if obs:
                    recent_source_text_parts.append(obs[:1600])
                    pre_edit_source_text_parts.append(obs[:1600])
                    if "[NO_MATCH]" not in obs and "[ERROR]" not in obs:
                        pre_edit_visible_source_parts.append(obs[:1600])
                if sum(len(s) for s in recent_source_text_parts) > 9000:
                    break
            recent_source_text = "\n".join(reversed(recent_source_text_parts))
            pre_edit_source_text = "\n".join(reversed(pre_edit_source_text_parts))
            pre_edit_visible_source_text = "\n".join(reversed(pre_edit_visible_source_parts))


            try:
                issue_member_names = {
                    member for _cls, member in self._swe_issue_member_mentions(limit=8)
                }
                if issue_member_names and cumulative_source_text:
                    source_lines = cumulative_source_text.splitlines()
                    added_def_names = []
                    for added_line in added_lines:
                        m_def = re.match(r"^([ \t]*)(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", added_line)
                        if not m_def:
                            continue
                        indent = len(m_def.group(1).replace("\t", "    "))
                        name = m_def.group(2)
                        if name not in issue_member_names:
                            continue
                        if indent <= 4:
                            continue
                        stripped_added = added_line.strip()
                        for idx, src_line in enumerate(source_lines):
                            if src_line.strip() != stripped_added:
                                continue

                            enclosing_def = ""
                            enclosing_class = ""
                            for j in range(idx - 1, -1, -1):
                                prev = source_lines[j]
                                if not prev.strip():
                                    continue
                                prev_indent = len(prev) - len(prev.lstrip(" \t"))
                                prev_indent = len(prev[: len(prev) - len(prev.lstrip(" \t"))].replace("\t", "    "))
                                if prev_indent >= indent:
                                    continue
                                m_prev_def = re.match(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b", prev)
                                m_prev_cls = re.match(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", prev)
                                if m_prev_def and not enclosing_def:
                                    enclosing_def = m_prev_def.group(1)
                                if m_prev_cls:
                                    enclosing_class = m_prev_cls.group(1)
                                    break


                            if enclosing_def:
                                added_def_names.append(f"`{name}` under `{enclosing_def}`")
                            break
                    if added_def_names:
                        notes.append(
                            "added issue-visible method definition appears nested inside an existing function/method "
                            "instead of at class scope: " + ", ".join(added_def_names[:3])
                        )
            except Exception:
                pass


            def _string_memberships(text: str) -> List[List[str]]:
                groups: List[List[str]] = []
                for m in re.finditer(r"\bin\s*\[([^\]\n]+)\]", text):
                    vals = re.findall(r"(['\"])(.*?)\1", m.group(1))
                    if len(vals) >= 2:
                        groups.append([v for _q, v in vals])
                return groups

            old_groups = _string_memberships(old_content)
            new_groups = _string_memberships(new_content)
            for old_g, new_g in zip(old_groups, new_groups):
                if old_g != new_g and sorted(old_g) == sorted(new_g):
                    notes.append(
                        "updated lines appear to only reorder string literals inside a Python `in [...]` "
                        "membership check; membership order is normally semantic no-op"
                    )
                    break


            try:
                issue_text = str(getattr(self, "_question", {}).get("question", "") or "")
                issue_l = issue_text.lower()
                mapping_warnings = []
                for m_map in re.finditer(
                    r"(['\"])([^'\"]{1,40})\1\s*:\s*(['\"])([^'\"]{1,40})\3",
                    added_text,
                ):
                    key_opt, val_opt = m_map.group(2), m_map.group(4)
                    if key_opt == val_opt:
                        continue
                    if not re.search(rf"\b{re.escape(key_opt.lower())}\b", issue_l):
                        continue
                    if re.search(rf"['\"]{re.escape(key_opt)}['\"]", old_content + "\n" + pre_edit_source_text):
                        mapping_warnings.append(f"`{key_opt}` -> `{val_opt}`")
                if mapping_warnings:
                    notes.append(
                        "added code translates issue-visible option literal(s) to different public option strings "
                        "while visible source already branches on the original literal(s): "
                        + ", ".join(mapping_warnings[:4])
                    )
                conditional_literal_swaps = []
                for line in added_lines:
                    if " if " not in line or " else " not in line or "==" not in line:
                        continue
                    literals = [m.group(2) for m in re.finditer(r"(['\"])([^'\"]{1,40})\1", line)]
                    issue_literals = [
                        lit for lit in literals
                        if re.search(rf"\b{re.escape(lit.lower())}\b", issue_l)
                    ]
                    if len(set(issue_literals)) < 2:
                        continue
                    if not re.search(r"\b(align|option|mode|style|format|kind|type)\b", line, flags=re.I):
                        continue
                    conditional_literal_swaps.append(self._shorten_one_line(line.strip(), 140))
                if conditional_literal_swaps:
                    notes.append(
                        "added conditional appears to translate/swap issue-visible option literal values at a call site "
                        "while leaving existing helper/branch semantics otherwise unchanged; current source state should "
                        "be checked against the visible option-branch code before ending: "
                        + "; ".join(conditional_literal_swaps[:2])
                    )

                def _option_branches(text: str) -> List[Tuple[Tuple[str, ...], str]]:
                    branches: List[Tuple[Tuple[str, ...], str]] = []
                    lines = text.splitlines()
                    for i, line in enumerate(lines):
                        if not re.search(r"\b(?:if|elif)\b.*\bin\s*\[", line):
                            continue
                        lits = tuple(m.group(2) for m in re.finditer(r"(['\"])([^'\"]{1,40})\1", line))
                        if len(lits) < 2:
                            continue
                        body = "\n".join(lines[i + 1:i + 5])
                        m_assign = re.search(r"(?m)^\s*(offsets|result|value|ret|return_value)\s*=\s*(.+)$", body)
                        if not m_assign:


                            continue
                        branches.append((lits, self._shorten_one_line(m_assign.group(0).strip(), 140)))
                    return branches

                old_branches = _option_branches(old_content)
                new_branches = _option_branches(new_content)
                issue_lit_set = {
                    lit for lit in re.findall(r"[`'\"]([A-Za-z_][A-Za-z0-9_-]{1,40})[`'\"]", issue_text)
                    if re.search(rf"\b{re.escape(lit.lower())}\b", issue_l)
                }

                for lits, _expr in old_branches + new_branches:
                    for lit in lits:
                        if re.search(rf"\b{re.escape(lit.lower())}\b", issue_l):
                            issue_lit_set.add(lit)


                def _literal_expr_map(branches: List[Tuple[Tuple[str, ...], str]]) -> Dict[str, str]:
                    out: Dict[str, str] = {}
                    for lits, expr in branches:
                        for lit in lits:
                            out.setdefault(lit, expr)
                    return out

                old_lit_expr = _literal_expr_map(old_branches)
                new_lit_expr = _literal_expr_map(new_branches)
                unrelated_expr_changes = []
                for lit in sorted(set(old_lit_expr) & set(new_lit_expr)):
                    if lit in issue_lit_set:
                        continue
                    if old_lit_expr[lit] == new_lit_expr[lit]:
                        continue


                    if not any(
                        l in issue_lit_set
                        for l in set(old_lit_expr) | set(new_lit_expr)
                    ):
                        continue
                    unrelated_expr_changes.append(
                        f"`{lit}`: `{old_lit_expr[lit]}` -> `{new_lit_expr[lit]}`"
                    )
                if unrelated_expr_changes:
                    notes.append(
                        "updated option-branch logic also changes computation for option literal(s) "
                        "not visibly mentioned in the issue: "
                        + "; ".join(unrelated_expr_changes[:4])
                    )

                branch_formula_warnings = []
                for (old_lits, old_expr), (new_lits, new_expr) in zip(old_branches, new_branches):
                    if set(old_lits) != set(new_lits):
                        continue
                    if old_expr == new_expr:
                        continue
                    unrelated = [lit for lit in new_lits if lit not in issue_lit_set]
                    related = [lit for lit in new_lits if lit in issue_lit_set]
                    if unrelated and related:
                        branch_formula_warnings.append(
                            f"branch {list(new_lits)} changed formula `{old_expr}` -> `{new_expr}`"
                        )
                if branch_formula_warnings:
                    notes.append(
                        "updated code changes the computation for branch(es) that mix issue-visible option literals "
                        "with unrelated option literals; this may alter unrelated option behavior: "
                        + "; ".join(branch_formula_warnings[:2])
                    )


                def _option_expr_map_any(text: str) -> Dict[str, str]:
                    out: Dict[str, str] = {}
                    lines = text.splitlines()
                    for i, line in enumerate(lines):
                        if not re.search(r"\b(?:if|elif)\b", line):
                            continue
                        if not re.search(r"\b(align|option|mode|style|format|kind|type|orientation)\b", line, flags=re.I):
                            continue
                        lits = tuple(m.group(2) for m in re.finditer(r"(['\"])([^'\"]{1,40})\1", line))
                        if not lits:
                            continue
                        body = "\n".join(lines[i + 1:i + 5])
                        m_assign = re.search(r"(?m)^\s*(offsets|result|value|ret|return_value)\s*=\s*(.+)$", body)
                        if not m_assign:
                            continue
                        expr = self._shorten_one_line(m_assign.group(0).strip(), 140)
                        for lit in lits:
                            out.setdefault(lit, expr)
                    return out

                if re.search(r"\b(revers(?:ed|e)?|swapp?ed|opposite|inverted)\b", issue_l):
                    old_any = _option_expr_map_any(old_content)
                    new_any = _option_expr_map_any(new_content)
                    issue_option_lits = [
                        lit for lit in sorted(issue_lit_set)
                        if lit in old_any and lit in new_any
                    ]
                    if len(issue_option_lits) >= 2:
                        if len(issue_option_lits) == 2:
                            lit_a, lit_b = issue_option_lits
                            old_a, old_b = old_any.get(lit_a, ""), old_any.get(lit_b, "")
                            new_a, new_b = new_any.get(lit_a, ""), new_any.get(lit_b, "")
                            if old_a and old_b and new_a and new_b and not (
                                new_a == old_b and new_b == old_a
                            ):
                                duplicate_note = ""
                                if new_a == new_b:
                                    duplicate_note = (
                                        " Both issue-visible literals now map to the same visible computation."
                                    )
                                notes.append(
                                    "issue describes reversed/swapped option semantics; source-visible branch "
                                    f"mapping before this edit was `{lit_a}` -> `{old_a}` and `{lit_b}` -> `{old_b}`, "
                                    f"but after this edit it is `{lit_a}` -> `{new_a}` and `{lit_b}` -> `{new_b}`. "
                                    "This current source state is not a direct swap of the two visible option "
                                    f"semantics.{duplicate_note}"
                                )
                        old_issue_exprs = {old_any[lit] for lit in issue_option_lits}
                        invented_issue_exprs = [
                            f"`{lit}` -> `{new_any[lit]}`"
                            for lit in issue_option_lits
                            if new_any[lit] not in old_issue_exprs
                        ]
                        if invented_issue_exprs:
                            notes.append(
                                "issue describes reversed/swapped option semantics, but updated branch logic assigns "
                                "issue-visible option literal(s) to computation(s) not seen in their pre-edit option "
                                "branches; current source state changed formulas instead of only remapping existing "
                                "option semantics: " + "; ".join(invented_issue_exprs[:4])
                            )


                    change_text = added_text + "\n" + removed_text
                    source_has_issue_option_branches = bool(issue_option_lits)
                    touched_issue_literal = any(
                        re.search(rf"['\"]{re.escape(lit)}['\"]|\b{re.escape(lit)}\b", change_text)
                        for lit in issue_option_lits
                    )
                    touched_option_selector = bool(re.search(
                        r"\b(align|option|mode|style|format|kind|type|orientation)\b",
                        change_text,
                        flags=re.I,
                    ))
                    if (
                        source_has_issue_option_branches
                        and not touched_issue_literal
                        and not touched_option_selector
                        and re.search(r"\b(offsets?|descent|height|width|coord|position)\b", change_text, flags=re.I)
                    ):
                        notes.append(
                            "issue describes reversed/swapped option semantics, but the edit changes downstream "
                            "coordinate/offset arithmetic without visibly changing how issue-visible option values "
                            "are selected or branched in the current source"
                        )
            except Exception:
                pass

            no_match_tokens: Dict[str, str] = {}
            no_match_attrs: Dict[str, str] = {}
            stop = {
                "def", "class", "return", "self", "from", "import", "none",
                "true", "false", "with", "file", "line", "lines",
            }
            for t in reversed(getattr(traj, "turns", []) or []):
                if getattr(t, "action_type", "") != "search_code":
                    continue
                raw = getattr(t, "raw_action", {}) or {}
                args = raw.get("tool_args", {}) or {}
                query = str(args.get("query") or "")
                obs = self._strip_swe_memory(str(getattr(t, "observation", "") or ""))
                no_match = "[NO_MATCH]" in obs or ("[REPEATED" in obs and "Cached search status: NO_MATCH" in obs)
                if not no_match:
                    continue
                query_norm = query.replace("\\.", ".")
                for attr in re.findall(r"\.([A-Za-z_][A-Za-z0-9_]{2,})\b", query_norm):
                    if attr.lower() not in stop:
                        no_match_attrs.setdefault(attr, query)
                for dotted in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", query_norm):
                    for attr in dotted.split("."):
                        if attr.lower() not in stop:
                            no_match_attrs.setdefault(attr, query)
                for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query):
                    if tok.lower() in stop:
                        continue
                    no_match_tokens.setdefault(tok, query)
                if len(no_match_tokens) >= 16:
                    break
            risky = []
            for tok, query in no_match_tokens.items():
                if not re.search(rf"\b{re.escape(tok)}\b", added_text):
                    continue
                if re.search(rf"\b{re.escape(tok)}\b", pre_edit_source_text):
                    continue


                if re.search(rf"(?m)^\s*(?:async\s+def|def|class)\s+{re.escape(tok)}\b", added_text):
                    continue
                if (
                    re.search(rf"\.\s*{re.escape(tok)}\s*\(", added_text)
                    or re.search(rf"(?<!def\s)\b{re.escape(tok)}\s*\(", added_text)
                ):
                    risky.append(f"`{tok}` (query `{self._shorten_one_line(query, 60)}`)")
            if risky:
                notes.append(
                    "updated added lines call name(s) that recent search_code marked NO_MATCH: "
                    + ", ".join(risky[:4])
                )

            attr_after_no_match = []
            for m_attr in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", added_text):
                attr = m_attr.group(2)
                query = no_match_attrs.get(attr)
                if not query:
                    continue


                if re.search(rf"\.\s*{re.escape(attr)}\b", pre_edit_visible_source_text):
                    continue
                item = f"`.{attr}` (query `{self._shorten_one_line(query, 60)}`)"
                if item not in attr_after_no_match:
                    attr_after_no_match.append(item)
                if len(attr_after_no_match) >= 4:
                    break
            if attr_after_no_match:
                notes.append(
                    "updated added lines read/call attribute(s) that recent search_code marked NO_MATCH: "
                    + ", ".join(attr_after_no_match)
                )

            no_match_related_names = []
            added_names = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", added_text))
            for tok, query in no_match_tokens.items():
                tok_l = tok.lower()
                if len(tok_l) < 4:
                    continue
                if re.search(rf"\b{re.escape(tok)}\b", pre_edit_visible_source_text):
                    continue
                for name in sorted(added_names):
                    name_l = name.lower()
                    if name_l == tok_l or tok_l not in name_l:
                        continue


                    if re.search(rf"\b{re.escape(name)}\s*\(", added_text) or re.search(
                        rf"\.\s*{re.escape(name)}\b", added_text
                    ):
                        item = f"`{name}` related to NO_MATCH token `{tok}` (query `{self._shorten_one_line(query, 60)}`)"
                        if item not in no_match_related_names:
                            no_match_related_names.append(item)
                        break
                if len(no_match_related_names) >= 4:
                    break
            if no_match_related_names:
                notes.append(
                    "updated added lines introduce call/name(s) related to recent NO_MATCH search token(s): "
                    + ", ".join(no_match_related_names)
                )

            if re.search(r"\bzip\s*\([^)]*\bfillvalue\s*=", added_text, flags=re.S):
                notes.append(
                    "added code passes `fillvalue=` to builtin `zip(...)`; Python's builtin zip does not accept that keyword, so dimension-padding needs an appropriate helper/import pattern"
                )


            try:
                import ast
                import difflib

                old_lines_for_map = (full_old_content or old_content).splitlines()
                new_lines_for_map = (full_new_content or new_content).splitlines()
                changed_new_lines: set[int] = set()
                matcher = difflib.SequenceMatcher(None, old_lines_for_map, new_lines_for_map)
                for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
                    if tag != "equal":
                        changed_new_lines.update(range(j1 + 1, j2 + 1))

                if changed_new_lines:
                    tree = ast.parse(full_new_content or new_content)
                    local_param_order: Dict[str, List[str]] = {}
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            params = [a.arg for a in (list(node.args.posonlyargs) + list(node.args.args))]
                            if params:
                                local_param_order.setdefault(node.name, params)

                    duplicate_arg_notes = []
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        if not isinstance(node.func, ast.Name):
                            continue
                        params = local_param_order.get(node.func.id)
                        if not params:
                            continue
                        call_start = getattr(node, "lineno", 0) or 0
                        call_end = getattr(node, "end_lineno", call_start) or call_start
                        if not changed_new_lines.intersection(range(call_start, call_end + 1)):
                            continue
                        positional_count = len(getattr(node, "args", []) or [])
                        for kw in getattr(node, "keywords", []) or []:
                            if not kw.arg or kw.arg not in params:
                                continue
                            if positional_count > params.index(kw.arg):
                                duplicate_arg_notes.append(
                                    f"`{node.func.id}` receives `{kw.arg}` both positionally and by keyword"
                                )
                                break
                        if len(duplicate_arg_notes) >= 3:
                            break
                    if duplicate_arg_notes:
                        notes.append(
                            "updated local function call has a likely Python call-shape error: "
                            + "; ".join(duplicate_arg_notes)
                        )


                    try:
                        import builtins as _builtins
                        import keyword as _keyword

                        old_identifiers = set(re.findall(
                            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                            full_old_content or old_content,
                        ))
                        added_identifiers = set(re.findall(
                            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                            added_text,
                        ))
                        newly_introduced = added_identifiers - old_identifiers
                        bound_names = set()
                        has_star_import = False
                        for node2 in ast.walk(tree):
                            if isinstance(node2, ast.Import):
                                for alias in node2.names:
                                    bound_names.add((alias.asname or alias.name.split(".", 1)[0]))
                            elif isinstance(node2, ast.ImportFrom):
                                for alias in node2.names:
                                    if alias.name == "*":
                                        has_star_import = True
                                        continue
                                    bound_names.add(alias.asname or alias.name)
                            elif isinstance(node2, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                bound_names.add(node2.name)
                                if isinstance(node2, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    for arg in (
                                        list(node2.args.posonlyargs)
                                        + list(node2.args.args)
                                        + list(node2.args.kwonlyargs)
                                    ):
                                        bound_names.add(arg.arg)
                                    if node2.args.vararg:
                                        bound_names.add(node2.args.vararg.arg)
                                    if node2.args.kwarg:
                                        bound_names.add(node2.args.kwarg.arg)
                            elif isinstance(node2, ast.arg):
                                bound_names.add(node2.arg)
                            elif isinstance(node2, ast.ExceptHandler) and node2.name:
                                bound_names.add(str(node2.name))
                            elif isinstance(node2, ast.Name) and isinstance(node2.ctx, (ast.Store, ast.Del)):
                                bound_names.add(node2.id)

                        builtin_names = set(dir(_builtins))
                        if newly_introduced:
                            missing_names = []
                            if not has_star_import:
                                for node2 in ast.walk(tree):
                                    if not isinstance(node2, ast.Name) or not isinstance(node2.ctx, ast.Load):
                                        continue
                                    name = node2.id
                                    if name not in newly_introduced:
                                        continue
                                    lineno = getattr(node2, "lineno", 0) or 0
                                    if lineno not in changed_new_lines:
                                        continue
                                    if (
                                        name in bound_names
                                        or name in builtin_names
                                        or _keyword.iskeyword(name)
                                        or name in {"None", "True", "False", "__name__"}
                                    ):
                                        continue
                                    if name not in missing_names:
                                        missing_names.append(name)
                                    if len(missing_names) >= 4:
                                        break
                            if missing_names:
                                notes.append(
                                    "updated changed line(s) use newly introduced bare name(s) not imported or defined "
                                    "in the current file: "
                                    + ", ".join(f"`{n}`" for n in missing_names)
                                )


                        original_for_cumulative = ""
                        try:
                            if path and self._repo_path:
                                import subprocess
                                full = self._resolve_repo_path(path)
                                rel = (
                                    os.path.relpath(full, self._repo_path)
                                    if os.path.isabs(full) else str(path).lstrip("./")
                                )
                                original_for_cumulative = subprocess.run(
                                    ["git", "show", f"HEAD:{rel}"],
                                    cwd=self._repo_path,
                                    capture_output=True,
                                    text=True,
                                    timeout=5,
                                ).stdout
                            elif path:
                                original_for_cumulative = self._code_workspace_original.get(
                                    str(path).strip(),
                                    "",
                                )
                        except Exception:
                            original_for_cumulative = ""

                        if original_for_cumulative and cumulative_source_text and not has_star_import:
                            original_lines = original_for_cumulative.splitlines()
                            current_lines = cumulative_source_text.splitlines()
                            cumulative_changed_lines: set[int] = set()
                            matcher2 = difflib.SequenceMatcher(None, original_lines, current_lines)
                            for tag, _i1, _i2, j1, j2 in matcher2.get_opcodes():
                                if tag != "equal":
                                    cumulative_changed_lines.update(range(j1 + 1, j2 + 1))
                            original_identifiers = set(re.findall(
                                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                                original_for_cumulative,
                            ))
                            current_identifiers = set(re.findall(
                                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                                cumulative_source_text,
                            ))
                            cumulative_new_names = current_identifiers - original_identifiers
                            cumulative_missing = []
                            if cumulative_changed_lines and cumulative_new_names:
                                for node2 in ast.walk(tree):
                                    if not isinstance(node2, ast.Name) or not isinstance(node2.ctx, ast.Load):
                                        continue
                                    name = node2.id
                                    if name not in cumulative_new_names:
                                        continue
                                    lineno = getattr(node2, "lineno", 0) or 0
                                    if lineno not in cumulative_changed_lines:
                                        continue
                                    if (
                                        name in bound_names
                                        or name in builtin_names
                                        or _keyword.iskeyword(name)
                                        or name in {"None", "True", "False", "__name__"}
                                    ):
                                        continue
                                    if name not in cumulative_missing:
                                        cumulative_missing.append(name)
                                    if len(cumulative_missing) >= 4:
                                        break
                            if cumulative_missing:
                                notes.append(
                                    "current cumulative source diff still uses newly introduced bare name(s) "
                                    "not imported or defined in the current file: "
                                    + ", ".join(f"`{n}`" for n in cumulative_missing)
                                )

                    except Exception:
                        pass
            except Exception:
                pass

            if (
                "serializer_factory" in new_content
                and re.search(r"serializer_factory\([^)]*\)\.serialize\(\)", new_content)
                and re.search(r"['\"].*?\[\s*['\"]%s['\"]\s*\].*?['\"]\s*%\s*\(", new_content)
            ):
                notes.append(
                    "visible serializer_factory(...).serialize() pattern returns a serialized string representation; "
                    "wrapping the `%s` placeholder in additional quotes inside an index format may double-quote "
                    "or otherwise change the serialized reference"
                )

            issue_l_for_format = str(getattr(self, "_question", {}).get("question", "") or "").lower()
            if (
                re.search(r"\boffset\b|\bget_offset\s*\(", added_text)
                and re.search(r"\blabels?\s*=\s*\[.*(?:offset.*label|label.*offset)", added_text, flags=re.S)
                and re.search(r"\b(offset|large|range|formatter|multiplicative|scale)\b", issue_l_for_format)
            ):
                notes.append(
                    "added code string-concatenates a formatter offset with each label; formatter offsets/scales are display metadata, so visible source/API evidence should support whether the offset belongs in every label or should be handled through the formatter state"
                )
            if (
                re.search(r"\bget_offset\s*\(", added_text)
                and re.search(
                    r"\blabels?\s*=\s*\[[^\n\]]*(?:"
                    r"float\s*\([^)]*\)\s*(?:[*/+\-])\s*offset|"
                    r"offset\s*(?:[*/+\-])\s*float\s*\([^)]*\)"
                    r")",
                    added_text,
                )
                and re.search(r"\b(formatter|offset|large|range|legend|ScalarFormatter)\b", issue_l_for_format, flags=re.I)
            ):
                notes.append(
                    "added code performs arithmetic on formatted label text and `formatter.get_offset()` output; "
                    "`format_ticks(locs)` returns display strings and formatter offsets are display text, so this "
                    "does not use the original numeric locator values and may produce invalid label formatting"
                )
            if (
                re.search(r"\bget_offset\s*\(", added_text)
                and re.search(
                    r"\bloc(?:s|ations?)?\b\s*(?:[*/+\-]?=)\s*[^;\n]*\boffset\b|\boffset\b\s*(?:[*/+\-])\s*\bloc(?:s|ations?)?\b",
                    added_text,
                    flags=re.I,
                )
                and re.search(r"\b(formatter|offset|large|range|legend|ScalarFormatter)\b", issue_l_for_format, flags=re.I)
            ):
                notes.append(
                    "added code changes numeric locator values using `formatter.get_offset()` output; "
                    "locator values are already source numeric coordinates and `get_offset()` is formatter display metadata, "
                    "so visible source/API evidence should support any arithmetic on locs before ending"
                )

            if "memoryview" in added_text and "memoryview" in issue_l_for_format:


                helper_lacks_memoryview = False
                helper_match = re.search(
                    r"(?s)def\s+make_bytes\s*\([^)]*\):(?P<body>.*?)(?=^\s{0,8}(?:def|class)\s|\Z)",
                    cumulative_source_text,
                    flags=re.M,
                )
                if helper_match:
                    body = helper_match.group("body")
                    helper_lacks_memoryview = "memoryview" not in body
                if (
                    helper_lacks_memoryview
                    and re.search(r"not\s+isinstance\s*\([^)]*memoryview", added_text)
                ):
                    notes.append(
                        "added code treats `memoryview` as a non-iterated content value, but visible `make_bytes()` source still lacks a direct `memoryview`/`bytes(value)` branch, so the value may still reach generic string conversion"
                    )


            percent_format_warnings = []
            for line in added_lines:
                m_fmt = re.search(r"(['\"])(.*?)(?<!\\)\1\s*%\s*\((.*)\)", line)
                if not m_fmt:
                    continue
                fmt_body = m_fmt.group(2)
                args_body = m_fmt.group(3).strip()


                fmt_clean = fmt_body.replace("%%", "")
                placeholders = re.findall(
                    r"%(?:\([^)]+\))?[#0 +\\-]?(?:\d+|\*)?(?:\.\d+)?[bcdeEfFgGnosxXrisa]",
                    fmt_clean,
                )
                if not placeholders:
                    continue
                depth = 0
                arg_count = 1 if args_body and args_body != "," else 0
                for ch in args_body:
                    if ch in "([{":
                        depth += 1
                    elif ch in ")]}":
                        depth = max(0, depth - 1)
                    elif ch == "," and depth == 0:
                        arg_count += 1

                if args_body.endswith(",") and arg_count > 1:
                    arg_count -= 1
                if arg_count and len(placeholders) != arg_count:
                    percent_format_warnings.append(
                        f"`{self._shorten_one_line(line.strip(), 120)}` has {len(placeholders)} placeholder(s) but {arg_count} tuple value(s)"
                    )
            if percent_format_warnings:
                notes.append(
                    "old-style `%` string formatting placeholder count does not match visible tuple arity: "
                    + "; ".join(percent_format_warnings[:2])
                )


            try:
                import importlib
                imported_modules: Dict[str, str] = {}
                directly_imported_names = set()
                defined_names = set(re.findall(r"(?m)^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", added_text))
                non_import_added = "\n".join(
                    l for l in added_lines
                    if not l.strip().startswith(("import ", "from "))
                )
                for line in added_lines:
                    stripped = line.strip()
                    m_from = re.match(r"from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+(.+)", stripped)
                    if m_from:
                        for part in m_from.group(2).split(","):
                            name_part = part.strip().split(" as ", 1)[-1].strip()
                            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name_part):
                                directly_imported_names.add(name_part)
                        continue
                    m_imp = re.match(r"import\s+(.+)", stripped)
                    if not m_imp:
                        continue
                    for part in m_imp.group(1).split(","):
                        piece = part.strip()
                        if not piece:
                            continue
                        if " as " in piece:
                            mod_name, alias = [p.strip() for p in piece.split(" as ", 1)]
                        else:
                            mod_name = piece
                            alias = piece.split(".", 1)[0]
                        if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", mod_name) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", alias):
                            imported_modules[alias] = mod_name
                call_names = set(re.findall(r"(?<![\.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(", non_import_added))
                import_style_warnings = []
                for alias, mod_name in imported_modules.items():
                    try:
                        module = importlib.import_module(mod_name)
                    except Exception:
                        continue
                    module_attrs = set(dir(module))
                    for cname in sorted(call_names):
                        if cname in defined_names or cname in directly_imported_names:
                            continue
                        if cname not in module_attrs:
                            continue
                        if re.search(rf"\b{re.escape(alias)}\.{re.escape(cname)}\s*\(", non_import_added):
                            continue
                        import_style_warnings.append(
                            f"`import {mod_name}` added but `{cname}(...)` is called unqualified"
                        )
                        break
                if import_style_warnings:
                    notes.append(
                        "import/call style mismatch in added lines: "
                        + "; ".join(import_style_warnings[:2])
                    )


                try:
                    import builtins as _builtins
                    cumulative_direct_imports = set(directly_imported_names)
                    for m_from_all in re.finditer(
                        r"(?m)^\s*from\s+[A-Za-z_][A-Za-z0-9_\.]*\s+import\s+(.+)",
                        cumulative_source_text,
                    ):
                        for part in m_from_all.group(1).split(","):
                            name_part = part.strip().split(" as ", 1)[-1].strip()
                            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name_part):
                                cumulative_direct_imports.add(name_part)
                    cumulative_defined = set(re.findall(
                        r"(?m)^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                        cumulative_source_text,
                    ))
                    cumulative_calls = set(re.findall(
                        r"(?<![\.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                        cumulative_source_text,
                    ))
                    builtin_names = set(dir(_builtins))
                    cumulative_warnings = []
                    for alias, mod_name in imported_modules.items():
                        try:
                            module = importlib.import_module(mod_name)
                        except Exception:
                            continue
                        module_attrs = set(dir(module))
                        for cname in sorted(cumulative_calls):
                            if (
                                cname in cumulative_direct_imports
                                or cname in cumulative_defined
                                or cname in builtin_names
                                or cname not in module_attrs
                            ):
                                continue
                            if re.search(rf"\b{re.escape(alias)}\.{re.escape(cname)}\s*\(", cumulative_source_text):
                                continue
                            cumulative_warnings.append(
                                f"`import {mod_name}` added while `{cname}(...)` is called unqualified in the current source"
                            )
                            break
                    if cumulative_warnings:
                        notes.append(
                            "cumulative import/call style mismatch in current source: "
                            + "; ".join(cumulative_warnings[:2])
                        )
                except Exception:
                    pass
            except Exception:
                pass


            common_attrs = {
                "append", "extend", "insert", "remove", "pop", "clear",
                "items", "keys", "values", "get", "setdefault", "update",
                "copy", "format", "join", "split", "strip", "rstrip", "lstrip",
                "replace", "startswith", "endswith", "lower", "upper",
                "args", "kwargs", "shape", "dtype", "ndim", "size",
            }
            visible_module_aliases = set()
            for m_imp in re.finditer(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)(?:\s+as\s+([A-Za-z_][A-Za-z0-9_]*))?", pre_edit_source_text + "\n" + added_text):
                visible_module_aliases.add(m_imp.group(2) or m_imp.group(1).split(".", 1)[0])
            for m_from in re.finditer(r"(?m)^\s*from\s+[A-Za-z_][A-Za-z0-9_\.]*\s+import\s+(.+)", pre_edit_source_text + "\n" + added_text):
                for part in m_from.group(1).split(","):
                    visible_module_aliases.add(part.strip().split(" as ", 1)[-1].strip())
            unverified_attrs = []
            non_import_added_for_attrs = "\n".join(
                l for l in added_lines
                if not l.strip().startswith(("import ", "from "))
            )
            for m_attr in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", non_import_added_for_attrs):
                base, attr = m_attr.group(1), m_attr.group(2)
                if attr.startswith("__") or attr in common_attrs or base in visible_module_aliases:
                    continue


                after = added_text[m_attr.end():m_attr.end() + 8]
                if re.match(r"\s*=(?!=)", after):
                    continue
                if (
                    re.search(rf"\.\s*{re.escape(attr)}\b", pre_edit_visible_source_text)
                    or re.search(rf"\bdef\s+{re.escape(attr)}\b", pre_edit_visible_source_text)
                    or re.search(rf"\b{re.escape(attr)}\s*=", pre_edit_visible_source_text)
                ):
                    continue
                item = f"`.{attr}`"
                if item not in unverified_attrs:
                    unverified_attrs.append(item)
                if len(unverified_attrs) >= 4:
                    break
            if unverified_attrs:
                notes.append(
                    "added lines read/call object attribute(s) not seen in visible source/search evidence before the edit: "
                    + ", ".join(unverified_attrs)
                )

            property_call_warnings = []
            property_style_text = recent_source_text + "\n" + pre_full_source_text[:200000]
            for m_call in re.finditer(r"\bself\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", added_text):
                name = m_call.group(1)
                if re.search(rf"\bdef\s+{re.escape(name)}\b", property_style_text):
                    continue
                if re.search(rf"\bself\.{re.escape(name)}\s*=", property_style_text):
                    property_call_warnings.append(f"`self.{name}()` while visible source assigns `self.{name} = ...`")
            if property_call_warnings:
                notes.append(
                    "property/method style mismatch in added lines: "
                    + ", ".join(property_call_warnings[:3])
                )

            keyword_style_warnings = []
            for m_call in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()\n]{0,220})\)", added_text):
                fname, args_text = m_call.group(1), m_call.group(2)
                if "=" in args_text:
                    continue
                kw_hits = re.findall(
                    rf"\b{re.escape(fname)}\s*\([^)]*\b([A-Za-z_][A-Za-z0-9_]*)\s*=",
                    recent_source_text,
                )
                kw_hits = [kw for kw in kw_hits if kw not in {"self", "cls"}]
                if not kw_hits:
                    continue


                if args_text.count(",") >= 1:
                    keyword_style_warnings.append(
                        f"`{fname}(...)` added without keyword(s) seen in source: "
                        + ", ".join(sorted(set(kw_hits))[:3])
                    )
            if keyword_style_warnings:
                notes.append(
                    "keyword-argument style mismatch in added lines: "
                    + "; ".join(keyword_style_warnings[:3])
                )

            issue_l = str(getattr(self, "_question", {}).get("question", "") or "").lower()
            if (
                "preserv" in issue_l
                and re.search(r"\b(if\s+not|skip|without|except|return|pass)\b", added_text, flags=re.I)
                and re.search(r"\b(update_wrapper|wraps|wrapper|decorator|assignment|attribute)\b", issue_l + "\n" + added_text, flags=re.I)
            ):
                notes.append(
                    "issue text asks to preserve wrapper/attribute behavior; added conditional/skip logic may remove existing behavior for the reported case"
                )
            if (
                "preserv" in issue_l
                and re.search(r"\b(wrapper|decorator|assignment|attribute|update_wrapper|wraps)\b", issue_l + "\n" + old_content + "\n" + new_content, flags=re.I)
                and re.search(r"(?m)^\s*update_wrapper\s*\(", old_content)
                and re.search(r"(?s)\bif\b[^\n]{0,160}:\s*\n\s*update_wrapper\s*\(", new_content)
            ):
                notes.append(
                    "issue text asks to preserve wrapper/attribute behavior; the edit changes an unconditional `update_wrapper(...)` preservation call into a conditional call, which may skip preservation for the reported wrapper case"
                )
            if (
                re.search(r"\b(partial|wrapper|decorator|preserv|attribute)\b", issue_l, flags=re.I)
                and re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z_][A-Za-z0-9_]*\.func\b", added_text)
            ):
                notes.append(
                    "issue/source context involves wrapper/decorator/partial behavior; added code unwraps `.func` into the original variable, which may drop wrapper/partial arguments or attributes rather than preserving them"
                )


        issue = str(getattr(self, "_question", {}).get("question", "") or "")
        unbound_vars = set()
        for pat in [
            r"UnboundLocalError:\s*local variable '([A-Za-z_][A-Za-z0-9_]*)' referenced before assignment",
            r"local variable '([A-Za-z_][A-Za-z0-9_]*)' referenced before assignment",
        ]:
            unbound_vars.update(re.findall(pat, issue))
        if unbound_vars and new_content:
            lines = new_content.splitlines()
            for var in sorted(unbound_vars):
                if (
                    re.search(rf"\b{re.escape(var)}\s*=\s*None\b", added_text)
                    and re.search(rf"\b{re.escape(var)}\s+in\s+", new_content)
                ):
                    notes.append(
                        f"issue names unbound local `{var}`; added initialization to None while visible "
                        f"source later uses `{var}` in a membership expression, which may change the "
                        "original membership/boolean semantics"
                    )
                if (
                    re.search(rf"\b{re.escape(var)}\s+in\s+[^\n]+\s+if\s+", added_text)
                    or re.search(rf"=\s*{re.escape(var)}\s+in\s+[^\n]+\s+if\s+", added_text)
                ):
                    notes.append(
                        f"issue names unbound local `{var}`; added code wraps an existing membership "
                        "read in a new conditional, which may alter behavior beyond ensuring the "
                        "variable is assigned"
                    )
                assigns: List[Tuple[int, int]] = []
                warned = False
                for idx, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    indent = len(line) - len(line.lstrip(" "))
                    if re.match(rf"^{re.escape(var)}\s*(?::[^=]+)?=", stripped):
                        assigns.append((idx, indent))
                        continue
                    if not re.search(rf"\b{re.escape(var)}\b", stripped):
                        continue
                    if not assigns:
                        continue
                    if any(a_indent <= indent for _a_idx, a_indent in assigns):
                        continue
                    if any(a_idx < idx and a_indent > indent for a_idx, a_indent in assigns):
                        notes.append(
                            f"issue names unbound local `{var}`; updated replacement still has a later read "
                            "at an outer indentation than the visible assignment"
                        )
                        warned = True
                    if warned:
                        break

        if not notes:
            return ""
        return " Post-edit source consistency note: " + "; ".join(notes[:3]) + "."

    def _handle_edit_file(self, args: Dict, traj: "Trajectory") -> str:

        path = args.get("path", "")
        instruction = args.get("instruction", "").strip()
        if not path:
            return "[edit_file] [ERROR] `path` is required."
        if not instruction:

            legacy_old = args.get("old_content", "")
            legacy_new = args.get("new_content", "")
            if legacy_old or legacy_new:
                return (
                    "[edit_file] [ERROR] This tool now takes `instruction` (natural language). "
                    "Describe the change in words — M_exec will generate the exact edit. "
                    "Example: instruction=\"in except HTTPError block, add TooManyRedirects\"."
                )
            return "[edit_file] [ERROR] `instruction` is required (describe the change in natural language)."


        if self._task_type == "code_generation":
            path_lower = path.lower()
            if '/tests/' in path_lower or '/test_' in path_lower or path_lower.startswith('tests/'):
                return f"[edit_file] [ERROR] Cannot edit test files. Only edit source code files."


        if self._repo_path:
            clean = path.strip()
            full_path = self._resolve_repo_path(clean)
            if not os.path.isfile(full_path):
                return f"[edit_file] [ERROR] File '{path}' not found. Use search_code or view_file first."
            try:
                with open(full_path, 'r', errors='replace') as f:
                    file_content = f.read()
            except Exception as e:
                return f"[edit_file] [ERROR] Cannot read '{path}': {e}"


            edit_context = self._select_edit_context(path, instruction, file_content, traj)
            extra_context = self._recent_view_context(path, traj)
            literal_edit = False
            literal_pair = self._literal_replacement_from_instruction(
                instruction=instruction,
                file_content=file_content,
                edit_context=edit_context,
            )
            if literal_pair:
                old_content, new_content = literal_pair
                literal_edit = True
            else:
                try:
                    old_content, new_content = self._m_exec_generate_edit(
                        path=path, instruction=instruction, file_content=edit_context,
                        extra_context=extra_context,
                    )
                except Exception as e1:
                    try:
                        old_content, new_content = self._m_exec_generate_edit(
                            path=path,
                            instruction=instruction,
                            file_content=edit_context,
                            extra_context=extra_context,
                            strict_note=(
                                f"The previous edit generation failed with {type(e1).__name__}: {str(e1)[:200]}. "
                                "Do not include prose or markdown. Do not invent file contents. "
                                "Use an exact small anchor copied from File content."
                            ),
                        )
                    except Exception as e2:
                        return (
                            f"[edit_file] [ERROR] M_exec failed to generate valid edit twice: "
                            f"first={type(e1).__name__}: {str(e1)[:180]}; "
                            f"second={type(e2).__name__}: {str(e2)[:180]}\n"
                            f"No file content was modified."
                        )
            if not old_content:
                return (
                    f"[edit_file] [ERROR] M_exec returned empty old_content for: {instruction[:200]}\n"
                    f"No file content was modified."
                )

            def _find_effective_old(candidate: str) -> str:
                effective = candidate
                if candidate in file_content:
                    return effective

                def _norm_ws(s: str) -> str:
                    return "\n".join(" ".join(line.split()) for line in s.split("\n"))
                file_normalized = _norm_ws(file_content)
                old_normalized = _norm_ws(candidate)
                if old_normalized in file_normalized:


                    file_lines = file_content.split("\n")
                    old_lines_stripped = [" ".join(l.split()) for l in candidate.split("\n") if l.strip()]
                    if old_lines_stripped:
                        n_old = len(candidate.split("\n"))
                        for i in range(len(file_lines) - n_old + 1):
                            window = "\n".join(file_lines[i:i+n_old])
                            if _norm_ws(window) == old_normalized:
                                effective = window
                                break
                if effective not in file_content:


                    try:
                        import difflib as _difflib
                        file_lines = file_content.split("\n")
                        cand_lines = candidate.split("\n")
                        cand_norm = _norm_ws(candidate).strip()
                        if cand_norm and 1 <= len(cand_lines) <= 80:
                            best: Tuple[float, str] = (0.0, "")
                            second = 0.0
                            for n_old in sorted(set([
                                len(cand_lines),
                                max(1, len(cand_lines) - 1),
                                len(cand_lines) + 1,
                            ])):
                                if n_old > len(file_lines):
                                    continue
                                for i in range(0, len(file_lines) - n_old + 1):
                                    window = "\n".join(file_lines[i:i + n_old])
                                    ratio = _difflib.SequenceMatcher(
                                        None, cand_norm, _norm_ws(window).strip()
                                    ).ratio()
                                    if ratio > best[0]:
                                        second = best[0]
                                        best = (ratio, window)
                                    elif ratio > second:
                                        second = ratio
                            if best[0] >= 0.94 and (best[0] - second) >= 0.03:
                                effective = best[1]
                    except Exception:
                        pass
                return effective if effective in file_content else ""

            effective_old = _find_effective_old(old_content)
            if not effective_old:


                try:
                    old_content, new_content = self._m_exec_generate_edit(
                        path=path,
                        instruction=instruction,
                        file_content=edit_context,
                        extra_context=extra_context,
                        strict_note=(
                            "The previous old_content did not appear in the real file. "
                            "Choose a shorter exact anchor copied verbatim from File content; "
                            "do not use code from memory."
                        ),
                    )
                    effective_old = _find_effective_old(old_content)
                except Exception:
                    effective_old = ""
                if not effective_old:
                    return (
                        f"[edit_file] [ERROR] old_content not found in '{path}' (tried fuzzy whitespace match). "
                        f"No file content was modified."
                    )
            edit_pos = file_content.index(effective_old)
            new_file_content = file_content.replace(effective_old, new_content, 1)
            if new_file_content == file_content:
                return (
                    f"[edit_file] [NO_CHANGE] M_exec generated a replacement that leaves '{path}' unchanged. "
                    "No file content was modified. The requested change may need a more precise source anchor "
                    "or different semantic instruction."
                )
            syntax_note = (
                "exact literal replacement; Python syntax check passed for this file"
                if literal_edit else
                "Python syntax check passed for this file"
            )

            if path.endswith('.py'):
                baseline_syntax = self._python_syntax_error(file_content, path)
                new_syntax = self._python_syntax_error(new_file_content, path)
                if new_syntax and self._is_same_preexisting_syntax_error(
                    baseline_syntax, new_syntax, file_content, edit_pos, effective_old, new_content
                ):
                    syntax_note = (
                        "no new full-file SyntaxError was introduced; the same pre-existing "
                        "syntax issue remains outside the edited region"
                    )
                elif new_syntax:
                    e = new_syntax
                    attempted_excerpt = self._format_numbered_excerpt_from_content(
                        new_file_content, e.lineno or 1, radius=5
                    )


                    retry_error = ""
                    try:
                        retry_old, retry_new = self._m_exec_generate_edit(
                            path=path,
                            instruction=instruction,
                            file_content=edit_context,
                            extra_context=extra_context,
                            strict_note=(
                                f"The previous replacement caused SyntaxError: {e.msg} at line {e.lineno}. "
                                "The attempted result around the error was:\n"
                                f"{attempted_excerpt}\n"
                                "Generate a syntactically valid minimal edit. Keep method/function/class "
                                "indentation consistent; do not insert code inside a docstring, comment, "
                                "or unrelated block. Use an exact anchor from File content."
                            ),
                        )
                        retry_effective_old = _find_effective_old(retry_old)
                        if retry_effective_old:
                            retry_file_content = file_content.replace(retry_effective_old, retry_new, 1)
                            retry_pos = file_content.index(retry_effective_old)
                            retry_syntax = self._python_syntax_error(retry_file_content, path)
                            if retry_syntax is None or self._is_same_preexisting_syntax_error(
                                baseline_syntax, retry_syntax, file_content, retry_pos, retry_effective_old, retry_new
                            ):
                                old_content = retry_old
                                new_content = retry_new
                                effective_old = retry_effective_old
                                edit_pos = retry_pos
                                new_file_content = retry_file_content
                                if retry_syntax is not None:
                                    syntax_note = (
                                        "no new full-file SyntaxError was introduced; the same pre-existing "
                                        "syntax issue remains outside the edited region"
                                    )
                            else:
                                raise retry_syntax
                        else:
                            retry_error = "retry old_content was not found in the real file"
                    except SyntaxError as e2:
                        retry_error = f"retry also caused SyntaxError: {e2.msg} at line {e2.lineno}"
                    except Exception as e2:
                        retry_error = f"retry failed: {type(e2).__name__}: {str(e2)[:160]}"
                    if retry_error:
                        return (
                            f"[edit_file] [LINT_ERROR] Your edit would introduce a syntax error:\n"
                            f"  {e.msg} at line {e.lineno}\n"
                            f"Attempted result around the syntax error (not written to disk):\n"
                            f"{attempted_excerpt}\n"
                            f"The file was NOT modified. Internal syntax-recovery note: {retry_error}."
                        )
            with open(full_path, 'w') as f:
                f.write(new_file_content)
        else:

            if path not in self._code_workspace:
                return f"[edit_file] [ERROR] File '{path}' not found."
            file_content = self._code_workspace[path]
            try:
                edit_context = self._select_edit_context(path, instruction, file_content, traj)
                extra_context = self._recent_view_context(path, traj)
                literal_edit = False
                literal_pair = self._literal_replacement_from_instruction(
                    instruction=instruction,
                    file_content=file_content,
                    edit_context=edit_context,
                )
                if literal_pair:
                    old_content, new_content = literal_pair
                    literal_edit = True
                else:
                    old_content, new_content = self._m_exec_generate_edit(
                        path=path, instruction=instruction, file_content=edit_context,
                        extra_context=extra_context,
                    )
            except Exception as e:
                return (
                    f"[edit_file] [ERROR] M_exec failed to generate edit: {type(e).__name__}: {str(e)[:300]}"
                )
            if not old_content:
                return f"[edit_file] [ERROR] M_exec returned empty old_content."
            if old_content not in file_content:
                return f"[edit_file] [ERROR] M_exec's old_content not found in '{path}'. Try rephrasing the instruction."
            edit_pos = file_content.index(old_content)
            new_file_content = file_content.replace(old_content, new_content, 1)
            if new_file_content == file_content:
                return (
                    f"[edit_file] [NO_CHANGE] M_exec generated a replacement that leaves '{path}' unchanged. "
                    "No file content was modified. The requested change may need a more precise source anchor "
                    "or different semantic instruction."
                )
            syntax_note = (
                "exact literal replacement; Python syntax check passed for this file"
                if literal_edit else
                "Python syntax check passed for this file"
            )
            if path.endswith('.py'):
                baseline_syntax = self._python_syntax_error(file_content, path)
                new_syntax = self._python_syntax_error(new_file_content, path)
                if new_syntax and self._is_same_preexisting_syntax_error(
                    baseline_syntax, new_syntax, file_content, edit_pos, old_content, new_content
                ):
                    syntax_note = (
                        "no new full-file SyntaxError was introduced; the same pre-existing "
                        "syntax issue remains outside the edited region"
                    )
                elif new_syntax:
                    e = new_syntax
                    attempted_excerpt = self._format_numbered_excerpt_from_content(
                        new_file_content, e.lineno or 1, radius=5
                    )
                    retry_error = ""
                    try:
                        retry_old, retry_new = self._m_exec_generate_edit(
                            path=path,
                            instruction=instruction,
                            file_content=edit_context,
                            extra_context=extra_context,
                            strict_note=(
                                f"The previous replacement caused SyntaxError: {e.msg} at line {e.lineno}. "
                                "The attempted result around the error was:\n"
                                f"{attempted_excerpt}\n"
                                "Generate a syntactically valid minimal edit. Keep indentation consistent; "
                                "do not insert code inside a docstring, comment, or unrelated block."
                            ),
                        )
                        if retry_old in file_content:
                            retry_file_content = file_content.replace(retry_old, retry_new, 1)
                            retry_pos = file_content.index(retry_old)
                            retry_syntax = self._python_syntax_error(retry_file_content, path)
                            if retry_syntax is None or self._is_same_preexisting_syntax_error(
                                baseline_syntax, retry_syntax, file_content, retry_pos, retry_old, retry_new
                            ):
                                old_content = retry_old
                                new_content = retry_new
                                edit_pos = retry_pos
                                new_file_content = retry_file_content
                                if retry_syntax is not None:
                                    syntax_note = (
                                        "no new full-file SyntaxError was introduced; the same pre-existing "
                                        "syntax issue remains outside the edited region"
                                    )
                            else:
                                raise retry_syntax
                        else:
                            retry_error = "retry old_content was not found in the file"
                    except SyntaxError as e2:
                        retry_error = f"retry also caused SyntaxError: {e2.msg} at line {e2.lineno}"
                    except Exception as e2:
                        retry_error = f"retry failed: {type(e2).__name__}: {str(e2)[:160]}"
                    if retry_error:
                        return (
                            f"[edit_file] [LINT_ERROR] M_exec's edit would introduce a syntax error:\n"
                            f"  {e.msg} at line {e.lineno}\n"
                            f"Attempted result around the syntax error (not written):\n"
                            f"{attempted_excerpt}\n"
                            f"The file was NOT modified. Internal syntax-recovery note: {retry_error}."
                        )
            self._code_workspace[path] = new_file_content
            new_file_content = self._code_workspace[path]


        all_lines = new_file_content.split('\n')
        chars_before = new_file_content[:edit_pos].count('\n')
        edit_lines = new_content.count('\n') + 1
        ctx_start = max(0, chars_before - 5)
        ctx_end = min(len(all_lines), chars_before + edit_lines + 5)
        snippet = "\n".join(f"{i+1:4d} | {all_lines[i]}" for i in range(ctx_start, ctx_end))
        above = f"({ctx_start} more lines above)\n" if ctx_start > 0 else ""
        below = f"\n({len(all_lines) - ctx_end} more lines below)" if ctx_end < len(all_lines) else ""
        norm_path = str(path or "").strip().lstrip("./")
        prior_successful_edits = 0
        for t in getattr(traj, "turns", []) or []:
            if getattr(t, "action_type", "") != "edit_file":
                continue
            raw = getattr(t, "raw_action", {}) or {}
            t_args = raw.get("tool_args", {}) or {}
            t_path = str(t_args.get("path", "")).strip().lstrip("./")
            if t_path == norm_path and "[OK]" in self._strip_swe_memory(str(getattr(t, "observation", "") or "")):
                prior_successful_edits += 1
        edit_count_note = (
            f"This is successful edit #{prior_successful_edits + 1} for this file in this episode; "
            "the diff is cumulative. "
        )
        diff_summary = self._current_source_diff_summary(path)
        diff_summary_note = (diff_summary + " ") if diff_summary else ""
        diff_excerpt = self._current_source_diff_excerpt(path, max_chars=900)
        diff_excerpt_note = (
            "Current source diff excerpt for this file:\n"
            f"```diff\n{diff_excerpt}\n```\n\n"
            if diff_excerpt else ""
        )
        workspace_has_diff = self._workspace_diff_is_nonempty()
        if workspace_has_diff:
            status_header = f"[edit_file] [OK] Successfully edited '{path}'."
            workspace_diff_note = (
                "The current workspace diff is non-empty and will be submitted if the episode ends. "
            )
        else:
            status_header = (
                f"[edit_file] [NO_CHANGE_NET] Edited '{path}', but the current source workspace diff is empty."
            )
            workspace_diff_note = (
                "The current workspace diff is empty; if the episode ends now, no source patch will be submitted. "
            )
        sanity_note = self._post_edit_source_sanity_note(
            old_content=effective_old if self._repo_path else old_content,
            new_content=new_content,
            traj=traj,
            full_old_content=file_content,
            full_new_content=new_file_content,
            path=path,
        )
        net_noop_note = self._net_membership_order_noop_note(path, new_file_content)
        if net_noop_note:
            if sanity_note.strip():
                sanity_note = sanity_note.rstrip(".") + "; " + net_noop_note + "."
            else:
                sanity_note = " Post-edit source consistency note: " + net_noop_note + "."
        request_note = self._post_edit_requested_change_note(
            instruction=instruction,
            before_content=file_content,
            after_content=new_file_content,
        )
        if request_note:
            if sanity_note.strip():
                sanity_note = sanity_note.rstrip(".") + ";" + request_note
            else:
                sanity_note = request_note
        option_swap_state_note = self._post_edit_option_swap_state_note(path, new_file_content)
        has_sanity_warning = bool(sanity_note.strip())
        if workspace_has_diff and has_sanity_warning:
            status_header = f"[edit_file] [WARN] Edited '{path}' with source consistency warning(s)."
        early_sanity_note = ""
        if has_sanity_warning:
            early_sanity_note = (
                "[SWE_WARNING] "
                + self._shorten_one_line(sanity_note.strip(), 900)
                + "\n"
            )
        return (
            f"{status_header}\n"
            f"{early_sanity_note}"
            f"{diff_excerpt_note}"
            f"Here is the updated code around your edit (lines {ctx_start+1}-{ctx_end} of {len(all_lines)}):\n"
            f"{above}{snippet}{below}\n\n"
            f"[SWE_NOTE] Source edit applied; {syntax_note}. "
            f"{edit_count_note}{diff_summary_note}"
            f"{workspace_diff_note}"
            "The updated snippet above is the current source state around the edit."
            f"{option_swap_state_note}"
            f"{sanity_note}"
        )

    def _generate_workspace_diff(self) -> str:


        if self._repo_path:
            import subprocess
            try:

                result = subprocess.run(
                    ["git", "diff", "--", ".", ":(exclude)tests/", ":(exclude)*/tests/",
                     ":(exclude)test_*", ":(exclude)*/test_*"],
                    cwd=self._repo_path, capture_output=True, text=True, timeout=10,
                )
                diff = result.stdout
                if diff.strip():
                    if not diff.endswith("\n"):
                        diff = diff + "\n"
                    self._cached_diff = diff
                return self._cached_diff or ""
            except Exception as e:
                logger.warning(f"[SWE] git diff failed: {e}")
                return self._cached_diff or ""


        import difflib
        if not self._code_workspace_original:
            return ""
        diff_parts = []
        for path in sorted(set(list(self._code_workspace.keys()) + list(self._code_workspace_original.keys()))):
            original = self._code_workspace_original.get(path, "")
            modified = self._code_workspace.get(path, "")
            if original == modified:
                continue
            diff = list(difflib.unified_diff(
                original.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            ))
            if diff:
                header = f"diff --git a/{path} b/{path}\n"
                diff_parts.append(header + "".join(diff))
        return "\n".join(diff_parts) if diff_parts else ""

    def _handle_run_tests(self, args: Dict) -> str:

        test_cmd = args.get("test_cmd", "pytest")
        import subprocess


        instance_id = self._extra.get("instance_id", "")
        if instance_id:
            result = self._run_tests_in_swe_env(test_cmd, instance_id)
            if result is not None:
                return result

        if not self._code_workspace and not self._repo_path:
            return "[run_tests] No code workspace loaded for this task."


        import tempfile
        try:
            cwd = self._repo_path if self._repo_path else None
            if not cwd:
                cwd = tempfile.mkdtemp(prefix="skillflow_swe_")
                for path, content in self._code_workspace.items():
                    full_path = os.path.join(cwd, path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, 'w') as f:
                        f.write(content)
            result = subprocess.run(
                test_cmd, shell=True, capture_output=True, text=True,
                timeout=60, cwd=cwd,
            )

            stdout_full = result.stdout.strip()
            stderr_full = result.stderr.strip()
            stdout = stdout_full[-2000:]
            stderr = stderr_full[-1000:]
            clip_note = ""
            if len(stdout_full) > 2000:
                clip_note += f"(stdout {len(stdout_full)}ch, showing last 2000)\n"
            if len(stderr_full) > 1000:
                clip_note += f"(stderr {len(stderr_full)}ch, showing last 1000)\n"
            if result.returncode == 0:
                return f"[run_tests] [PASSED]\n{clip_note}{stdout}"
            return f"[run_tests] [FAILED] (exit {result.returncode})\n{clip_note}[stderr]\n{stderr}\n[stdout]\n{stdout}"
        except subprocess.TimeoutExpired:
            return "[run_tests] Timed out (60s)"
        except Exception as e:
            return f"[run_tests] Error: {e}"

    def _run_tests_in_swe_env(self, test_cmd: str, instance_id: str):

        import subprocess
        try:
            from training.swe_bench_eval import (
                _load_verified_dataset, _verified_cache, _load_specs,
                _swe_bench_specs, _repo_dir, _env_python,
            )
            _load_verified_dataset()
            _load_specs()
            verified = _verified_cache.get(instance_id)
            if not verified:
                return None
            repo = verified["repo"]
            version = verified["version"]
            env_py = _env_python(repo, version)
            repo_path = _repo_dir(repo)
            if not env_py or not repo_path.exists():
                return None


            test_cwd = str(self._repo_path) if self._repo_path else str(repo_path)


            if not self._repo_path:
                base_commit = verified["base_commit"]
                subprocess.run(["git", "checkout", base_commit, "-q"], cwd=test_cwd, capture_output=True, timeout=10)
                subprocess.run(["git", "checkout", ".", "-q"], cwd=test_cwd, capture_output=True, timeout=10)

                for path, content in self._code_workspace.items():
                    full_path = repo_path / path
                    if full_path.exists():
                        full_path.write_text(content)


            spec = _swe_bench_specs.get(repo, {}).get(version, {})
            real_test_cmd = spec.get("test_cmd", "pytest -rA")
            if isinstance(real_test_cmd, list):
                real_test_cmd = real_test_cmd[-1]


            test_arg = test_cmd.strip()
            if "runtests.py" in real_test_cmd:
                cmd = [env_py, "./tests/runtests.py", "--settings=test_sqlite", "--parallel", "1", test_arg]
            else:
                cmd = [env_py, "-m", "pytest", "-x", test_arg]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
                cwd=test_cwd,
            )

            stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
            stderr = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            output = f"{stdout}\n{stderr}".strip()

            if result.returncode == 0:
                return f"[run_tests] [PASSED]\n{output}"
            return f"[run_tests] [FAILED] (exit {result.returncode})\n{output}"

        except subprocess.TimeoutExpired:
            return "[run_tests] Timed out (60s)"
        except Exception as e:
            logger.debug(f"[run_tests] SWE env error: {e}")
            return None
        finally:

            if not self._repo_path:
                try:
                    subprocess.run(["git", "checkout", ".", "-q"], cwd=str(repo_path), capture_output=True, timeout=10)
                    subprocess.run(["git", "clean", "-fd", "-q"], cwd=str(repo_path), capture_output=True, timeout=10)
                except Exception:
                    pass


    def _handle_act(self, args: Dict) -> str:

        action_text = args.get("action", "")
        if not self._ragen_adapter:
            return "[act] [ERROR] No interactive environment loaded."
        try:
            obs, reward, done, info = self._ragen_adapter.step(action_text)
            if done and reward > 0:
                self._env_done = True
                self._env_reward = reward
                return f"[act] [SUCCESS] Task completed! Reward: {reward}\n{obs}"
            elif done:
                self._env_done = True
                self._env_reward = 0.0
                return f"[act] [FAILED] Episode ended without success.\n{obs}"

            if "Nothing happens." in obs:
                return f"[act] [INVALID] Action '{action_text}' had no effect. Choose from the admissible actions.\n{obs}"
            return f"[act] [OK]\n{obs}"
        except Exception as e:
            return f"[act] [ERROR] {e}"

    @staticmethod
    def _clean_webshop_obs(obs: str) -> str:


        if "[SEP] Back to Search" in obs:
            obs = re.sub(r"^.*?\[SEP\]\s*Back to Search\s*\[SEP\]\s*", "Back to Search | ", obs, flags=re.DOTALL)
        elif "Instruction:" in obs:
            obs = re.sub(r"^Instruction:.*?\[SEP\]\s*", "", obs, flags=re.DOTALL)

        obs = obs.replace("[SEP]", "|")
        return obs

    def _handle_search_product(self, args: Dict) -> str:

        query = args.get("query", "")
        if not self._ragen_adapter:
            return "[search_product] [ERROR] WebShop environment not available."
        try:
            obs, reward, done, info = self._ragen_adapter.step(f"search[{query}]")
            obs = self._clean_webshop_obs(obs)
            self._last_webshop_obs = obs  

            if done and reward > 0:
                self._env_done = True
                self._env_reward = reward
                return f"[search_product] [SUCCESS] {obs}"

            if not obs or len(obs.strip()) < 10:
                return f"[search_product] [NO_RESULTS] No products found for '{query}'. Try different keywords."

            return f"[search_product] [OK] Results for '{query}':\n{obs}"
        except Exception as e:
            return f"[search_product] [ERROR] {e}"

    def _handle_click(self, args: Dict) -> str:

        element = args.get("element", "")
        if not self._ragen_adapter:
            return "[click] [ERROR] WebShop environment not available."
        try:
            pre_click_obs = getattr(self, '_last_webshop_obs', '')
            obs, reward, done, info = self._ragen_adapter.step(f"click[{element}]")
            obs = self._clean_webshop_obs(obs)
            self._last_webshop_obs = obs

            if done and reward > 0:
                self._env_done = True
                self._env_reward = reward
                return f"[click] [SUCCESS] Purchase completed! Reward: {reward}\n{obs}"
            elif done:
                self._env_done = True
                self._env_reward = 0.0
                return f"[click] [DONE] Episode ended without purchase.\n{obs}"


            if pre_click_obs and obs == pre_click_obs:
                return (
                    f"[click] [FAILED] Element '{element}' not found on current page. "
                    f"The page did not change. Try a different element from the current page, "
                    f"or use search_product to find the right product.\n"
                    f"Current page:\n{obs}"
                )

            return f"[click] [OK] Page changed.\n{obs}"
        except Exception as e:
            return f"[click] [ERROR] {e}"


    def _handle_skill_invoke(self, args: Dict) -> str:

        skill_id = args.get("skill_id")
        skill = self.workspace.get_by_id(skill_id) if self.workspace else None
        if skill:
            return (
                f"[Strategy: {skill.name}]\n"
                f"{skill.plan}\n"
                f"Pitfall: {skill.pitfall}\n"
                f"Constraint: {skill.constraint}\n\n"
                f"Now follow this strategy step by step using the available tools."
            )
        return f"[ERROR] Skill {skill_id} not found. Use other tools directly."


    def _force_terminate(
        self,
        traj: Trajectory,
    ) -> Tuple[float, bool, Dict]:

        answer = traj.final_answer or ""
        if answer:
            answer = self._clean_accept_answer(answer, traj)


        if self._task_type == "code_generation":
            workspace_diff = self._generate_workspace_diff()
            if workspace_diff:
                answer = workspace_diff
                n_edits = sum(1 for t in traj.turns if t.action_type == "edit_file"
                              and any(tag in (t.observation or "") for tag in ("[OK]", "[WARN]", "[NO_CHANGE_NET]")))
                logger.info(
                    f"[Env] code_generation: generated diff from {n_edits} edit(s), "
                    f"diff_len={len(workspace_diff)}"
                )


        if self._task_type in ("webshop", "alfworld", "interactive_agent") and self._env_done and self._env_reward > 0:

            r_answer = min(float(self._env_reward), 1.0)
            if str(self.reward_mode).lower() in {"outcome_only", "paper", "outcome"}:
                r_process = 0.0
                r_total = r_answer
            else:
                r_process = 0.1
                r_total = r_answer + r_process
            r_tilde = max(r_total + self.epsilon_min, self.epsilon_min)
            r_skill = 0.0
        else:
            r_total, r_answer, r_process, r_skill, r_tilde = compute_full_reward(
                pred=answer,
                gold=self._gold,
                task_type=self._task_type,
                turns=traj.turns,
                extra=self._extra,
                epsilon_min=self.epsilon_min,
                experience_store=self._experience_store,
                reward_mode=self.reward_mode,
            )

        traj.final_answer = answer
        traj.reward = r_total
        traj.answer_reward = r_answer
        traj.r_tilde = r_tilde
        traj.completed = True
        traj.truncated = True

        return r_total, True, {
            "final_answer": answer,
            "truncated": True,
        }


    def _auto_inject_best_skill(
        self, question: Dict, messages: List[Dict], traj: "Trajectory"
    ) -> List[Dict]:

        task_type = self._task_type
        skill_id_for_tracking = "general"

        if self.workspace:
            type_doc = self.workspace.get_type_document(task_type) if hasattr(self.workspace, 'get_type_document') else None
            if type_doc and type_doc.consolidated_strategy and len(type_doc.consolidated_strategy.split()) > 50:
                strategy_text = (
                    f"\n\n## Active Strategy for {task_type}\n"
                    f"{type_doc.consolidated_strategy}\n"
                )
                type_skills = self.workspace.get_skills_by_task_type(task_type)
                skill_id_for_tracking = type_skills[0].meta.skill_id if type_skills else "general"
            elif self.workspace.size > 0:
                q_text = str(question.get("question", ""))
                candidates = self.workspace.retrieve(q_text, task_type=task_type, top_k=1)
                if candidates:
                    skill = candidates[0]
                    strategy_text = (
                        f"\n\n## Active Strategy: [{skill.meta.skill_id}] {skill.name}\n"
                        f"{skill.plan}\n"
                        f"Pitfall: {skill.pitfall}\n"
                        f"Follow this strategy step by step using the available tools.\n"
                    )
                    skill_id_for_tracking = skill.meta.skill_id
                else:
                    from src.skills.format import GENERAL_SKILL
                    skill = GENERAL_SKILL
                    strategy_text = (
                        f"\n\n## Active Strategy: [{skill.meta.skill_id}] {skill.name}\n"
                        f"{skill.plan}\n"
                    )
                    skill_id_for_tracking = "general"
            else:
                from src.skills.format import GENERAL_SKILL
                skill = GENERAL_SKILL
                strategy_text = (
                    f"\n\n## Active Strategy: [{skill.meta.skill_id}] {skill.name}\n"
                    f"{skill.plan}\n"
                )
                skill_id_for_tracking = "general"
        else:
            from src.skills.format import GENERAL_SKILL
            skill = GENERAL_SKILL
            strategy_text = (
                f"\n\n## Active Strategy: [{skill.meta.skill_id}] {skill.name}\n"
                f"{skill.plan}\n"
            )


        turn = Turn(
            supervisor_input="",
            supervisor_output=f"skill_invoke({skill_id_for_tracking})",
            action_type="skill_invoke",
            skill_id=skill_id_for_tracking,
            instruction="auto",
            observation=strategy_text.strip(),
        )
        traj.add_turn(turn)

        return messages


    def _get_injected_tip(self) -> str:

        if hasattr(self, '_injected_skill_ids') and self._injected_skill_ids and self.workspace:
            tips = []
            for sid in self._injected_skill_ids:
                skill = self.workspace.get_by_id(sid)
                if skill and skill.plan:
                    tips.append(skill.plan.strip())
            if tips:
                return "Strategy tip: " + " | ".join(tips)
        return ""

    def _format_context(self) -> str:

        parts = []
        q_text = str(self._question.get("question", ""))
        if q_text:
            parts.append(f"Task ({self._task_type}): {q_text}")

        tip = self._get_injected_tip()
        if tip:
            parts.append(tip)
        if self._context:
            for i, ctx in enumerate(self._context[:5]):
                if isinstance(ctx, dict):
                    text = ctx.get("text", ctx.get("content", str(ctx)))
                else:
                    text = str(ctx)
                parts.append(f"[Doc {i+1}] {text[:1000]}")
        return "\n\n".join(parts)

    def _clean_accept_answer(self, answer: str, traj: Trajectory) -> str:

        if not answer.strip():
            return answer

        stripped = answer.lstrip()

        if stripped.startswith("[PASS]") or stripped.startswith("[FAIL]"):
            for t in reversed(traj.turns):
                obs = t.observation or ""
                if obs and not obs.startswith("[PASS]") and not obs.startswith("[FAIL]"):
                    if self._task_type == "code_generation":
                        blocks = re.findall(r"```(?:python)?\n(.*?)```", obs, re.DOTALL)
                        for b in blocks:
                            if "def " in b:
                                return b.strip()
                    result_m = re.search(r"\[RESULT\]\s*(.+)", obs)
                    if result_m:
                        return result_m.group(1).strip()
            return answer

        if stripped.startswith("[RESULT]"):
            answer = re.sub(r"^\[RESULT\]\s*", "", stripped).strip()
            eq_match = re.search(r"=\s*(-?[\d,.]+)\s*$", answer)
            if eq_match:
                answer = eq_match.group(1).replace(",", "")
            return answer

        if stripped.startswith("[Match"):
            clean = re.sub(r"\[Match \d+\]\s*\(score=[\d.]+\)\s*\S*\n?", "", stripped)
            if clean.strip():
                answer = clean.strip()

        if stripped.startswith("[NO_CONTEXT]"):
            return ""

        return answer

    @staticmethod
    def _extract_code_from_trajectory(traj: Trajectory) -> Optional[str]:

        best_code = ""
        for turn in reversed(traj.turns):
            obs = turn.observation or ""
            blocks = re.findall(r"```(?:python)?\n(.*?)```", obs, re.DOTALL)
            for block in blocks:
                if "def " in block and len(block) > len(best_code):
                    best_code = block.strip()
            inst = turn.instruction or ""
            blocks_inst = re.findall(r"```(?:python)?\n(.*?)```", inst, re.DOTALL)
            for block in blocks_inst:
                if "def " in block and len(block) > len(best_code):
                    best_code = block.strip()
        return best_code if best_code else None

    def _apply_skill_plan(
        self,
        skill_id: Optional[str],
        raw_instruction: str,
        traj: Trajectory,
    ) -> str:

        if not skill_id or not self.workspace:
            return raw_instruction

        skill = self.workspace.get_by_id(skill_id)
        if skill is None:
            return raw_instruction

        enhanced = (
            f"[Context: applying '{skill.name}' strategy]\n"
            f"{raw_instruction}"
        )
        if skill.constraint:
            enhanced += f"\n(Constraint: {skill.constraint})"
        return enhanced


    def _execute_python(self, code: str, timeout: int = 10) -> str:

        if not code.strip():
            return "[ERROR] Empty code"

        import subprocess
        import tempfile
        import os

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir="/tmp"
            ) as f:
                f.write(code)
                tmp_path = f.name


            python_cmd = sys.executable  
            cwd = None
            if self._repo_path and self._task_type == "code_generation":
                cwd = self._repo_path
                instance_id = self._extra.get("instance_id", "")
                if instance_id:
                    try:
                        from training.swe_bench_eval import _load_verified_dataset, _verified_cache, _env_python
                        _load_verified_dataset()
                        verified = _verified_cache.get(instance_id)
                        if verified:
                            env_py = _env_python(verified["repo"], verified["version"])
                            if env_py:
                                python_cmd = str(env_py)
                    except Exception:
                        pass

            result = subprocess.run(
                [python_cmd, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={**os.environ, "PYTHONPATH": cwd or ""},
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                output = stdout if stdout else "[OK] Code executed successfully (no output)"
            else:
                output = f"[ERROR] Exit code {result.returncode}"
                if stderr:
                    output += f"\n{stderr[-500:]}"
                if stdout:
                    output += f"\n[STDOUT]\n{stdout}"

        except subprocess.TimeoutExpired:
            output = f"[ERROR] Execution timed out ({timeout}s)"
        except Exception as e:
            output = f"[ERROR] {type(e).__name__}: {str(e)[:200]}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return output

    def _build_code_gen_prompt(self, tool_type: str, instruction: str,
                               traj: "Trajectory") -> str:

        if tool_type == "test_code":
            test_info = ""
            if self._extra:
                test_str = self._extra.get("test", "")
                entry_point = self._extra.get("entry_point", "")
                if test_str:
                    test_info = f"\n\nTest harness:\n{test_str[:500]}"
                if entry_point:
                    test_info += f"\nEntry point: {entry_point}"

            prev_test_feedback = ""
            for h in reversed(self._history[-3:]):
                obs = h.get("observation", "")
                if h.get("action_type") == "test_code" and ("FAIL" in obs or "[ERROR]" in obs):
                    error_part = ""
                    if "[Test Result]" in obs:
                        error_part = obs[obs.index("[Test Result]"):][:400]
                    elif "[ERROR]" in obs:
                        error_part = obs[obs.index("[ERROR]"):][:400]
                    elif "FAIL" in obs:
                        error_part = obs[obs.index("FAIL"):][:400]

                    code_part = ""
                    if "[Code]" in obs:
                        code_part = obs[obs.index("[Code]"):obs.index("[Code]")+300]

                    prev_test_feedback = (
                        f"\n\n## Previous Test Report (FAILED):\n"
                        f"{error_part}\n\n"
                        f"Previous code (excerpt):\n{code_part}\n\n"
                        f"Fix the bugs based on the error above.\n"
                    )
                    break

            original_question = str(self._question.get("question", ""))[:800]
            return (
                f"## Problem:\n{original_question}\n"
                f"{test_info}\n"
                f"{prev_test_feedback}\n"
                f"## Planning:\n"
                f"Think step-by-step about the algorithm, then write the code.\n\n"
                f"## Code:\n"
                f"Write ONLY the Python function. Output code inside ``` block."
            )

        else:  
            prev_error = ""
            for h in reversed(self._history[-3:]):
                obs = h.get("observation", "")
                if h.get("action_type") == "python_execute" and "[ERROR]" in obs:
                    err_start = obs.index("[ERROR]")
                    error_msg = obs[err_start:err_start+300]
                    code_snippet = ""
                    if "[Code]" in obs:
                        code_snippet = obs[obs.index("[Code]"):obs.index("[Code]")+200]
                    prev_error = (
                        f"\n\nPrevious attempt FAILED:\n{error_msg}\n"
                        f"{code_snippet}\n"
                        f"Fix the error in your new code.\n"
                    )
                    break

            original_question = str(self._question.get("question", ""))[:800]
            hint = instruction[:200] if instruction != original_question else ""
            return (
                f"Write a complete Python script to solve this problem. "
                f"Print the final answer. Output ONLY the code inside a ```python block.\n\n"
                f"For repository/SWE debugging, use small synthetic data and local imports; "
                f"do not rely on internet downloads or external example datasets.\n"
                f"Problem: {original_question}\n"
                f"{f'Hint: {hint}' if hint else ''}\n"
                f"{prev_error}\n"
                f"```python\n"
            )

    @staticmethod
    def _auto_fix_code(code: str) -> str:

        import ast as _ast

        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.rstrip()

            if re.match(r'^(import \w+) as\s*$', stripped):
                module = re.match(r'^import (\w+) as\s*$', stripped).group(1)
                alias_map = {'numpy': 'np', 'sympy': 'sp', 'matplotlib': 'plt', 'pandas': 'pd'}
                alias = alias_map.get(module, module[0])
                fixed_lines.append(f"import {module} as {alias}")
                continue

            if re.match(r'^from \w+ import\s*$', stripped):
                module = re.match(r'^from (\w+) import\s*$', stripped).group(1)
                if module == 'sympy':
                    fixed_lines.append("from sympy import symbols, solve, Rational, sqrt")
                elif module == 'math':
                    fixed_lines.append("from math import gcd, sqrt, factorial")
                else:
                    fixed_lines.append(stripped + " *")
                continue

            fixed_lines.append(line)

        code = '\n'.join(fixed_lines)

        try:
            _ast.parse(code)
        except SyntaxError:
            while lines and code.strip():
                try:
                    _ast.parse(code)
                    break
                except SyntaxError:
                    lines = code.split('\n')[:-1]
                    code = '\n'.join(lines)

        return code

    @staticmethod
    def _sanitize_code(code: str) -> str:

        code = code.replace('\u2011', '-')
        code = code.replace('\u2010', '-')
        code = code.replace('\u2012', '-')
        code = code.replace('\u2013', '-')
        code = code.replace('\u2014', '-')
        code = code.replace('\u2212', '-')
        code = code.replace('\u00a0', ' ')
        code = code.replace('\u202f', ' ')
        code = code.replace('\u2009', ' ')
        code = code.replace('\u2018', "'").replace('\u2019', "'")
        code = code.replace('\u201c', '"').replace('\u201d', '"')
        import re as _re
        code = _re.sub(r'__ +(__\w+__)', r'\1', code)
        code = _re.sub(r'(\w)__ +(\w)', r'\1__\2', code)
        code = _re.sub(r'\bsp ([A-Z])', r'sp.\1', code)
        code = _re.sub(r'\bmath \.', r'math.', code)
        code = _re.sub(r'\bnp ([a-z])', r'np.\1', code)
        return code

    @staticmethod
    def _extract_code_block(text: str) -> str:

        code_block = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if code_block:
            code = code_block.group(1).strip()
            code = GenericTaskEnvironment._sanitize_code(code)
            return GenericTaskEnvironment._auto_fix_code(code)
        if "def " in text:
            def_pos = text.find("def ")
            code = text[def_pos:].strip()
            code = GenericTaskEnvironment._sanitize_code(code)
            return GenericTaskEnvironment._auto_fix_code(code)
        if "import " in text:
            imp_pos = text.find("import ")
            code = text[imp_pos:].strip()
            code = GenericTaskEnvironment._sanitize_code(code)
            return GenericTaskEnvironment._auto_fix_code(code)
        code = GenericTaskEnvironment._sanitize_code(text.strip())
        return GenericTaskEnvironment._auto_fix_code(code)

    def _test_code(self, code: str, test_code: str) -> str:

        if not code.strip():
            return "[ERROR] Empty code"

        if not test_code.strip() and self._extra:
            test_code = self._extra.get("test", "") or ""
            entry_point = self._extra.get("entry_point", "")
            if "def check(" in test_code and entry_point:
                test_code = test_code + f"\ncheck({entry_point})"

        if not test_code.strip():
            return "[ERROR] No test cases provided"

        individual_tests = self._extra.get("test_cases", []) if self._extra else []
        if isinstance(individual_tests, str):
            individual_tests = [t.strip() for t in individual_tests.split('\n') if t.strip()]

        has_assert_tests = (
            individual_tests
            and any(t.strip().startswith('assert ') for t in individual_tests[:3])
        )

        if has_assert_tests:
            return self._run_tests_with_detail(code, individual_tests)

        full_code = code + "\n\n" + test_code
        return self._execute_python(full_code, timeout=15)

    def _run_tests_with_detail(self, code: str, test_cases: list) -> str:

        import subprocess
        import tempfile
        import os

        passed = 0
        total = len(test_cases)
        details = []

        for i, test in enumerate(test_cases[:8], 1):
            test_script = (
                f"{code}\n\n"
                f"# Test case\n"
                f"try:\n"
                f"    {test}\n"
                f"    print('__PASS__')\n"
                f"except AssertionError as e:\n"
                f"    print(f'__FAIL__ {{e}}')\n"
                f"except Exception as e:\n"
                f"    print(f'__ERROR__ {{type(e).__name__}}: {{e}}')\n"
            )
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, dir="/tmp"
                ) as f:
                    f.write(test_script)
                    tmp_path = f.name

                result = subprocess.run(
                    [sys.executable, tmp_path],  
                    capture_output=True, text=True, timeout=10,
                    env={**os.environ, "PYTHONPATH": ""},
                )

                stdout = result.stdout.strip()
                stderr = result.stderr.strip()

                if "__PASS__" in stdout:
                    passed += 1
                    details.append(f"  Test {i}: PASS")
                elif "__FAIL__" in stdout:
                    fail_msg = stdout.split("__FAIL__")[-1].strip()
                    test_preview = test.strip()[:120]
                    detail = f"  Test {i}: FAIL | {test_preview}"
                    if fail_msg:
                        detail += f"\n    Error: {fail_msg[:200]}"
                    if stderr:
                        detail += f"\n    Stderr: {stderr[-200:]}"
                    details.append(detail)
                elif "__ERROR__" in stdout:
                    err_msg = stdout.split("__ERROR__")[-1].strip()[:200]
                    details.append(f"  Test {i}: ERROR | {err_msg}")
                else:
                    err_info = stderr[-300:] if stderr else "Unknown error"
                    details.append(f"  Test {i}: CRASH | {err_info}")

            except subprocess.TimeoutExpired:
                details.append(f"  Test {i}: TIMEOUT (10s)")
            except Exception as e:
                details.append(f"  Test {i}: ERROR | {type(e).__name__}: {str(e)[:100]}")
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        rate = passed / max(total, 1)
        status = "ALL_PASS" if passed == total else ("PARTIAL" if passed > 0 else "ALL_FAIL")
        header = f"[{status}] {passed}/{total} tests passed ({rate:.0%})"

        return header + "\n" + "\n".join(details)


    _embed_model = None

    @classmethod
    def _get_embed_model(cls):

        if cls._embed_model is None:
            try:

                import os
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                os.environ.setdefault("USE_TF", "0")
                from sentence_transformers import SentenceTransformer
                cls._embed_model = SentenceTransformer(
                    "BAAI/bge-base-en-v1.5", device="cpu"
                )
                logger.info("Loaded embedding model for search/fact_verify")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}, falling back to keyword")
                cls._embed_model = "FAILED"
        return cls._embed_model if cls._embed_model != "FAILED" else None

    def _embed_score(self, query: str, passages: list) -> list:

        model = self._get_embed_model()
        if model is None:
            return self._keyword_score(query, passages)

        texts = [query] + passages
        embeddings = model.encode(texts, normalize_embeddings=True)
        scores = (embeddings[0] @ embeddings[1:].T).tolist()
        return scores

    @staticmethod
    def _keyword_score(query: str, passages: list) -> list:

        query_terms = set(query.lower().split()) - {
            "the", "a", "an", "is", "was", "are", "were", "in", "on",
            "at", "to", "for", "of", "and", "or", "that", "this", "it"
        }
        scores = []
        for text in passages:
            hits = sum(1 for t in query_terms if t in text.lower())
            scores.append(hits / max(len(query_terms), 1))
        return scores

    def _get_passage_texts(self) -> list:

        texts = []
        for p in (self._context or []):
            if isinstance(p, dict):
                texts.append(p.get("text", str(p)))
            else:
                texts.append(str(p))
        return texts

    def _verify_fact(self, claim: str) -> str:

        import re
        if not claim.strip():
            return "[ERROR] Empty claim"

        if not self._context:
            return "[NO_CONTEXT] No passages available"

        passages = self._get_passage_texts()
        scores = self._embed_score(claim, passages)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_passage = passages[best_idx]
        best_score = scores[best_idx]


        claim_tokens = re.findall(r'\b[A-Z][\w\'-]+\b|\b\d[\d,.]*\b|"[^"]+"', claim)


        q_text = str(self._question.get("question", "") if hasattr(self, '_question') else "")
        question_tokens = set(re.findall(r'\b[A-Z][\w\'-]+\b', q_text))

        common_caps = {"The", "A", "An", "In", "On", "At", "Of", "To", "For", "By",
                       "And", "Or", "But", "Not", "What", "Who", "Where", "When",
                       "Which", "How", "Why", "Is", "Are", "Was", "Were", "Be",
                       "Based", "Answer", "Passage", "Passages", "Following",
                       "Question", "This", "That", "These", "Those"}
        question_tokens = question_tokens | common_caps

        distinctive = [t for t in claim_tokens if t.strip('"') not in question_tokens]

        if not distinctive:

            evidence = f"[Evidence] (sim={best_score:.2f})\n{best_passage[:400]}"
            if best_score >= 0.55:
                return f"[PARTIAL] (no_distinctive_tokens, sim={best_score:.0%})\n{evidence}"
            return f"[NOT_SUPPORTED] (no_distinctive_tokens, sim={best_score:.0%})\n{evidence}"


        best_coverage = 0.0
        best_evidence_idx = best_idx
        best_found: List[str] = []
        best_not_found: List[str] = list(distinctive)

        for idx, passage in enumerate(passages):
            passage_lower = passage.lower()
            found: List[str] = []
            not_found: List[str] = []
            for t in distinctive:
                clean = t.strip('"').lower()
                if clean in passage_lower:
                    found.append(t)
                else:
                    not_found.append(t)
            coverage = len(found) / max(len(distinctive), 1)
            if coverage > best_coverage:
                best_coverage = coverage
                best_evidence_idx = idx
                best_found = found
                best_not_found = not_found

        evidence_passage = passages[best_evidence_idx]
        evidence = (
            f"[Evidence] (passage {best_evidence_idx}, sim={scores[best_evidence_idx]:.2f})\n"
            f"{evidence_passage[:400]}"
        )


        if best_coverage >= 0.8:
            return (
                f"[SUPPORTED] claim tokens found in passage ({len(best_found)}/{len(distinctive)}): {best_found}\n"
                f"{evidence}"
            )
        elif best_coverage >= 0.4:
            return (
                f"[PARTIAL] partial token coverage: found={best_found}, missing={best_not_found}\n"
                f"{evidence}"
            )
        else:
            return (
                f"[NOT_SUPPORTED] key tokens absent: missing={best_not_found}\n"
                f"{evidence}"
            )

    def _search_passages(self, query: str) -> str:

        if not query.strip():
            return "[ERROR] Empty query"

        if not self._context:

            return self._search_external_corpus(query)

        passages = self._get_passage_texts()
        if not passages:
            return "[NO_CONTEXT] No readable passages"

        bm25_scores = self._bm25_score(query, passages)
        dense_scores = self._embed_score(query, passages)

        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_norm = [s / max(bm25_max, 1e-6) for s in bm25_scores]

        hybrid_scores = [
            0.4 * b + 0.6 * d
            for b, d in zip(bm25_norm, dense_scores)
        ]

        sorted_idx = sorted(range(len(hybrid_scores)),
                            key=lambda i: hybrid_scores[i], reverse=True)

        query_terms = self._extract_query_terms(query)

        previously_seen = set(self._retrieved_passage_ids)
        results = []

        for rank, idx in enumerate(sorted_idx[:3], 1):
            if hybrid_scores[idx] < 0.15:
                continue
            full_text = passages[idx]
            matched_terms = [t for t in query_terms if t.lower() in full_text.lower()]
            match_info = f" keywords={matched_terms}" if matched_terms else ""
            title = self._get_passage_title(idx)


            if matched_terms:
                sents = re.split(r'(?<=[.!?])\s+', full_text)
                keyword_idx = set()
                for i, s in enumerate(sents):
                    if any(t.lower() in s.lower() for t in matched_terms):
                        keyword_idx.add(i)

                expanded = set()
                for i in keyword_idx:
                    expanded.add(i)
                    if i > 0:
                        expanded.add(i - 1)
                    if i < len(sents) - 1:
                        expanded.add(i + 1)
                if expanded:
                    ordered = [sents[i].strip() for i in sorted(expanded)]
                    p_text = " ".join(ordered)[:500]
                else:
                    p_text = full_text[:300]
            else:
                p_text = full_text[:300]

            header = f"[Match {rank}] (score={hybrid_scores[idx]:.2f}{match_info})"
            if title:
                header += f" {title}"
            results.append(f"{header}\n{p_text}")
            self._retrieved_passage_ids.add(idx)

        if not results:
            return f"[search] [NO_MATCH] No relevant passages found for: {query}. Try different keywords or provide your answer based on existing information."


        new_passage_ids = [idx for idx in sorted_idx[:4] if idx not in previously_seen]
        prefix = "[search] [OK]"
        if not new_passage_ids and previously_seen:
            prefix = "[search] [REPEATED] All results were already returned in previous searches. Use 'lookup' to find specific information in existing results, or provide your answer now."

        return prefix + "\n\n---\n\n".join(results)

    @staticmethod
    def _extract_query_terms(query: str) -> list:

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "to", "for", "and", "or", "but", "on", "at",
            "by", "with", "from", "that", "this", "it", "as", "not",
            "what", "who", "where", "when", "which", "how", "does", "did",
            "do", "has", "have", "had", "will", "would", "can", "could",
        }
        tokens = re.findall(r'\b\w+\b', query)
        return [t for t in tokens if t.lower() not in stop_words and len(t) > 1]


    _external_corpus: list = []
    _external_index: dict = {}

    _external_corpus_lock = None

    @classmethod
    def _load_external_corpus(cls):

        if cls._external_index:
            return
        import threading
        if cls._external_corpus_lock is None:
            cls._external_corpus_lock = threading.Lock()
        with cls._external_corpus_lock:

            if cls._external_index:
                return
            import pickle
            corpus_root = os.environ.get("MEDRAG_TEXTBOOKS_DIR", "data/medrag_textbooks")
            index_path = os.path.join(corpus_root, "bm25_index.pkl")
            corpus_path = os.path.join(corpus_root, "all_chunks.jsonl")
            if not os.path.exists(index_path):
                logger.warning(f"[Search] BM25 index not found: {index_path}")
                return
            t0 = time.time()
            with open(index_path, 'rb') as f:
                cls._external_index = pickle.load(f)

            corpus = []
            with open(corpus_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            corpus.append(d.get("contents", d.get("content", "")))
                        except Exception:
                            pass
            cls._external_corpus = corpus
            logger.info(f"[Search] Loaded external corpus: {len(cls._external_corpus)} chunks + index in {time.time()-t0:.1f}s")

    def _search_external_corpus(self, query: str) -> str:

        self._load_external_corpus()
        if not self._external_index:
            return "[NO_CONTEXT] No external knowledge base available"

        import math as _math
        from collections import defaultdict as _ddict

        k1, b = 1.5, 0.75
        idx = self._external_index
        query_terms = re.findall(r'\b\w+\b', query.lower())
        scores = _ddict(float)
        for qt in query_terms:
            if qt not in idx['idf']:
                continue
            term_idf = idx['idf'][qt]
            for doc_id, tf in idx['inverted_index'].get(qt, []):
                dl = idx['doc_lens'][doc_id]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / idx['avg_dl']))
                scores[doc_id] += term_idf * tf_norm

        top = sorted(scores.items(), key=lambda x: -x[1])[:3]

        results = []
        for rank, (doc_id, score) in enumerate(top, 1):
            if score < 1.0:
                continue
            text = self._external_corpus[doc_id] if doc_id < len(self._external_corpus) else ""
            matched = [t for t in query_terms if t in text.lower()]
            results.append(
                f"[Result {rank}] (score={score:.1f}, keywords={matched})\n{text[:500]}"
            )

        if not results:
            return "[search] [NO_RESULTS] No relevant passages found."
        return "[search] [OK]\n" + "\n\n".join(results)

    def _bm25_score(self, query: str, passages: list) -> list:

        import math as _math

        query_terms = self._extract_query_terms(query)
        if not query_terms:
            return [0.0] * len(passages)

        doc_tokens_list = []
        doc_lens = []
        for p in passages:
            tokens = re.findall(r'\b\w+\b', p.lower())
            doc_tokens_list.append(tokens)
            doc_lens.append(len(tokens))

        avg_dl = sum(doc_lens) / max(len(doc_lens), 1)
        n_docs = len(passages)

        from collections import Counter
        df = Counter()
        for tokens in doc_tokens_list:
            df.update(set(tokens))

        k1, b = 1.5, 0.75
        scores = []
        for i, (tokens, dl) in enumerate(zip(doc_tokens_list, doc_lens)):
            tf_map = Counter(tokens)
            score = 0.0
            for qt in query_terms:
                qt_lower = qt.lower()
                tf = tf_map.get(qt_lower, 0)
                if tf == 0:
                    continue
                idf = _math.log(1 + (n_docs - df.get(qt_lower, 0) + 0.5) /
                                (df.get(qt_lower, 0) + 0.5))
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
                score += idf * tf_norm
            scores.append(score)

        return scores

    def _get_passage_title(self, idx: int) -> str:

        if not self._context or idx >= len(self._context):
            return ""
        p = self._context[idx]
        if isinstance(p, dict):
            return p.get("title", "")
        text = str(p)
        m = re.match(r'^\[([^\]]+)\]', text)
        return m.group(1) if m else ""


def _env_flag(name, default=True):

    import os
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() not in {"0", "false", "no", "off"}


def _alfworld_input_fixes_enabled():
    return _env_flag("ALFWORLD_INPUT_FIXES", True)


def _alfworld_decision_block_enabled():


    import os
    if "ALFWORLD_DECISION_BLOCK" in os.environ:
        return _env_flag("ALFWORLD_DECISION_BLOCK", True)
    return False


def _alfworld_semantic_guard_enabled():


    return _env_flag("ALFWORLD_SEMANTIC_GUARD", False)


def _alfworld_progress_block_enabled():


    return _env_flag("ALFWORLD_PROGRESS_BLOCK", False)


def _alfworld_skill_trailer_enabled():


    return _env_flag("ALFWORLD_SKILL_TRAILER", False)


def _alfworld_skill_apply_block_enabled():


    return _env_flag("ALFWORLD_SKILL_APPLY_BLOCK", False)


def _alfworld_strong_guidance_enabled():


    return _env_flag("ALFWORLD_STRONG_GUIDANCE", False)


def _alfworld_invalid_action_feedback_enabled():


    return _env_flag("ALFWORLD_INVALID_ACTION_FEEDBACK", False)


def _alfworld_loop_guard_enabled():


    return _env_flag("ALFWORLD_LOOP_GUARD", False)


def _alfworld_canonicalize_action_enabled():


    return _env_flag("ALFWORLD_CANONICALIZE_ACTION", False)


def _alfworld_env_feedback_enabled():


    return _env_flag("ALFWORLD_ENV_FEEDBACK", True)


def _alfworld_history_max():


    import os
    try:
        return int(os.environ.get("ALFWORLD_HISTORY_MAX", "6"))
    except Exception:
        return 6


def _alfworld_history_obs_chars():


    import os
    try:
        return int(os.environ.get("ALFWORLD_HISTORY_OBS_CHARS", "360"))
    except Exception:
        return 360


def _alfworld_visit_feedback_enabled():


    return _env_flag("ALFWORLD_VISIT_FEEDBACK", False)


def _alfworld_invalid_keep_state_enabled():


    return _env_flag("ALFWORLD_INVALID_KEEP_STATE", True)


def _alfworld_task_brief_enabled():


    return _env_flag("ALFWORLD_TASK_BRIEF", True)


def _alfworld_visible_memory_enabled():


    return _env_flag("ALFWORLD_VISIBLE_MEMORY", True)


_ALFWORLD_SKILL_TRAILER = """Static checklist before choosing (generic, no hidden state):
- Copy exactly one current admissible action string. Never invent a likely location/object/action.
- If the current observation says a container is closed and `open <that container>` is listed, open it before leaving; closed containers are unchecked until opened.
- If an exact `take <task target> ...` action is listed and your hand is free, take it now. Never take sibling/wrong classes.
- If holding the exact target for a plain PLACE task and final `move target to destination` is listed, do it now. If absent, navigate/open toward the destination; do not drop it elsewhere.
- For clean/cool/heat/hot tasks: while holding target, do the required state with sinkbasin/fridge/microwave before final destination move. Do not move target to destination before state is done.
- For count=2: after placing the first target, do not take it back; search for another distinct target and return to the same destination instance.
- For desklamp: hold exact target, then use a listed desklamp action; never place/drop the target.
- If target is not visible/takeable now, prefer listed open actions and listed unvisited/less-recent go actions; avoid checked no-target locations while unchecked choices remain.
- Treat exact class names strictly: pan≠pot/kettle/spatula; peppershaker≠saltshaker; pencil≠pen; soapbar≠soapbottle; cup≠mug/bowl; knife≠butterknife."""


def _build_alfworld_task_brief(task_description: str) -> str:

    if not _alfworld_task_brief_enabled():
        return ""
    parsed = _parse_alfworld_task(task_description or "")
    target = parsed.get("target_class") or ""
    dest = parsed.get("dest_class") or ""
    verb = parsed.get("verb")
    count = int(parsed.get("count") or 1)
    lines = [
        "[ALFWORLD VISIBLE TASK FACTS]",
        "- state summary only from the visible task instruction.",
    ]
    if target:
        lines.append(f"- visible target class phrase: {target}.")
    if dest:
        lines.append(f"- visible destination class phrase: {dest}.")
    if verb in {"clean", "cool", "heat"}:
        lines.append(f"- visible state word in instruction: {verb}.")
    if parsed.get("examine_with_desklamp"):
        lines.append("- visible desklamp wording appears in the instruction.")
    if count > 1:
        lines.append(f"- visible count requirement word/number: {count}.")
    return "\n".join(lines)


def _soften_alfworld_visible_memory_lines(lines):

    if _alfworld_strong_guidance_enabled():
        return lines
    softened = []
    for line in lines:
        s = str(line)
        low = s.lower()
        if "next action should" in low or "recommended next" in low:
            continue

        replacements = [
            ("- listed ways to put down the non-target object:", "- visible put-down actions for the non-target held object:"),
            ("- holding next count target; exact same-destination move currently listed:", "- visible same-destination move actions for the next count target:"),
            ("- holding target and required state is already done; exact final destination move currently listed:", "- visible final destination move actions after required state:"),
            ("- holding target and required state is already done; navigate to the final destination with:", "- visible destination navigation actions after required state:"),
            ("- holding target; required `clean` state action currently listed:", "- visible required `clean` state actions:"),
            ("- holding target; required `cool` state action currently listed:", "- visible required `cool` state actions:"),
            ("- holding target; required `heat` state action currently listed:", "- visible required `heat` state actions:"),
            ("- holding target; required `clean` state is not done, so navigate to the appliance with:", "- visible appliance navigation actions for the `clean` phase:"),
            ("- holding target; required `cool` state is not done, so navigate to the appliance with:", "- visible appliance navigation actions for the `cool` phase:"),
            ("- holding target; required `heat` state is not done, so navigate to the appliance with:", "- visible appliance navigation actions for the `heat` phase:"),
            ("- holding exact target; exact final destination move currently listed:", "- visible final destination move actions:"),
            ("- holding exact target; navigate to the final destination with:", "- visible destination navigation actions:"),
            ("- holding exact target for desklamp task; exact lamp action currently listed:", "- visible desklamp-use actions while holding target:"),
            ("- exact target take actions currently listed:", "- visible exact-target take actions:"),
            ("- count=2 source memory: return to previously target-visible/taken source locations before broad search:", "- count=2 source memory: previously target-visible/taken source locations currently reachable:"),
            ("- SOURCE PRIORITY NOW — choose one of these listed go actions before the raw/unordered navigation list:", "- source-priority listed go actions from visible history/task class:"),
            ("- current listed go targets not yet checked:", "- unchecked listed go targets:"),
        ]
        for old, new in replacements:
            if s.startswith(old):
                s = new + s[len(old):]
                break

        s = s.replace("do NOT choose", "avoid")
        s = s.replace("Do NOT choose", "Avoid")
        s = s.replace("do NOT output", "avoid outputting")
        s = s.replace("Do NOT output", "Avoid outputting")
        s = s.replace("do NOT take back", "already placed progress appears in take actions")
        s = s.replace("do NOT", "avoid")
        s = s.replace("Do NOT", "Avoid")
        s = s.replace("do not choose", "avoid")
        s = s.replace("Do not choose", "Avoid")
        s = s.replace("do not", "avoid")
        s = s.replace("Do not", "Avoid")
        s = s.replace("must", "should")
        s = s.replace("CRITICAL", "visible note")
        s = s.replace("choose SOURCE PRIORITY first", "source-priority actions are also visible")
        s = s.replace("choose a different listed go/open action", "alternative listed go/open actions may be useful")
        s = s.replace("choose one of these listed go actions", "source-priority listed go actions")
        s = s.replace("copy a listed go/open action first, avoid invent the move.", "the exact move is not visible in the current admissible list.")
        s = s.replace("copy one listed go/open action first, avoid invent the move.", "the exact move is not visible in the current admissible list.")
        softened.append(s)
    return softened


def _build_alfworld_visible_memory(env_self, avail_actions) -> str:

    if not _alfworld_visible_memory_enabled():
        return ""

    react_history = list(getattr(env_self, "_react_history", []) or [])
    parsed = _parse_alfworld_task(getattr(env_self, "_task_description", "") or "")
    target = (parsed.get("target_class") or "").lower().strip()
    dest = (parsed.get("dest_class") or "").lower().strip()
    verb = parsed.get("verb")
    count = int(parsed.get("count") or 1)
    avail = [str(a) for a in (avail_actions or []) if str(a) != "help"]
    current_obs = str(getattr(env_self, "_current_obs", "") or "")

    lines = [
        "[ALFWORLD VISIBLE STATE FEEDBACK]",
        "State summary only from the visible task/observation/history/admissible actions; no hidden state, reward signal, candidate ranking, or action recommendation.",
    ]


    obs_one = " ".join(current_obs.split())
    if obs_one:
        lines.append("- current observation excerpt: " + obs_one[:260] + ("..." if len(obs_one) > 260 else ""))


    nav_targets = [a[6:].strip() for a in avail if a.lower().startswith("go to ")]
    open_targets = [a[5:].strip() for a in avail if a.lower().startswith("open ")]
    take_actions = [a for a in avail if a.lower().startswith("take ")]
    move_actions = [a for a in avail if a.lower().startswith("move ")]
    state_actions = [a for a in avail if a.lower().startswith(("clean ", "cool ", "heat "))]
    use_actions = [a for a in avail if a.lower().startswith("use ")]
    cats = []
    for name, vals in [
        ("go", nav_targets), ("open", open_targets), ("take", take_actions),
        ("move", move_actions), ("state", state_actions), ("use", use_actions),
    ]:
        if vals:
            cats.append(f"{name}={len(vals)}")
    if cats:
        lines.append("- current admissible action type counts: " + ", ".join(cats) + ".")
    if nav_targets:
        lines.append("- current visible go targets: " + ", ".join(nav_targets[:18]) + (" ..." if len(nav_targets) > 18 else "."))
    if open_targets:
        lines.append("- current visible open targets: " + ", ".join(open_targets[:12]) + (" ..." if len(open_targets) > 12 else "."))


    held = _extract_held_objects_from_actions(avail, target_class=target) if avail else []
    if held:
        lines.append("- held-object estimate from current admissible move/state actions: " + ", ".join(held[:6]) + ".")


    if target:
        target_mentions = []
        for a in avail:
            obj = _alfworld_action_object(a)
            if obj and _target_matches_obj(obj, target):
                target_mentions.append(a)
            elif a.lower().startswith("go to ") and _alfworld_object_class(a[6:].strip()) == target:
                target_mentions.append(a)
        if target_mentions:
            lines.append("- current admissible strings mentioning the target class: " + " | ".join(target_mentions[:10]) + (" ..." if len(target_mentions) > 10 else "."))
    if dest:
        dest_mentions = [
            a for a in avail
            if (a.lower().startswith("go to ") and _alfworld_object_class(a[6:].strip()) == dest)
            or (a.lower().startswith("move ") and _alfworld_object_class(_alfworld_action_move_dest(a)) == dest)
        ]
        if dest_mentions:
            lines.append("- current admissible strings mentioning the destination class: " + " | ".join(dest_mentions[:10]) + (" ..." if len(dest_mentions) > 10 else "."))
    if verb in {"clean", "cool", "heat"} and target:
        appliance = {"clean": "sinkbasin", "cool": "fridge", "heat": "microwave"}.get(verb)
        state_mentions = [
            a for a in avail
            if a.lower().startswith(f"{verb} ") and _target_matches_obj(_alfworld_action_object(a), target)
        ]
        appliance_mentions = [
            a for a in avail
            if appliance and (
                (a.lower().startswith("go to ") and _alfworld_object_class(a[6:].strip()) == appliance)
                or appliance in a.lower()
            )
        ]
        if state_mentions:
            lines.append(f"- current admissible strings for visible `{verb}` state on the target class: " + " | ".join(state_mentions[:6]) + (" ..." if len(state_mentions) > 6 else "."))
        elif appliance_mentions:
            lines.append(f"- current admissible strings mentioning the visible `{verb}` appliance class: " + " | ".join(appliance_mentions[:8]) + (" ..." if len(appliance_mentions) > 8 else "."))
    if parsed.get("examine_with_desklamp"):
        lamp_mentions = [a for a in avail if "desklamp" in a.lower() or a.lower().startswith("use ")]
        if lamp_mentions:
            lines.append("- current admissible strings mentioning desklamp/use actions: " + " | ".join(lamp_mentions[:8]) + (" ..." if len(lamp_mentions) > 8 else "."))


    if react_history:
        recent_actions = [str(a) for _obs, a in react_history[-8:]]
        lines.append("- recent action history: " + " | ".join(recent_actions) + ".")

        state_done_actions = []
        for _obs, act in react_history:
            act_s = str(act or "")
            low = act_s.lower()
            if low.startswith(("clean ", "cool ", "heat ")):
                obj = _alfworld_action_object(act_s)
                if (not target) or _target_matches_obj(obj, target):
                    state_done_actions.append(act_s)
        if state_done_actions:
            lines.append("- visible state-transform actions already taken: " + " | ".join(state_done_actions[-4:]) + ".")

        if target:
            placed_records = _alfworld_placed_target_records(env_self, target, dest or None)
            if placed_records:
                placed_desc = ", ".join(f"{obj}->{dst}" for obj, dst in placed_records[:6])
                count_part = f" ({len(placed_records)}/{count})" if count > 1 else ""
                lines.append("- visible target placement records from history" + count_part + ": " + placed_desc + ".")

        visited = {}
        total = len(react_history)
        for j, (_pre_obs, act) in enumerate(react_history):
            act_s = str(act or "")
            if not act_s.startswith("go to "):
                continue
            loc = act_s[6:].strip()
            if j + 1 < total:
                result = react_history[j + 1][0]
            else:
                result = current_obs
            result_s = " ".join(str(result or "").split())
            if result_s.lower().startswith("[state_unchanged]"):
                continue
            visited[loc] = result_s[:160]
        if visited:
            lines.append("- visited locations from history:")
            for loc, obs_s in list(visited.items())[-8:]:
                if target:
                    status = "target text visible" if _observation_mentions_object_class(obs_s, target) else "target text not visible"
                else:
                    status = "observed"
                lines.append(f"  • {loc}: {status}; {obs_s}")
            unchecked = [x for x in nav_targets if x not in set(visited.keys())]
            if unchecked:
                lines.append("- currently visible go targets not recorded as visited in history: " + ", ".join(unchecked[:18]) + (" ..." if len(unchecked) > 18 else "."))

    return "\n".join(lines)

def _insert_alfworld_task_brief_and_memory(prompt: str, env_self, avail_actions) -> str:
    parts = []
    brief = _build_alfworld_task_brief(getattr(env_self, "_task_description", "") or "")
    if brief:
        parts.append(brief)
    mem = _build_alfworld_visible_memory(env_self, avail_actions)
    if mem:
        parts.append(mem)
    if not parts:
        return prompt
    block = "\n\n".join(parts)
    marker = "Now it's your turn to take an action."
    if marker in prompt:
        return prompt.replace(marker, block + "\n\n" + marker, 1)
    return prompt.rstrip() + "\n\n" + block + "\n"

def _insert_alfworld_skill_trailer(prompt: str, task_description: str = "") -> str:
    if not _alfworld_skill_trailer_enabled():
        return prompt
    marker = "Now it's your turn to take an action."
    task_line = f"Task reminder: {task_description.strip()}\n\n" if task_description else ""
    trailer = task_line + _ALFWORLD_SKILL_TRAILER.strip()
    if marker in prompt:
        return prompt.replace(marker, trailer + "\n\n" + marker, 1)
    return prompt.rstrip() + "\n\n" + trailer + "\n"


def _insert_alfworld_skill_apply_block(prompt: str, env_self, avail_actions) -> str:
    if not _alfworld_skill_apply_block_enabled() or _alfworld_decision_block_enabled():
        return prompt
    marker = "Now it's your turn to take an action."
    try:
        block = _build_alfworld_decision_state_block(env_self, avail_actions)
    except Exception as exc:
        block = f"[APPLY LEARNED ALFWORLD SKILL]\nCould not build state note: {exc}"
    block = "[APPLY LEARNED ALFWORLD SKILL TO CURRENT STATE]\n" + str(block).strip()
    if marker in prompt:
        return prompt.replace(marker, block + "\n\n" + marker, 1)
    return prompt.rstrip() + "\n\n" + block + "\n"


_WEBSHOP_NAV_CLICKABLES = {
    "back to search", "< prev", "prev", "next >", "description",
    "features", "reviews", "buy now",
}
_WEBSHOP_OPTION_LABELS = {
    "color", "size", "scent", "flavor name", "flavor", "flavour", "style", "pattern",
    "quantity", "pack", "count", "dimension", "dimensions", "material",
    "fit", "fit type", "item shape", "shape",
}
_WEBSHOP_COLOR_WORDS = {
    "black", "blue", "brown", "charcoal", "green", "grey", "gray", "orange",
    "pink", "purple", "red", "white", "yellow", "navy", "silver", "gold",
}
_WEBSHOP_QUERY_ATTR_PHRASES = (


    "100 percent", "100 vegan", "soy free", "plant based", "dairy free",
    "non gmo", "gluten free", "artificial flavors", "slim fit", "loose fit",
    "classic fit", "straight leg", "machine washable", "machine wash",
    "slip resistant", "non slip", "rubber outsole", "rubber sole",
    "button closure", "contrast color", "high quality", "day comfort",
    "hand wash", "long sleeve", "short sleeve", "stretch fabric",
    "polyester spandex", "polyester heathers", "heathers cotton",
    "cotton heather", "needle sleeve", "eco friendly", "fleece throw",
    "for teen girls", "teen girls", "daily wear", "wireless bluetooth",
)


def _webshop_norm(text: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9.%]+", " ", str(text).lower())).strip()


def _webshop_price_limit(task: str):
    import re as _re
    m = _re.search(r"price\s+lower\s+than\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", str(task).lower())
    return float(m.group(1)) if m else None


def _webshop_task_attr_value(task: str, label: str) -> str:

    import re as _re
    raw = str(task or "")
    lab = _re.escape(label)
    labels = r"(?:color|size|fit type|flavor name|flavor|scent|style|pattern|count|number|dimensions?|width|height|item shape|shape)"


    pat = rf"\b{lab}\s*:\s*(.*?)(?=,\s*(?:and\s+)?{labels}\b\s*:?|,?\s*and\s+price\s+lower\s+than\b|$)"
    m = _re.search(pat, raw, flags=_re.IGNORECASE)
    if not m:
        return ""
    value = m.group(1).strip(" ,.;")


    value = _re.sub(r"^\s*and\s+", "", value, flags=_re.IGNORECASE).strip()
    return value


def _webshop_has_phrase(text: str, phrase: str) -> bool:
    import re as _re
    t = _webshop_norm(text)
    p = _webshop_norm(phrase)
    if not p:
        return False
    return bool(_re.search(rf"(?<![a-z0-9]){_re.escape(p)}(?![a-z0-9])", t))


def _webshop_compact_query(task: str) -> str:

    import re as _re
    raw = str(task or "").lower()
    price = _webshop_price_limit(raw)

    cleaned = raw.replace("100% vegan", "100 vegan")
    cleaned = _re.sub(r"\b(find me|show me|place order for|hello)\b", " ", cleaned)
    cleaned = _re.sub(r"\b(i am|i'm|im|i)\s+(am\s+)?(looking|look|need|want|would like|like|shopping|shop|searching)\s+(for|to buy|to find)?\b", " ", cleaned)
    cleaned = _re.sub(r",?\s*and\s+price\s+lower\s+than\s*\$?\s*[0-9]+(?:\.[0-9]+)?\s*dollars?", " ", cleaned)
    cleaned = _re.sub(r"\bprice\s+lower\s+than\s*\$?\s*[0-9]+(?:\.[0-9]+)?\s*dollars?\b", " ", cleaned)
    cleaned = _re.sub(r"\s+", " ", cleaned).strip(" ,-")


    prefix = cleaned.split(" with ")[0]
    prefix = _re.sub(r"\bfor\s+(daily wear|dry clean|tumble dry|living room|dining room|bedroom|office|teen girls|teen boys)\b(?:\s*,\s*\w+(?:\s+\w+)*)*", " ", prefix)
    segs = [seg.strip() for seg in prefix.split(",") if seg.strip()]
    if len(segs) > 1:
        tail = segs[-1]
        prev = segs[-2]
        if _re.search(r"\b(men|men's|women|women's|boys|girls|unisex)\b", prev) and not _re.search(r"\b(men|men's|women|women's|boys|girls|unisex)\b", tail):
            prefix = prev + " " + tail
        elif _re.search(r"\b(men|men's|women|women's|boys|girls|unisex)\b", tail):
            prefix = tail
        else:


            prefix = " ".join(segs)
    prefix = _re.sub(r"[^a-z0-9.&' -]+", " ", prefix)
    prefix = _re.sub(r"\s+", " ", prefix).strip(" -")

    changed = True
    while changed and prefix:
        changed = False
        for phrase in _WEBSHOP_QUERY_ATTR_PHRASES:
            pat = rf"^(?:{_re.escape(phrase)})\b[\s,-]*"
            new_prefix = _re.sub(pat, "", prefix).strip(" ,-")
            if new_prefix != prefix:
                prefix = _re.sub(r"\s+", " ", new_prefix).strip(" ,-")
                changed = True

    attr_terms = []
    if " with " in cleaned:
        attr_text = cleaned.split(" with ", 1)[1]
        labels = r"(?:color|size|fit type|flavor name|flavor|scent|style|pattern|count|number|dimensions?|width|height|item shape|shape)"
        attr_text = _re.sub(rf",?\s+and\s+(?={labels}\b\s*:?)", ", ", attr_text)
        for m in _re.finditer(rf"\b{labels}\b\s*:?\s*([^,;]+?)(?=\s+\b{labels}\b\s*:?|,\s*\b{labels}\b\s*:?|$)", attr_text):
            val = m.group(1)
            val = _re.sub(r"\b(and|or|is|are|should|be|option|value)\b", " ", val)
            val = _re.sub(r"[^a-z0-9.&+'\"| -]+", " ", val)
            val = _re.sub(r"\s+", " ", val).strip(" ,-")
            if val and val not in attr_terms:
                attr_terms.append(val)
        distinctive = _re.findall(r"\b[a-z]*\d+[a-z0-9]*(?:[.\-+x][a-z0-9]+)+\b|\b[a-z]+-[a-z0-9.-]+\b|\b\d+(?:\.\d+)?\s*(?:w|wide|x|in|inch|oz|lb|lbs)\b|\b[a-z]+\s+combi\b", attr_text)
        for val in distinctive:
            val = _re.sub(r"\s+", " ", val).strip(" ,-")
            if val and not any(val in existing for existing in attr_terms):
                attr_terms.append(val)

    kept = []
    for val in attr_terms:
        words = val.split()
        if len(words) > 10:
            val = " ".join(words[:10])
        if val and val not in kept:
            kept.append(val)
        if len(kept) >= 4:
            break


    try:
        descriptor_terms = []
        for ph in _webshop_visible_requirement_phrases(task):
            pn = _webshop_norm(ph)
            if not pn or pn == "box spring":
                continue
            if pn in _webshop_norm(prefix) or any(pn == _webshop_norm(x) for x in kept):
                continue


            if len(pn.split()) <= 4:
                descriptor_terms.append(ph)
        for ph in descriptor_terms[:5]:
            if ph not in kept:
                kept.append(ph)
    except Exception:
        pass

    q = " ".join([x for x in [prefix] + kept if x]).strip()
    q = _re.sub(r"\s+", " ", q).strip()
    if len(q) < 3:
        q = _re.sub(r"[^a-z0-9.&' -]+", " ", cleaned)
        q = _re.sub(r"\s+", " ", q).strip()[:100]
    if price is not None:
        price_s = str(int(price)) if float(price).is_integer() else str(price)
        if f"under {price_s}" not in q:
            q = (q + f" under {price_s}").strip()
    return q[:140]

def _webshop_tokens(obs: str):
    import re as _re
    text = str(obs or "")


    if "[SEP]" in text:
        parts = _re.split(r"\s*\[SEP\]\s*", text)
    else:
        parts = _re.split(r"\s*\|\s*", text)
    return [t.strip() for t in parts if t and t.strip()]


def _webshop_parse_product_options(obs: str):

    toks = _webshop_tokens(obs)
    low = [t.lower() for t in toks]
    if "buy now" not in low or "< prev" not in low:
        return {}, ""
    try:
        start = low.index("< prev") + 1
    except ValueError:
        start = 0

    title_idx = None
    for i, t in enumerate(low):
        if t.startswith("price:"):
            title_idx = max(i - 1, start)
            break
    if title_idx is None:
        return {}, ""
    option_span = toks[start:title_idx]
    title = toks[title_idx] if 0 <= title_idx < len(toks) else ""
    groups = {}
    current = None
    for tok in option_span:
        key = tok.strip().lower()
        if key in _WEBSHOP_OPTION_LABELS:
            current = key
            groups.setdefault(current, [])
        elif current and tok.strip():
            groups.setdefault(current, []).append(tok.strip())

    for k, vals in list(groups.items()):
        seen, uniq = set(), []
        for v in vals:
            nv = _webshop_norm(v)
            if nv and nv not in seen:
                seen.add(nv); uniq.append(v)
        groups[k] = uniq
    return groups, title


def _webshop_parse_results(obs: str):

    import re as _re
    toks = _webshop_tokens(obs)
    rows = []
    for i, tok in enumerate(toks[:-2]):
        if not _re.fullmatch(r"B[0-9A-Z]{9}", tok.strip().upper()):
            continue
        title = toks[i + 1].strip() if i + 1 < len(toks) else ""
        price_tok = toks[i + 2].strip() if i + 2 < len(toks) else ""
        m = _re.search(r"\$([0-9]+(?:\.[0-9]+)?)", price_tok)
        price = float(m.group(1)) if m else None
        if title:
            rows.append((tok.strip().lower(), title, price))
    return rows


def _webshop_visible_requirement_phrases(task: str) -> list[str]:

    import re as _re
    t = _webshop_norm(task)
    phrases = []
    known = [
        "queen size", "long handle", "dry skin", "dual band", "quad core",
        "power amplifier", "hands free", "usb port", "steel frame", "storage space",
        "living room", "slim fit", "loose fit", "straight leg", "elastic waist",
        "elastic closure", "faux fur", "long sleeve", "short sleeve", "regular fit",
        "classic fit", "button closure", "machine wash", "wash cold", "dry clean",
        "tumble dry", "polyester heathers", "heathers cotton", "cotton heather",
        "needle sleeve", "unique design", "relaxed fit", "polyester spandex",
        "tummy control", "high waist", "rubber outsole", "rubber sole", "non slip",
        "slip resistant", "anti slip", "tempered glass", "glass screen", "case cover",
    ]
    for ph in known:
        if _webshop_has_phrase(t, ph):
            phrases.append(ph)


    for lab in ("color", "size", "fit type", "item shape", "shape", "scent", "flavor name", "flavor"):
        val = _webshop_task_attr_value(task, lab)
        vn = _webshop_norm(val)
        if vn and vn not in {_webshop_norm(x) for x in phrases}:
            phrases.append(val)
    out, seen = [], set()
    for ph in phrases:
        n = _webshop_norm(ph)
        if n and n not in seen:
            seen.add(n); out.append(ph)
    return out


def _webshop_product_price(obs: str):
    import re as _re
    toks = _webshop_tokens(obs)
    for tok in toks:
        m = _re.search(r"price:\s*\$\s*([0-9]+(?:\.[0-9]+)?)", str(tok), flags=_re.I)
        if m:
            return float(m.group(1))
    return None


def _webshop_recent_searches(env_self) -> list[str]:
    import re as _re
    out = []
    for _obs, act in list(getattr(env_self, "_react_history", []) or []):
        m = _re.fullmatch(r"search\[(.*)\]", str(act or "").strip(), flags=_re.I)
        if m:
            out.append(m.group(1).strip())
    return out


def _webshop_best_visible_result(task: str, obs: str, avoid_asins=None):

    import re as _re
    rows = _webshop_parse_results(obs)
    if not rows:
        return None
    avoid_asins = {str(a).strip().lower() for a in (avoid_asins or []) if str(a).strip()}
    price_limit = _webshop_price_limit(task)


    simple_rank_first = _env_flag("WEBSHOP_SIMPLE_RANK_FIRST", False)
    if simple_rank_first and _env_flag("WEBSHOP_SEARCH_RANK_FIRST", True):
        for row_i, (asin, title, price) in enumerate(rows[:12]):
            if asin in avoid_asins:
                continue
            if price_limit is not None and price is not None and price >= price_limit:
                continue
            return (100.0 - float(row_i), asin, title, price)
    task_n = _webshop_norm(task)
    task_words = [
        w for w in task_n.split()
        if len(w) >= 3 and w not in {
            "looking", "look", "need", "want", "show", "some", "with",
            "that", "helps", "maintain", "price", "lower", "than",
            "dollars", "under", "color", "size", "scent", "flavor",
            "type", "types", "pack", "pair", "pairs", "shop", "buy",
        }
    ]
    head_words = {
        "lotion", "toothbrush", "toothbrushes", "serum", "mask", "towel",
        "wrap", "shake", "jacket", "coat", "loafers", "pants", "ottoman",
        "deodorant", "almonds", "shorts", "box", "storage", "protein",
        "blanket", "blankets", "peas", "receivers", "amplifiers", "shoes",
        "jeans", "shirts", "shirt", "t-shirt", "t-shirts", "tees", "tops",
        "sweaters", "sweater", "hoodie", "hoodies", "sweatshirt", "sweatshirts",
        "henley", "henleys", "lingerie", "sleepwear", "lounge", "bed", "beds",
        "media", "players", "player", "brush", "shorts", "suits", "suit",
        "blazer", "blazers",
    }
    price_limit = _webshop_price_limit(task)
    unit_reqs = _re.findall(r"\b\d+(?:\.\d+)?\s*(?:ml|oz|ounce|ounces|cm|inch|count|pack|pcs|pc)\b", task_n)
    color_reqs = [c for c in _WEBSHOP_COLOR_WORDS if _re.search(rf"(?<![a-z0-9]){_re.escape(c)}(?![a-z0-9])", task_n)]
    core_q = _re.sub(r"\s+under\s+\d+(?:\.\d+)?\b", "", _webshop_compact_query(task))
    core_words = [
        w for w in _webshop_norm(core_q).split()
        if len(w) >= 3 and w not in {"under", "and", "the", "men", "mens", "women", "womens"}
    ]
    req_size = _webshop_task_attr_value(task, "size")
    req_size_n = _webshop_norm(req_size)
    has_task_options = any(_webshop_task_attr_value(task, lab) for lab in ("color", "size", "fit type", "item shape", "scent", "flavor"))
    task_is_mens = bool(_re.search(r"\b(men|men's|mens)\b", str(task).lower()))
    task_is_womens = bool(_re.search(r"\b(women|women's|womens)\b", str(task).lower()))


    size_token_re = _re.compile(
        r"\b(?:\d+w\s*x\s*\d+l|\d+w|\d+t|x-small|small|medium|large|x-large|xx-large|xxx-large|\d+x-large|\d+x)\b"
    )

    best = None
    for row_i, (asin, title, price) in enumerate(rows[:12]):
        title_n = _webshop_norm(title)


        score = -1.0 * row_i
        if asin in avoid_asins:
            score -= 12.0
        if price_limit is not None and price is not None:
            score += 4.0 if price < price_limit else -10.0
        if row_i == 0:
            score += 7.0
        if has_task_options and "multiple size color options" in title_n:


            score += 12.0
        if task_is_mens and _re.search(r"\b(men|mens)\b", title_n):
            score += 3.0
        if task_is_womens and _re.search(r"\b(women|womens)\b", title_n):
            score += 3.0
        for w in task_words:
            if w in title_n.split() or w in title_n:
                score += 1.0
        for w in core_words:
            if w in title_n.split() or w in title_n:
                score += 2.0
        if "suits" in task_n or "sport coats" in task_n:
            if "swimsuit" in title_n:
                score -= 10.0
            if (
                _re.search(r"\b(suit|suits|blazer|blazers|tuxedo)\b", title_n)
                or "sport coat" in title_n
            ) and "swimsuit" not in title_n:
                score += 10.0
            elif any(x in title_n for x in ("t shirt", "t shirts", "tee", "tees", "cargo pants", "shirt")):
                score -= 8.0
        if ("hoodies" in task_n or "sweatshirts" in task_n) and not any(x in title_n for x in ("hoodie", "hooded", "sweatshirt")):
            score -= 5.0
        if "streaming media players" in task_n:
            if "dual band" in task_n and "dual band" in title_n:
                score += 8.0
            if "remote" in title_n and "player" not in title_n.replace("remote", ""):
                score -= 8.0
            if "wifi extender" in title_n or "signal repeater" in title_n:
                score -= 5.0
        if "bath" in task_n and "long handle" in task_n:
            if "long handle" in title_n:
                score += 7.0
            if "dry" in task_n and "dry" in title_n:
                score += 5.0
            if "bathtub cushion" in title_n and "long handle" not in title_n:
                score -= 8.0
        if "queen size beds" in task_n or ("queen" in task_n and "beds" in task_n):
            if "queen" in title_n:
                score += 7.0
            if "twin" in title_n:
                score -= 10.0
        task_heads = [w for w in task_words if w in head_words]
        title_has_head = any(h in title_n for h in task_heads)
        if task_heads:
            score += 5.0 if title_has_head else -6.0

        req_phrases = _webshop_visible_requirement_phrases(task)
        matched_req = 0
        for ph in req_phrases:
            pn = _webshop_norm(ph)
            if not pn:
                continue


            words = [w for w in pn.split() if len(w) >= 3]
            exact = _webshop_has_phrase(title_n, pn)
            partial = words and all(w in title_n for w in words)
            any_word = words and any(w in title_n for w in words)
            if exact or partial:
                score += 5.0
                matched_req += 1
            elif any_word:
                score += 1.5
                matched_req += 0.3


        if req_phrases and matched_req == 0:
            score -= 4.0


        if "body lotion" in task_n and "lotion" not in title_n:
            score -= 8.0
        if "body lotion" in task_n and any(x in title_n for x in ("self tanner", "deodorant", "cleansing milk", "body wash")):
            score -= 6.0
        if "fresh scent" in task_n and any(x in title_n for x in ("cinnamon paprika", "coconut", "honeydew", "stone crop")):
            score -= 5.0
        if "hair mask" in task_n and not ("hair" in title_n and ("mask" in title_n or "treatment" in title_n or "conditioner" in title_n)):
            score -= 6.0
        for req in unit_reqs:
            rn = _webshop_norm(req)
            rn_alt = rn.replace(" ounce", " oz").replace(" ounces", " oz")
            rn_alt2 = rn.replace(" oz", " ounce")
            if rn in title_n or rn_alt in title_n or rn_alt2 in title_n:
                score += 3.0
            elif any(u in title_n for u in ("60ml", "60 ml", "100ml", "100 ml", "500ml", "500 ml", "2 oz", "4.25 oz", "12 oz", "16 oz")) and rn not in title_n and rn_alt not in title_n:
                score -= 2.0
        if "variety" in task_n and "variety" in title_n:
            score += 2.0
        if "peas" in task_n and any(x in title_n for x in ("paste", "seaweed", "soup", "vegetables", "protein bar")) and "peas" not in title_n:
            score -= 4.0
        if "peas" in task_n and any(x in title_n for x in ("snack", "snacks", "pops", "puffed")):
            score += 2.0
        for color in color_reqs:
            if color in title_n:
                score += 2.0
        visible_colors = [c for c in _WEBSHOP_COLOR_WORDS if _re.search(rf"(?<![a-z0-9]){_re.escape(c)}(?![a-z0-9])", title_n)]
        if color_reqs and visible_colors and not any(c in color_reqs for c in visible_colors):
            score -= 3.0
        if "long sleeve" in task_n:
            if "long sleeve" in title_n:
                score += 3.0
            if "short sleeve" in title_n:
                score -= 4.0
        if "short sleeve" in task_n and "long sleeve" not in task_n:
            if "short sleeve" in title_n:
                score += 2.0
            if "long sleeve" in title_n:
                score -= 2.0
        if task_is_mens and ("women " in title_n or "women s" in title_n or "womens" in title_n):
            score -= 10.0
        if task_is_womens and (" men " in f" {title_n} " or " men s" in f" {title_n} " or " mens" in f" {title_n} "):
            score -= 10.0
        if req_size_n:
            if req_size_n in title_n:
                score += 3.0
            elif size_token_re.search(title_n):
                score -= 3.0

        if best is None or score > best[0]:
            best = (score, asin, title, price)
    return best


def _webshop_strong_guidance_enabled():


    return _env_flag("WEBSHOP_ALLOW_STRONG_GUIDANCE", False) and _env_flag("WEBSHOP_STRONG_GUIDANCE", False)


def _webshop_env_feedback_enabled():


    return _env_flag("WEBSHOP_ENV_FEEDBACK", False)


def _webshop_exact_option_guard_enabled():


    return False


def _webshop_soft_exact_warning_enabled():


    return _env_flag("WEBSHOP_SOFT_EXACT_WARNING", False)


def _webshop_visible_state_feedback_enabled():


    return _env_flag("WEBSHOP_VISIBLE_STATE_FEEDBACK", False)


def _webshop_prompt_style():

    val = os.environ.get("WEBSHOP_PROMPT_STYLE", "raw")
    val = str(val or "raw").strip().lower().replace("-", "_")
    aliases = {
        "default": "raw",
        "local": "raw",
        "skill_rl": "skillrl",
        "ragen": "skillrl",
        "agentrk": "agentbench",
        "agentrl": "agentbench",
    }
    return aliases.get(val, val if val in {"raw", "skillrl", "agentbench"} else "raw")


def _webshop_skill_enabled():
    return _env_flag("WEBSHOP_SKILL_ENABLED", True)


def _webshop_skill_placement():

    val = os.environ.get("WEBSHOP_SKILL_PLACEMENT", "prefix")
    val = str(val or "prefix").strip().lower().replace("-", "_")
    if val in {"0", "false", "none", "off", "no"}:
        return "off"
    if val in {"memory", "retrieved", "skillrl", "skillrl_memory"}:
        return "skillrl_memory"
    return "prefix"


def _format_webshop_actions_dict(avail):

    if isinstance(avail, dict):
        return str({
            "has_search_bar": bool(avail.get("has_search_bar")),
            "clickables": [str(x) for x in avail.get("clickables", [])],
        })
    return str(avail)


def _render_webshop_prompt(
    *,
    init: bool,
    task_description: str,
    current_observation: str,
    available_actions,
    action_history: str = "",
    step_count: int = 0,
    history_length: int = 0,
    current_step: int = 1,
    skill_text: str = "",
):
    from training.react_prompts import (
        WEBSHOP_TEMPLATE_NO_HIS, WEBSHOP_TEMPLATE,
        WEBSHOP_TEMPLATE_NO_HIS_SKILLRL, WEBSHOP_TEMPLATE_SKILLRL,
        WEBSHOP_TEMPLATE_WITH_MEMORY_SKILLRL,
        WEBSHOP_TEMPLATE_NO_HIS_AGENTBENCH, WEBSHOP_TEMPLATE_AGENTBENCH,
    )
    style = _webshop_prompt_style()
    placement = _webshop_skill_placement() if _webshop_skill_enabled() else "off"
    skill_text = (skill_text or "").strip()
    formatted_actions = _format_webshop_actions(available_actions)

    if style == "skillrl":
        use_memory = bool(skill_text) and placement == "skillrl_memory" and (not init or _env_flag("WEBSHOP_SKILL_ON_INIT", True))
        if use_memory:
            return WEBSHOP_TEMPLATE_WITH_MEMORY_SKILLRL.format(
                task_description=task_description,
                retrieved_memories=skill_text,
                step_count=step_count,
                history_length=history_length,
                action_history=action_history,
                current_step=current_step,
                current_observation=current_observation,
                available_actions=formatted_actions,
            )
        tmpl = WEBSHOP_TEMPLATE_NO_HIS_SKILLRL if init else WEBSHOP_TEMPLATE_SKILLRL
        kwargs = dict(
            task_description=task_description,
            current_observation=current_observation,
            available_actions=formatted_actions,
        )
        if not init:
            kwargs.update(
                step_count=step_count,
                history_length=history_length,
                action_history=action_history,
                current_step=current_step,
            )
        return tmpl.format(**kwargs)

    if style == "agentbench":
        tmpl = WEBSHOP_TEMPLATE_NO_HIS_AGENTBENCH if init else WEBSHOP_TEMPLATE_AGENTBENCH
        return tmpl.format(
            current_observation=current_observation,
            available_actions_dict=_format_webshop_actions_dict(available_actions),
            action_history=action_history,
        )


    tip_prefix = (
        f"\n# Learned Strategy (follow this)\n{skill_text}\n\n"
        if skill_text and placement == "prefix" else ""
    )
    tmpl = WEBSHOP_TEMPLATE_NO_HIS if init else WEBSHOP_TEMPLATE
    kwargs = dict(
        task_description=task_description,
        current_observation=current_observation,
        available_actions=formatted_actions,
    )
    if not init:
        kwargs.update(
            step_count=step_count,
            history_length=history_length,
            action_history=action_history,
            current_step=current_step,
        )
    return tip_prefix + tmpl.format(**kwargs)


def _webshop_opened_asins(env_self):
    import re as _re
    out = []
    for _obs, act in list(getattr(env_self, "_react_history", []) or []):
        a = str(act or "").strip().lower()
        m = _re.fullmatch(r"click\[(b[0-9a-z]{9})\]", a)
        if m:
            out.append(m.group(1))
    return out


def _webshop_clicked_values(env_self, groups=None):

    import re as _re
    visible_norms = None
    if groups:
        visible_norms = set()
        for vals in (groups or {}).values():
            for v in vals:
                nv = _webshop_norm(v)
                if nv:
                    visible_norms.add(nv)
    vals_rev = []
    for _obs, act in reversed(list(getattr(env_self, "_react_history", []) or [])):
        a = str(act or "").strip()
        al = a.lower()
        if al.startswith("search["):
            break
        if al == "click[back to search]":
            break
        if al.startswith("click[") and a.endswith("]"):
            val = a[6:-1].strip()
            vnorm = val.lower()
            if _re.fullmatch(r"b[0-9a-z]{9}", vnorm):


                break
            if vnorm in _WEBSHOP_NAV_CLICKABLES:
                continue
            nv = _webshop_norm(val)
            if visible_norms is not None and nv not in visible_norms:

                continue
            if val:
                vals_rev.append(val)
    vals = list(reversed(vals_rev))

    out, seen = [], set()
    for v in vals:
        nv = _webshop_norm(v)
        if nv and nv not in seen:
            seen.add(nv)
            out.append(v)
    return out

def _webshop_option_suggestion(task: str, group: str, values, selected):

    import re as _re
    task_l = _webshop_norm(task)
    selected_n = {_webshop_norm(x) for x in selected}
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return None, ""

    if any(_webshop_norm(v) in selected_n for v in vals):
        return None, "already selected"

    norm_vals = [(v, _webshop_norm(v)) for v in vals]

    def contains_word(word: str) -> bool:
        return bool(_re.search(rf"(?<![a-z0-9]){_re.escape(word)}(?![a-z0-9])", task_l))

    def contains_phrase(phrase: str) -> bool:


        return bool(_re.search(rf"(?<![a-z0-9]){_re.escape(phrase)}(?![a-z0-9])", task_l))


    if group in {"size", "quantity", "pack", "count"} and contains_word("one"):
        for v, nv in norm_vals:
            if nv in {"1pcs", "1 pcs", "1pc", "1 pc", "1"} or nv.startswith("1pcs"):
                return v, "task says one; visible size/count options include 1pcs"


    if group in {"size", "fit type"}:
        req_val = _webshop_task_attr_value(task, group)
        req_n = _webshop_norm(req_val)
        if req_n:
            for v, nv in norm_vals:
                if nv == req_n:
                    return v, f"task requests exact {group} `{req_val}` and matching visible option exists"
            if _webshop_exact_option_guard_enabled():
                return None, f"task requests exact {group} `{req_val}` but no visible option exactly matches"


    if group == "color":
        requested_color = _webshop_task_attr_value(task, "color")
        requested_color_n = _webshop_norm(requested_color)
        if requested_color_n:
            for v, nv in norm_vals:
                if nv == requested_color_n or contains_phrase(nv) and nv == requested_color_n:
                    return v, f"task requests exact color `{requested_color}` and matching visible option exists"


            simple_color = requested_color_n in _WEBSHOP_COLOR_WORDS
            if not simple_color and _webshop_exact_option_guard_enabled():
                return None, f"task requests exact color `{requested_color}` but no visible color option exactly matches"
        requested = [c for c in _WEBSHOP_COLOR_WORDS if contains_word(c)]
        for color in requested:
            exact = [v for v, nv in norm_vals if nv == color]
            if exact:
                return exact[0], f"task requests color `{color}`; exact single-color option is visible"
        for color in requested:
            contains = [v for v, nv in norm_vals if color in nv.split()]
            if contains:
                return contains[0], f"task requests color `{color}`; closest visible color option"


    if group not in {"size", "fit type"}:
        for v, nv in norm_vals:
            if nv and contains_phrase(nv):
                return v, f"visible option phrase `{v}` appears in the task"


    unit_reqs = _re.findall(r"\b\d+(?:\.\d+)?\s*(?:ml|oz|ounce|ounces|cm|inch|count|pack|pcs|pc)\b", task_l)
    for req in unit_reqs:
        req_n = _webshop_norm(req)
        for v, nv in norm_vals:
            if req_n == nv or req_n in nv:
                return v, f"task requests `{req}` and matching visible option exists"


    if group in {"item shape", "shape"}:
        req_val = _webshop_task_attr_value(task, "item shape") or _webshop_task_attr_value(task, "shape")
        req_n = _webshop_norm(req_val)
        if req_n:
            for v, nv in norm_vals:
                if nv == req_n or contains_phrase(nv) and nv == req_n:
                    return v, f"task requests exact {group} `{req_val}` and matching visible option exists"
            if _webshop_exact_option_guard_enabled():
                return None, f"task requests exact {group} `{req_val}` but no visible option exactly matches"


    if group in {"scent", "flavor name", "flavor", "flavour"}:
        cue_words = []
        for cue in ("scent", "flavor", "flavour"):
            if contains_word(cue):
                cue_words.append(cue)


        if cue_words:
            return None, f"task asks for {', '.join(cue_words)} but no visible option value exactly matches; avoid contradictory option values"


    if len(vals) == 1 and group in task_l:
        return vals[0], f"only one visible {group} option"
    return None, ""


def _webshop_missing_visible_exact_attrs(task: str, title: str, groups, selected) -> list[str]:

    missing = []
    selected_text = " ".join(str(x) for x in (selected or []))
    option_text = " ".join(" ".join(vals) for vals in (groups or {}).values())
    visible_text = f"{title} {selected_text} {option_text}"
    checks = [
        ("color", "color"),
        ("size", "size"),
        ("fit type", "fit type"),
        ("item shape", "item shape"),
        ("scent", "scent"),
        ("flavor name", "flavor name"),
        ("flavor", "flavor"),
    ]
    for label, group in checks:
        req = _webshop_task_attr_value(task, label)
        if not req:
            continue


        if _webshop_has_phrase(visible_text, req):
            continue
        req_n = _webshop_norm(req)
        if label == "color" and req_n in _WEBSHOP_COLOR_WORDS and _webshop_has_phrase(visible_text, req_n):
            continue
        missing.append(f"{label}={req}")
    return missing


def _append_webshop_neutral_env_feedback(env_self, obs: str, action_str: str, info: dict) -> str:
    if not _webshop_env_feedback_enabled():
        return str(obs)
    text = str(obs)
    action = str(action_str or "").strip()
    al = action.lower()
    hist = [str(a or "") for _o, a in list(getattr(env_self, "_react_history", []) or [])]
    notes = []
    if al.startswith("search[") and hist.count(action) >= 1:
        if any(str(a).lower() == "click[back to search]" for a in hist[-3:]):
            notes.append("same search query was reused after Back; this is a repeated query from the visible action history.")
        else:
            notes.append("same search query was already used earlier in the visible action history.")
    if al == "click[back to search]":
        recent = hist[-4:]
        opened_asins = []
        import re as _re
        for a in recent:
            m = _re.fullmatch(r"click\[(b[0-9a-z]{9})\]", a.strip().lower())
            if m:
                opened_asins.append(m.group(1))
        if opened_asins:
            notes.append(f"returned from product page to search after inspecting ASIN {opened_asins[-1]}; that ASIN is now part of the visible action history.")
        else:
            if hist and str(hist[-1]).lower().startswith("search["):
                notes.append("clicked Back immediately after a search result page; the result list is no longer visible in the current observation.")
            else:
                notes.append("returned to search from the previous visible page state.")
    if not notes:
        return text
    return text.rstrip() + "\n[WEBSHOP ENV FEEDBACK] " + " ".join(notes)


def _build_webshop_visible_state_block(env_self, avail_actions) -> str:

    task = str(getattr(env_self, "_task_description", "") or "")
    obs = str(getattr(env_self, "_current_obs", "") or "")
    groups, title = _webshop_parse_product_options(obs)
    selected = _webshop_clicked_values(env_self, groups)
    price_limit = _webshop_price_limit(task)
    product_price = _webshop_product_price(obs)

    toks = _webshop_tokens(obs)
    low_toks = [t.lower() for t in toks]
    has_search_bar = bool(isinstance(avail_actions, dict) and avail_actions.get("has_search_bar"))
    is_product = bool(groups) or ("buy now" in low_toks and "< prev" in low_toks)
    is_results = ("page 1" in " ".join(low_toks) or "total results" in " ".join(low_toks)) and not is_product
    is_start = (has_search_bar or bool(low_toks and low_toks[-1] == "search")) and not is_product and not is_results

    lines = ["[WEBSHOP VISIBLE STATE FEEDBACK]"]
    lines.append("State summary only from the visible task/page/history; no hidden target, reward, candidate ranking, or next-action recommendation.")
    if price_limit is not None:
        price_s = str(int(price_limit)) if float(price_limit).is_integer() else str(price_limit)
        lines.append(f"- task price limit visible in instruction: < ${price_s}.")

    requested_attrs = []
    for lab in ("color", "size", "fit type", "item shape", "shape", "scent", "flavor name", "flavor"):
        val = _webshop_task_attr_value(task, lab)
        if val:
            requested_attrs.append(f"{lab}={val}")
    if requested_attrs:
        lines.append("- task attribute values visible in instruction: " + "; ".join(requested_attrs[:8]) + ".")

    recent_searches = _webshop_recent_searches(env_self)
    opened_asins = _webshop_opened_asins(env_self)
    if recent_searches:
        lines.append("- recent search queries in history: " + " | ".join(recent_searches[-3:]) + ".")
    if opened_asins:
        lines.append("- ASINs opened in history: " + ", ".join(opened_asins[-8:]) + ".")

    if is_start:
        lines.append("- current page type: search/start page; search box is visible.")
    elif is_results:
        lines.append("- current page type: search results page.")
        rows = _webshop_parse_results(obs)
        if rows:
            row_summaries = []
            for asin, row_title, row_price in rows[:5]:
                price_part = f"${row_price:g}" if row_price is not None else "price not parsed"
                row_summaries.append(f"{asin}: {row_title[:90]} ({price_part})")
            lines.append("- visible result rows shown at top of current page: " + " | ".join(row_summaries) + ".")
        if "next >" in low_toks:
            lines.append("- current page has visible pagination control: Next >.")
    elif is_product:
        lines.append("- current page type: product page.")
        if title:
            lines.append(f"- product page title: {title[:180]}")
        if product_price is not None:
            lines.append(f"- product page price shown: ${product_price:g}.")
        if groups:
            group_text = "; ".join(f"{g}: {', '.join(v[:6])}" for g, v in groups.items())
            lines.append(f"- visible option groups on this product page: {group_text}.")
        if selected:
            lines.append("- option values clicked on the current product according to history: " + ", ".join(selected[-6:]) + ".")
        else:
            lines.append("- no option value click is recorded yet for the current product in history.")
        visible_controls = [t for t in toks if t.lower() in {"back to search", "< prev", "description", "features", "reviews", "buy now"}]
        if visible_controls:
            lines.append("- visible page controls: " + ", ".join(visible_controls) + ".")
    else:
        lines.append("- current page type: other WebShop page/state.")
    return "\n".join(lines)


def _insert_webshop_visible_state_block(prompt: str, env_self, avail_actions) -> str:
    if str(getattr(env_self, "_task_type", "")) != "webshop":
        return prompt
    if not _webshop_visible_state_feedback_enabled():
        return prompt
    try:
        block = _build_webshop_visible_state_block(env_self, avail_actions)
    except Exception as exc:
        block = f"[WEBSHOP VISIBLE STATE FEEDBACK]\nCould not build state note: {exc}"
    markers = [
        "Now it's your turn to take one action for the current step.",
        "Return exactly one executable action string in the form search[keywords] or click[value].",
        "Output ONLY the action you choose. No explanation, no reasoning, just the action.",
    ]
    for marker in markers:
        if marker in prompt:
            return prompt.replace(marker, block + "\n\n" + marker, 1)
    return prompt.rstrip() + "\n\n" + block + "\n"


def _reset_react(self, question):

    from training.react_prompts import (
        ALFWORLD_TEMPLATE_NO_HIS,
        _ALFWORLD_EXAMPLE,
    )

    self._question = question
    self._gold = str(question.get("answer", ""))
    self._task_type = str(question.get("task_type", ""))
    self._extra = question.get("extra", {})
    self._step = 0


    env_type = question.get("env_type", self._task_type)
    env_config = question.get("env_config", {})
    self._ragen_initial_obs = ""
    if self._ragen_adapter and self._task_type in ("webshop", "alfworld", "interactive_agent"):
        try:
            self._ragen_initial_obs = self._ragen_adapter.reset(env_type, env_config)
        except Exception as e:
            logger.warning(f"[ReAct] RAGEN reset failed: {e}")
            self._ragen_initial_obs = f"[ENV_UNAVAILABLE] {e}"

    self._env_done = False
    self._env_reward = 0.0


    self._react_history = []


    if self._task_type == "webshop":
        self._current_obs = str(self._ragen_initial_obs)
    else:
        self._current_obs = self._ragen_initial_obs


    if self._task_type == "webshop" and " [SEP] " in self._ragen_initial_obs:
        parts = self._ragen_initial_obs.split(" [SEP] ")
        self._task_description = parts[2] if len(parts) >= 3 else str(question.get("question", ""))
    elif self._task_type == "alfworld":
        task_marker = "Your task is to:"
        obs_text = str(self._current_obs or "")
        if task_marker in obs_text:
            self._task_description = obs_text.split(task_marker, 1)[1].strip()
        else:
            self._task_description = str(question.get("question", ""))
    else:
        self._task_description = str(question.get("question", ""))


    avail = self._ragen_adapter.available_actions if self._ragen_adapter else []


    skill_text = ""
    self._injected_skill_ids = []  
    _ws = self.workspace
    _ws_size = _ws.size if _ws else 0
    if (
        self.skill_mode != "policy_action"
        and _ws and _ws_size > 0
        and (self._task_type != "webshop" or _webshop_skill_enabled())
    ):
        candidates = _ws.retrieve(
            self._task_description, task_type=self._task_type, top_k=2
        )
        for tip in candidates:
            tip_plan = getattr(tip, 'plan', '') or ''
            tip_types = getattr(tip.meta, 'task_types', []) if tip.meta else []
            if tip_plan and self._task_type in tip_types:
                skill_text += f"- {tip_plan.strip()}\n"
                self._injected_skill_ids.append(getattr(tip.meta, 'skill_id', 'unknown'))
                logger.info(
                    f"[Skill] ReAct injected for {self._task_type}: "
                    f"[{tip.meta.skill_id}] \"{tip_plan[:60]}\" ({len(tip_plan.split())}w)"
                )
            elif tip_plan:
                logger.info(f"[Skill] ReAct type mismatch: task={self._task_type} tip_types={tip_types}")
        if not self._injected_skill_ids and candidates:
            logger.info(f"[Skill] ReAct no matching tips for {self._task_type} (ws={_ws_size}, cands={len(candidates)})")
    learned_tips = f"\nLearned strategies:\n{skill_text}" if skill_text else ""
    policy_skill_catalog = (
        self._react_skill_catalog_for_policy_action(self._task_description)
        if self.skill_mode == "policy_action" else ""
    )


    if self._task_type == "webshop":
        prompt = _render_webshop_prompt(
            init=True,
            task_description=self._task_description,
            current_observation=self._current_obs,
            available_actions=avail,
            skill_text=skill_text,
        )
        if policy_skill_catalog:
            prompt = policy_skill_catalog + "\n" + prompt
        prompt = _insert_webshop_visible_state_block(prompt, self, avail)
    elif self._task_type == "alfworld":
        tip_prefix = (
            f"\n# Learned Strategy (follow this)\n{skill_text.strip()}\n\n"
            if skill_text else ""
        )
        prompt = policy_skill_catalog + "\n" + tip_prefix + ALFWORLD_TEMPLATE_NO_HIS.format(
            example=_ALFWORLD_EXAMPLE,
            current_observation=self._current_obs,
            admissible_actions=_format_alfworld_actions(_order_alfworld_actions_for_prompt(avail, self)),
        )
        prompt = _insert_alfworld_skill_trailer(prompt, self._task_description)
        prompt = _insert_alfworld_task_brief_and_memory(prompt, self, avail)
        prompt = _insert_alfworld_skill_apply_block(prompt, self, avail)


        if _alfworld_decision_block_enabled():
            decision_block = _build_alfworld_decision_state_block(self, avail)
            prompt = prompt.replace(
                "Now it's your turn to take an action.",
                _ALFWORLD_DECISION_REMINDERS.strip() + "\n\n"
                + decision_block.strip() + "\n\n"
                + "Now it's your turn to take an action."
            )
    else:
        prompt = policy_skill_catalog + "\n" + tip_prefix + f"Task: {self._task_description}\nObservation: {self._current_obs}"

    self._current_prompt = prompt


    traj = Trajectory(
        question=self._task_description,
        gold_answer=self._gold,
        task_type=self._task_type,
    )


    for sid in getattr(self, '_injected_skill_ids', []):
        from training.trajectory import Turn
        traj.add_turn(Turn(
            supervisor_input="",
            supervisor_output=f"skill_invoke({sid})",
            action_type="skill_invoke",
            skill_id=sid,
            instruction="auto_inject",
            observation=skill_text.strip() if skill_text else "",
        ))

    return prompt, traj


def _emit_alfworld_trace(env_self, turn, action_str, obs, reward, done, info,
                          loop_detected, loop_reason, prev_obs_at_decision):

    import os, json, time
    task_type = getattr(env_self, "_task_type", "")
    if task_type == "alfworld":
        trace_dir = os.environ.get("REACT_TRACE_DIR") or os.environ.get("ALFWORLD_TRACE_DIR")
    elif task_type == "webshop":
        trace_dir = os.environ.get("REACT_TRACE_DIR") or os.environ.get("WEBSHOP_TRACE_DIR") or os.environ.get("ALFWORLD_TRACE_DIR")
    else:
        trace_dir = os.environ.get("REACT_TRACE_DIR")
    if not trace_dir or task_type not in {"alfworld", "webshop"}:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        avail = []
        try:
            avail = list(env_self._ragen_adapter.available_actions or [])
        except Exception:
            pass
        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            "task_type": task_type,
            "task_description": (env_self._task_description or "")[:1500],
            "step": env_self._step,
            "action_str": action_str,
            "loop_blocked": bool(loop_detected),
            "loop_reason": loop_reason or "",
            "prev_obs_at_decision": str(prev_obs_at_decision or "")[:2000],
            "history_tail": [
                {"obs": str(o)[:300], "act": a}
                for o, a in list(env_self._react_history)[-4:]
            ],
            "result_obs": str(obs)[:2000],
            "reward": float(reward) if reward is not None else 0.0,
            "done": bool(done),
            "admissible_after": [str(a) for a in avail[:60]],
            "supervisor_input_tail": (turn.supervisor_input or "")[-1500:] if turn.supervisor_input else "",
            "supervisor_output": (turn.supervisor_output or "")[:1500],
            "parse_error": bool(turn.parse_error),
            "info": {k: str(v)[:500] for k, v in (info or {}).items()},
        }
        if (
            _env_flag("REACT_TRACE_FULL_PROMPT", False)
            or (task_type == "alfworld" and _env_flag("ALFWORLD_TRACE_FULL_PROMPT", False))
            or (task_type == "webshop" and _env_flag("WEBSHOP_TRACE_FULL_PROMPT", False))
        ):
            record["supervisor_input"] = (turn.supervisor_input or "")[:20000]
        path = os.path.join(trace_dir, f"worker_pid{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _react_step(self, full_content, action_str, traj):

    from training.react_prompts import WEBSHOP_TEMPLATE, ALFWORLD_TEMPLATE

    self._step += 1


    _prev_obs_at_decision = self._current_obs


    turn = Turn(
        supervisor_input=self._current_prompt,
        supervisor_output=full_content or "",
        action_type=action_str.split("[")[0] if action_str and "[" in action_str else (action_str or "invalid"),
        instruction=action_str or "",
        parse_error=(action_str is None),
    )


    if action_str is None:
        turn.observation = "[INVALID] No valid <action> tag found."
        traj.turns.append(turn)
        _emit_alfworld_trace(self, turn, "<INVALID>", turn.observation,
                             0.0, False, {}, False, "parse_error",
                             _prev_obs_at_decision)

        return self._current_prompt, 0.0, False, {"observation": turn.observation}


    if str(action_str).strip().startswith("skill_invoke"):
        import re as _re
        m = _re.match(r"skill_invoke\[(.*?)\]", str(action_str).strip())
        if not m:
            m = _re.match(r"skill_invoke\((.*?)\)", str(action_str).strip())
        raw_sid = (m.group(1) if m else "").strip().strip("\"'")
        if raw_sid.startswith("skill_id="):
            raw_sid = raw_sid.split("=", 1)[1].strip().strip("\"'")
        obs = self._handle_skill_invoke({"skill_id": raw_sid})
        turn.action_type = "skill_invoke"
        turn.skill_id = raw_sid
        turn.instruction = raw_sid
        turn.observation = obs
        traj.turns.append(turn)
        self._react_history.append((self._current_obs, action_str))
        self._current_obs = str(obs)
        avail = self._ragen_adapter.available_actions if self._ragen_adapter else []
        next_prompt = _build_react_prompt(self, avail)
        self._current_prompt = next_prompt
        return next_prompt, 0.0, False, {"observation": obs, "skill_id": raw_sid}


    invalid_not_admissible = False
    semantic_blocked = False
    semantic_feedback = ""
    if self._task_type == "alfworld" and self._ragen_adapter is not None:
        try:
            avail_now = [str(a) for a in (self._ragen_adapter.available_actions or []) if str(a) != "help"]
        except Exception:
            avail_now = []
        canonical_action = _canonicalize_alfworld_action(action_str)
        if _alfworld_canonicalize_action_enabled() and canonical_action in avail_now:
            action_str = canonical_action
            turn.instruction = canonical_action
            turn.action_type = canonical_action.split(" ", 1)[0] if canonical_action else turn.action_type
        elif _alfworld_invalid_action_feedback_enabled() and action_str not in avail_now:
            invalid_not_admissible = True
        if not invalid_not_admissible and _alfworld_semantic_guard_enabled():
            semantic_feedback = _build_alfworld_semantic_action_feedback(self, action_str, avail_now)
            semantic_blocked = bool(semantic_feedback)


    loop_detected = False
    loop_reason = ""
    if _alfworld_loop_guard_enabled() and self._react_history:
        last_obs, last_act = self._react_history[-1]
        if last_act == action_str:
            last_obs_str = str(last_obs)
            if ("Nothing happens" in last_obs_str
                or "[REPEATED]" in last_obs_str
                or "[NO_PROGRESS]" in last_obs_str):
                loop_detected = True
                loop_reason = "no_progress_repeat"
            elif len(self._react_history) >= 2:
                _, prev_act = self._react_history[-2]
                if prev_act == action_str:
                    loop_detected = True
                    loop_reason = "triple_repeat"

    if invalid_not_admissible:
        try:
            avail_now = [str(a) for a in (self._ragen_adapter.available_actions or []) if str(a) != "help"]
        except Exception:
            avail_now = []
        obs = _build_alfworld_invalid_action_feedback(self, action_str, avail_now)
        reward, done, info = 0.0, False, {"invalid_not_admissible": True}
    elif semantic_blocked:
        obs = semantic_feedback
        reward, done, info = 0.0, False, {"semantic_blocked": True}
    elif loop_detected:
        obs = (
            f"[NO_PROGRESS] The previous action did not change the environment ({loop_reason}). "
            f"Environment state remains the same."
        )
        reward, done, info = 0.0, False, {"loop_blocked": True}
    else:

        try:
            obs, reward, done, info = self._ragen_adapter.step(action_str)
        except Exception as e:
            logger.warning(f"[ReAct] env.step failed: {e}")
            obs, reward, done, info = f"[ERROR] {e}", 0.0, False, {}

    if self._task_type == "alfworld":
        obs = _append_alfworld_neutral_env_feedback(self, obs, action_str, info)
    elif self._task_type == "webshop":
        obs = _append_webshop_neutral_env_feedback(self, obs, action_str, info)

    turn.observation = str(obs)
    traj.turns.append(turn)


    _emit_alfworld_trace(self, turn, action_str, obs, reward, done, info,
                         loop_detected, loop_reason, _prev_obs_at_decision)


    self._react_history.append((self._current_obs, action_str))

    if self._task_type == "webshop":
        self._current_obs = str(obs)
    else:
        self._current_obs = str(obs)


    if done:
        self._env_done = True
        self._env_reward = reward
        if self._task_type == "webshop":


            graded = float(reward or 0.0)
            traj.reward = graded
            traj.answer_reward = graded
            traj.r_tilde = max(graded + self.epsilon_min, self.epsilon_min)
        elif reward > 0:
            if str(self.reward_mode).lower() in {"outcome_only", "paper", "outcome"}:
                traj.reward = 1.0
            else:
                traj.reward = 1.0 + 0.1  
            traj.answer_reward = 1.0
            traj.r_tilde = max(traj.reward + self.epsilon_min, self.epsilon_min)
        else:
            traj.reward = 0.0
            traj.answer_reward = 0.0
            traj.r_tilde = self.epsilon_min
        traj.completed = True
        return "", traj.reward, True, info


    if self._step >= self.max_episode_steps:
        traj.reward = 0.0
        traj.answer_reward = 0.0
        traj.r_tilde = self.epsilon_min
        traj.completed = True
        traj.truncated = True
        return "", 0.0, True, {"truncated": True}


    avail = self._ragen_adapter.available_actions if self._ragen_adapter else []
    next_prompt = _build_react_prompt(self, avail)
    self._current_prompt = next_prompt

    return next_prompt, 0.0, False, {"observation": obs}


_ALFWORLD_DECISION_REMINDERS = """Decision reminders (apply now):
- Output exactly one string from the admissible-actions list. If the desired action is absent, navigate/open first; never invent it.
- Target class is fixed by the task. Match the exact object class only: `knife` is NOT `butterknife`, `pot` is NOT `pan`.
- Never take/heat/cool/clean a different class even if it is admissible.
- If the current observation says a cabinet/drawer/fridge/microwave is closed and `open ...` is listed, open it before leaving.
- If you are holding a wrong object, first put it down with an exact admissible `move <wrong> to <receptacle>` action; do not try to take the target while your hand is occupied.
- If current observation lists the target class AND your hand is free, take it now. If your hand is occupied by the target, continue the required state/destination sequence.
- If a receptacle was already visited and did not contain the target, do NOT bounce back to it; choose an unvisited admissible source (open closed cabinets/drawers if needed).
- For desklamp/examine tasks, once holding the target, keep holding it; do NOT move/drop it. Search unvisited likely lamp locations until exact `use desklamp ...` is listed.
- For clean/cool/heat tasks: first hold the exact target, then do exactly the required state action once. After that state is done, never clean/cool/heat it again; immediately move it to the destination.
- For count>1 tasks, after one successful move, find the next SAME-CLASS instance before completing.
"""


def _parse_alfworld_task(task_desc):

    import re
    out = {'target_class': None, 'dest_class': None, 'source_hint': None,
           'verb': None, 'count': 1, 'examine_with_desklamp': False}
    if not task_desc:
        return out
    s = task_desc.strip().lower().rstrip('.')

    m = re.search(r'pick up (?:the |a |some )?(\S+?)(?: \d)? from (\S+?)(?: \d)? and put it (?:in/on|in|on) (\S+?)(?: \d)?$', s)
    if m:
        out.update(target_class=m.group(1), source_hint=m.group(2), dest_class=m.group(3), verb=None)
        return out

    m = re.search(r'examine (?:the |a |some )?(\S+?)(?: \d)? with (?:the |a )?desklamp', s)
    if m:
        out.update(target_class=m.group(1), examine_with_desklamp=True, verb='examine_desklamp')
        return out

    m = re.search(r'look at (?:the |a |some )?(\S+?)(?: \d)? (?:under|with|by) (?:the |a )?(?:desklamp|lamp)', s)
    if m:
        out.update(target_class=m.group(1), examine_with_desklamp=True, verb='examine_desklamp')
        return out

    m = re.search(r'(heat|cool|clean) (?:some |a |the )?(\S+?)(?: \d)? and put it (?:in/on|in|on) (\S+?)(?: \d)?$', s)
    if m:
        out.update(verb=m.group(1), target_class=m.group(2), dest_class=m.group(3))
        return out

    m = re.search(r'(?:find|put) two (\S+?)(?: \d)? (?:and put them )?(?:in/on|in|on) (\S+?)(?: \d)?$', s)
    if m:
        out.update(target_class=m.group(1), count=2, dest_class=m.group(2), verb=None)
        return out


    m = re.search(r'put (?:a |some |the )?(?:(clean|washed|cool|cold|hot|heated|cooked) )?(\S+?)(?: \d)? (?:in/on|in|on) (\S+?)(?: \d)?$', s)
    if m:
        adjective, obj, dest = m.group(1), m.group(2), m.group(3)
        verb = None
        if adjective in ("clean", "washed"):
            verb = "clean"
        elif adjective in ("cool", "cold"):
            verb = "cool"
        elif adjective in ("hot", "heated", "cooked"):
            verb = "heat"
        out.update(target_class=obj, dest_class=dest, verb=verb)
        return out

    return out


def _canonicalize_alfworld_action(action):

    if action is None:
        return None
    import re
    s = str(action).strip().strip('"').strip("'").strip()
    if s.startswith(">"):
        s = s[1:].strip()
    if s.lower().startswith("action:"):
        s = s[7:].strip()
    m = re.match(r'^put\s+(.+?)\s+(?:in|on|into|onto)\s+(.+)$', s, flags=re.IGNORECASE)
    if m:
        return f"move {m.group(1).strip()} to {m.group(2).strip()}"
    m = re.match(r'^pick up\s+(.+?)\s+from\s+(.+)$', s, flags=re.IGNORECASE)
    if m:
        return f"take {m.group(1).strip()} from {m.group(2).strip()}"
    return s


def _alfworld_object_class(obj_name):

    if not obj_name:
        return ""
    import re
    s = str(obj_name).lower().strip()
    s = re.sub(r'^(?:a|an|the|some)\s+', '', s)
    s = re.sub(r'\s+\d+\s*$', '', s)
    return s.strip()


def _target_matches_obj(obj_name, target_class):
    if not obj_name or not target_class:
        return False
    return _alfworld_object_class(obj_name) == str(target_class).lower().strip()


def _alfworld_action_object(action):

    if not action:
        return ""
    import re
    s = str(action).strip()
    for pat in (
        r'^take\s+(.+?)\s+from\s+',
        r'^move\s+(.+?)\s+to\s+',
        r'^(?:clean|heat|cool)\s+(.+?)\s+with\s+',
        r'^use\s+(.+?)\s+',
    ):
        m = re.match(pat, s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _alfworld_action_move_dest(action):
    if not action:
        return ""
    import re
    m = re.match(r'^move\s+(.+?)\s+to\s+(.+)$', str(action).strip(), flags=re.IGNORECASE)
    return m.group(2).strip() if m else ""


def _alfworld_action_take_source(action):
    if not action:
        return ""
    import re
    m = re.match(r'^take\s+(.+?)\s+from\s+(.+)$', str(action).strip(), flags=re.IGNORECASE)
    return m.group(2).strip() if m else ""


def _alfworld_action_mentions_target(action, target_class):
    obj = _alfworld_action_object(action)
    return bool(obj and _target_matches_obj(obj, target_class))


def _observation_mentions_object_class(text, target_class):
    if not text or not target_class:
        return False
    import re
    t = re.escape(str(target_class).lower().strip())

    return re.search(rf'\b{t}\s+\d+\b', str(text).lower()) is not None


def _current_closed_open_options(current_obs, avail_actions):

    if not current_obs:
        return []
    import re
    obs = str(current_obs).lower()
    closed = []
    for m in re.finditer(r'\bthe\s+([a-z0-9]+(?:\s+\d+)?)\s+is\s+closed\b', obs):
        recep = m.group(1).strip()
        if recep and recep not in closed:
            closed.append(recep)
    opens = []
    avail = [str(a) for a in (avail_actions or [])]
    for recep in closed:
        cand = f"open {recep}"
        if cand in avail:
            opens.append(cand)
    return opens


def _alfworld_navigation_options(avail_actions, limit=12):
    return [str(a) for a in (avail_actions or []) if str(a).startswith("go to ")][:limit]


def _alfworld_open_options(avail_actions, limit=8):
    return [str(a) for a in (avail_actions or []) if str(a).startswith("open ")][:limit]


def _alfworld_source_priority(target_class, dest_class=None):

    t = str(target_class or "").lower().strip()
    d = str(dest_class or "").lower().strip()
    kitchenware = {
        "pot", "pan", "kettle",
        "knife", "butterknife", "fork", "spoon", "spatula", "ladle",
        "plate", "bowl", "mug", "cup",
        "glassbottle", "winebottle", "saltshaker", "peppershaker",
        "dishsponge", "soapbottle", "papertowelroll",
    }
    food = {"apple", "bread", "egg", "lettuce", "tomato", "potato"}
    living = {
        "book", "remotecontrol", "cellphone", "creditcard", "keychain",
        "pen", "pencil", "watch", "statue", "vase", "pillow", "box",
        "laptop", "alarmclock", "cd", "newspaper", "tissuebox",
        "spraybottle",
    }
    if t == "pillow":
        return ["bed", "sofa", "armchair", "dresser", "sidetable", "shelf", "drawer", "cabinet"]
    if t == "newspaper":
        return ["coffeetable", "sofa", "armchair", "diningtable", "sidetable", "dresser", "desk", "shelf", "drawer", "cabinet"]
    if t == "tissuebox":


        return ["countertop", "sinkbasin", "toilet", "bathtubbasin", "dresser", "desk", "shelf", "sidetable", "coffeetable", "drawer", "cabinet"]
    if t == "toiletpaper":
        return ["toiletpaperhanger", "countertop", "toilet", "bathtubbasin", "sinkbasin", "shelf", "dresser", "cabinet", "drawer"]
    if t == "laptop":


        return ["dresser", "desk", "sidetable", "bed", "sofa", "armchair", "safe", "shelf", "drawer", "cabinet"]
    if t == "cellphone":
        return ["dresser", "sidetable", "bed", "desk", "sofa", "armchair", "safe", "shelf", "drawer", "cabinet"]
    if t == "book":
        return ["desk", "sidetable", "coffeetable", "dresser", "bed", "shelf", "cabinet", "drawer"]
    if t in {"pen", "pencil"}:
        return ["desk", "sidetable", "dresser", "shelf", "bed", "coffeetable", "drawer", "safe", "cabinet"]
    if t in {"creditcard", "keychain", "watch"}:
        return ["sidetable", "dresser", "desk", "shelf", "bed", "coffeetable", "sofa", "armchair", "drawer", "safe", "cabinet"]
    if t == "soapbar":
        return ["sinkbasin", "countertop", "toilet", "bathtubbasin", "garbagecan", "cabinet", "drawer", "shelf"]
    if t == "candle":
        return ["countertop", "toilet", "bathtubbasin", "garbagecan", "sidetable", "diningtable", "shelf", "dresser", "drawer", "cabinet"]
    if t in {"pot", "pan", "kettle"}:


        primary = []
        if d in {"stoveburner", "sinkbasin", "countertop"}:
            primary.append(d)
        for cls in ("stoveburner", "sinkbasin", "countertop"):
            if cls not in primary:
                primary.append(cls)
        return primary + ["cabinet", "drawer", "shelf", "diningtable"]
    if t in {"knife", "butterknife", "fork", "spoon", "spatula", "ladle"}:
        return ["diningtable", "countertop", "sinkbasin", "drawer", "cabinet", "stoveburner", "microwave"]
    if t in {"saltshaker", "peppershaker"}:
        return ["countertop", "diningtable", "drawer", "cabinet", "shelf"]
    if t in {"mug", "cup"} and d in {"desk", "sidetable", "coffeetable", "shelf"}:
        return ["sidetable", "desk", "coffeetable", "shelf", "drawer", "cabinet", "countertop", "diningtable", "sinkbasin"]
    if t in {"plate", "bowl", "mug", "cup", "glassbottle", "winebottle"}:
        return ["countertop", "sinkbasin", "diningtable", "cabinet", "fridge", "microwave", "shelf"]
    if t in food:
        return ["fridge", "countertop", "diningtable", "sinkbasin", "stoveburner", "toaster", "microwave", "cabinet", "drawer"]
    if t in {"dishsponge", "soapbottle", "spraybottle"}:


        return ["sinkbasin", "countertop", "garbagecan", "cabinet", "shelf", "drawer"]
    if t in living:
        return ["sidetable", "coffeetable", "desk", "dresser", "sofa", "armchair", "shelf", "drawer"]
    if t in kitchenware:
        return ["countertop", "sinkbasin", "diningtable", "cabinet", "drawer", "shelf"]
    return ["countertop", "diningtable", "sinkbasin", "sidetable", "coffeetable", "shelf", "drawer", "cabinet"]


def _rank_alfworld_unvisited_go_options(
    target_class,
    avail_actions,
    visited_receptacles,
    limit=10,
    dest_class=None,
    dest_penalty=None,
):
    import re
    gos = [str(a) for a in (avail_actions or []) if str(a).startswith("go to ")]
    visited = set(visited_receptacles or [])
    unvisited = [g for g in gos if g[6:].strip() not in visited]
    pri = _alfworld_source_priority(target_class, dest_class=dest_class)
    pri_index = {name: i for i, name in enumerate(pri)}
    dest = str(dest_class or "").lower().strip()

    def _score(go_action):
        recep = go_action[6:].strip()
        cls = _alfworld_object_class(recep)
        base = pri_index.get(cls, len(pri) + 5)


        if dest and cls == dest:
            if dest_penalty is None:


                source_rank = pri_index.get(cls, len(pri) + 5)
                penalty = 0 if source_rank <= 2 else (4 if source_rank <= 4 else 8)
            else:
                penalty = dest_penalty
            base += penalty
        m = re.search(r'\s+(\d+)$', recep)
        num = int(m.group(1)) if m else 99
        return (base, num, recep)

    return sorted(unvisited, key=_score)[:limit]


def _rank_alfworld_lamp_search_go_options(avail_actions, visited_receptacles, limit=8):

    import re

    gos = [str(a) for a in (avail_actions or []) if str(a).startswith("go to ")]
    if isinstance(visited_receptacles, dict):
        visited = set(visited_receptacles.keys())
    else:
        visited = set(visited_receptacles or [])
    unvisited = [g for g in gos if g[6:].strip() not in visited]
    priority = [
        "desklamp", "sidetable", "desk", "dresser", "shelf", "countertop",
        "coffeetable", "diningtable", "bed", "sofa", "armchair",
        "cabinet", "drawer", "garbagecan",
    ]
    pri_index = {name: i for i, name in enumerate(priority)}

    def _score(go_action):
        recep = go_action[6:].strip()
        cls = _alfworld_object_class(recep)
        m = re.search(r'\s+(\d+)$', recep)
        num = int(m.group(1)) if m else 99
        return (pri_index.get(cls, len(priority) + 5), num, recep)

    return sorted(unvisited, key=_score)[:limit]


def _alfworld_source_focus_classes(target_class, dest_class=None):

    t = str(target_class or "").lower().strip()
    d = str(dest_class or "").lower().strip()
    if t in {"pot", "pan", "kettle"}:
        focus = []
        if d in {"stoveburner", "sinkbasin", "countertop"}:
            focus.append(d)
        for cls in ("stoveburner", "sinkbasin", "countertop"):
            if cls not in focus:
                focus.append(cls)
        return focus
    if t == "pillow":
        return ["bed", "sofa", "armchair"]
    if t == "newspaper":
        return ["coffeetable", "sofa", "armchair", "diningtable", "sidetable", "dresser", "desk"]
    if t == "book":
        return ["desk", "sidetable", "coffeetable", "dresser", "bed", "shelf", "cabinet"]
    if t == "tissuebox":
        return ["countertop", "sinkbasin", "toilet", "bathtubbasin", "dresser", "desk", "shelf", "sidetable"]
    if t == "toiletpaper":
        return ["toiletpaperhanger", "countertop", "toilet", "bathtubbasin", "sinkbasin"]
    if t in {"cellphone", "laptop"}:
        return ["dresser", "desk", "sidetable", "bed", "sofa", "armchair", "safe"]
    if t in {"pen", "pencil", "creditcard", "keychain", "watch"}:
        return ["desk", "sidetable", "dresser", "shelf", "bed", "coffeetable", "sofa", "armchair"]
    if t in {"dishsponge", "soapbottle", "spraybottle"}:
        return ["sinkbasin", "countertop", "garbagecan"]
    if t in {"apple", "bread", "egg", "lettuce", "tomato", "potato"}:
        return ["fridge", "countertop", "diningtable", "sinkbasin", "stoveburner", "toaster", "microwave"]
    if t in {"plate", "bowl", "mug", "cup", "glassbottle", "winebottle"}:
        if t in {"mug", "cup"} and d in {"desk", "sidetable", "coffeetable", "shelf"}:
            return ["sidetable", "desk", "coffeetable", "shelf", "drawer", "cabinet"]
        return ["countertop", "sinkbasin", "diningtable"]
    if t in {"saltshaker", "peppershaker"}:
        return ["countertop", "diningtable", "drawer"]
    if t in {"knife", "butterknife", "fork", "spoon", "spatula", "ladle"}:
        return ["diningtable", "countertop", "sinkbasin", "drawer"]
    return []


def _known_receptacles_with_object(visited_receptacles, obj_class, limit=6):
    out = []
    for recep, info in (visited_receptacles or {}).items():
        obs = str((info or {}).get("last_obs_short", "") or "")
        if _observation_mentions_object_class(obs, obj_class):
            out.append(recep)
    return out[:limit]


def _extract_held_objects_from_actions(avail_actions, target_class=None):

    import re
    held = []
    seen = set()
    for a in avail_actions or []:
        s = str(a)
        m = re.match(r'^(?:move|clean|heat|cool)\s+(.+?)\s+(?:to|with)\s+', s, flags=re.IGNORECASE)
        if m:
            obj = m.group(1).strip()
            if obj not in seen:
                held.append(obj)
                seen.add(obj)


        if target_class:
            m = re.match(r'^examine\s+(.+?)$', s, flags=re.IGNORECASE)
            if m:
                obj = m.group(1).strip()
                if _target_matches_obj(obj, target_class) and obj not in seen:
                    held.append(obj)
                    seen.add(obj)
    return held


def _alfworld_state_action_done(env_self, parsed):

    verb = parsed.get("verb")
    target = parsed.get("target_class")
    if verb not in ("clean", "heat", "cool") or not target:
        return False
    hay = [str(getattr(env_self, "_current_obs", "") or "")]
    for obs, _act in getattr(env_self, "_react_history", []) or []:
        hay.append(str(obs or ""))
    needle = {
        "clean": "clean",
        "heat": "heat",
        "cool": "cool",
    }[verb]
    for text in hay:
        low = text.lower()
        if f"you {needle} the {target}" in low or f"you {needle} {target}" in low:
            return True
    return False


def _alfworld_exact_options(avail_actions, prefix=None, contains=None, limit=8):
    out = []
    for a in avail_actions or []:
        s = str(a)
        low = s.lower()
        if prefix and not low.startswith(prefix):
            continue
        if contains and contains not in low:
            continue
        out.append(s)
    return out[:limit]


def _alfworld_placed_target_records(env_self, target_class, dest_class=None):

    if not target_class:
        return []
    records = []
    seen = set()
    for _obs, act in getattr(env_self, "_react_history", []) or []:
        obj = _alfworld_action_object(act)
        dest = _alfworld_action_move_dest(act)
        if not obj or not dest:
            continue
        if not _target_matches_obj(obj, target_class):
            continue
        if dest_class and _alfworld_object_class(dest) != str(dest_class).lower().strip():
            continue
        key = (obj.lower(), dest.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append((obj, dest))
    return records


def _alfworld_action_verb(action):
    if not action:
        return ""
    return str(action).strip().split(" ", 1)[0].lower()


def _build_alfworld_semantic_action_feedback(env_self, action_str, avail_actions):

    parsed = _parse_alfworld_task(getattr(env_self, "_task_description", "") or "")
    target = parsed.get("target_class")
    dest = parsed.get("dest_class")
    required_state = parsed.get("verb")
    if not target:
        return None

    avail = [str(a) for a in (avail_actions or []) if str(a) != "help"]
    action = str(action_str or "").strip()
    action_verb = _alfworld_action_verb(action)
    obj = _alfworld_action_object(action)
    obj_is_target = bool(obj and _target_matches_obj(obj, target))
    held = _extract_held_objects_from_actions(avail, target_class=target)
    held_target = [h for h in held if _target_matches_obj(h, target)]
    state_done = _alfworld_state_action_done(env_self, parsed)
    count = int(parsed.get("count", 1) or 1)
    placed_records = _alfworld_placed_target_records(env_self, target, dest)
    placed_objs = {obj.lower() for obj, _dest in placed_records}
    preferred_dest_instance = placed_records[0][1] if count > 1 and placed_records else ""

    def _append_best_next(lines):
        if parsed.get("examine_with_desklamp"):
            lamp_use = [a for a in avail if a.lower().startswith("use desklamp")]
            if lamp_use:
                lines.append("Correct next action: " + " | ".join(lamp_use[:4]))
            else:
                try:
                    visited = _extract_visited_receptacles(
                        getattr(env_self, "_react_history", []) or [],
                        str(getattr(env_self, "_current_obs", "") or ""),
                    )
                except Exception:
                    visited = {}
                lamp_search = _rank_alfworld_lamp_search_go_options(avail, visited, limit=6)
                if lamp_search:
                    lines.append("Keep holding the target and search unvisited lamp locations: " + " | ".join(lamp_search[:6]))
            return

        if held_target:
            target_obj = held_target[0]
        elif obj_is_target:
            target_obj = obj
        else:
            target_obj = ""

        if required_state in ("clean", "cool", "heat") and not state_done and target_obj:
            state_opts = _alfworld_exact_options(avail, prefix=f"{required_state} {target_obj.lower()} with ", limit=4)
            if state_opts:
                lines.append("Correct next action is the required state action: " + " | ".join(state_opts))
                return
            appliance = {"clean": "sinkbasin", "cool": "fridge", "heat": "microwave"}.get(required_state)
            nav_opts = [a for a in _alfworld_navigation_options(avail, limit=16) if appliance and appliance in a.lower()]
            open_opts = _current_closed_open_options(str(getattr(env_self, "_current_obs", "") or ""), avail)
            if open_opts:
                lines.append("Open the current closed receptacle/appliance first: " + " | ".join(open_opts[:4]))
            elif nav_opts:
                lines.append(f"Go to the required {appliance} first: " + " | ".join(nav_opts[:6]))
            return

        if dest and target_obj:
            if preferred_dest_instance:
                move_opts = [
                    a for a in avail
                    if a.lower() == f"move {target_obj.lower()} to {preferred_dest_instance.lower()}"
                ]
            else:
                move_opts = [a for a in avail if a.lower().startswith(f"move {target_obj.lower()} to ") and dest.lower() in a.lower()]
            if move_opts:
                lines.append("Correct next action is the destination move: " + " | ".join(move_opts[:4]))
                return
            if preferred_dest_instance:
                dest_nav = [a for a in _alfworld_navigation_options(avail, limit=16) if a[6:].strip().lower() == preferred_dest_instance.lower()]
            else:
                dest_nav = [a for a in _alfworld_navigation_options(avail, limit=16) if dest.lower() in a.lower()]
            if dest_nav:
                lines.append("Navigate to the destination first: " + " | ".join(dest_nav[:6]))
                return

        take_opts = _alfworld_exact_options(avail, prefix=f"take {target.lower()} ", limit=6)
        if placed_objs:
            take_opts = [a for a in take_opts if _alfworld_action_object(a).lower() not in placed_objs]
        if take_opts:
            lines.append("Correct target take actions: " + " | ".join(take_opts[:6]))

    if parsed.get("examine_with_desklamp") and held_target:
        lamp_use = [a for a in avail if a.lower().startswith("use desklamp")]
        if lamp_use and not action.lower().startswith("use desklamp"):
            lines = [
                f"[ACTION_BLOCKED_SEMANTIC] `{action}` would leave/ignore an available desklamp while holding the target.",
                "This task is complete only by using the lamp while still holding the exact target.",
            ]
            _append_best_next(lines)
            lines.append("Next response must be exactly one admissible action string that follows the task.")
            return "\n".join(lines)

    if count > 1 and obj_is_target and action_verb == "take" and obj.lower() in placed_objs:
        lines = [
            f"[ACTION_BLOCKED_SEMANTIC] `{action}` would take back `{obj}`, which was already placed for this count=2 task.",
            "Do not remove already placed target objects; find a different same-class instance.",
        ]
        _append_best_next(lines)
        lines.append("Next response must be exactly one admissible action string that follows the task.")
        return "\n".join(lines)

    if count > 1 and obj_is_target and action_verb == "move" and preferred_dest_instance:
        move_dest = _alfworld_action_move_dest(action)
        if move_dest and move_dest.lower() != preferred_dest_instance.lower():
            lines = [
                f"[ACTION_BLOCKED_SEMANTIC] `{action}` would scatter count-task objects across destinations.",
                f"The first target was placed in `{preferred_dest_instance}`; every later `{target}` must go to that same destination instance.",
            ]
            _append_best_next(lines)
            lines.append("Next response must be exactly one admissible action string that follows the task.")
            return "\n".join(lines)


    if obj and not obj_is_target and action_verb in ("take", "clean", "cool", "heat"):
        lines = [
            f"[ACTION_BLOCKED_SEMANTIC] `{action}` is admissible, but it manipulates `{_alfworld_object_class(obj)}`.",
            f"The task target class is exactly `{target}`. Do not switch classes.",
        ]
        _append_best_next(lines)
        lines.append("Next response must be exactly one admissible action string that follows the task.")
        return "\n".join(lines)

    if parsed.get("examine_with_desklamp") and obj_is_target and action_verb in ("move", "clean", "cool", "heat"):
        lines = [
            f"[ACTION_BLOCKED_SEMANTIC] `{action}` would stop the desklamp task sequence.",
            "For desklamp/examine tasks, once holding the target, keep holding it and search until exact `use desklamp ...` is admissible.",
        ]
        _append_best_next(lines)
        lines.append("Next response must be exactly one admissible action string that follows the task.")
        return "\n".join(lines)

    if obj_is_target and action_verb in ("clean", "cool", "heat"):
        if required_state not in ("clean", "cool", "heat"):
            lines = [
                f"[ACTION_BLOCKED_SEMANTIC] `{action}` is unnecessary for this place/examine task.",
                "Do not apply clean/cool/heat unless the task explicitly requires that state.",
            ]
            _append_best_next(lines)
            lines.append("Next response must be exactly one admissible action string that follows the task.")
            return "\n".join(lines)
        if state_done:
            lines = [
                f"[ACTION_BLOCKED_SEMANTIC] `{action}` would apply another state change after the required `{required_state}` is already done.",
                "Do not clean/cool/heat the target again; move it to the destination now.",
            ]
            _append_best_next(lines)
            lines.append("Next response must be exactly one admissible action string that follows the task.")
            return "\n".join(lines)
        if action_verb != required_state:
            lines = [
                f"[ACTION_BLOCKED_SEMANTIC] `{action}` is the wrong state action for this task.",
                f"The task requires `{required_state}` exactly, not `{action_verb}`.",
            ]
            _append_best_next(lines)
            lines.append("Next response must be exactly one admissible action string that follows the task.")
            return "\n".join(lines)

    if (
        obj_is_target
        and action_verb == "move"
        and required_state in ("clean", "cool", "heat")
        and not state_done
    ):
        lines = [
            f"[ACTION_BLOCKED_SEMANTIC] `{action}` would place the target before the required `{required_state}` state is done.",
            f"First perform the exact `{required_state} {obj} with ...` action while holding the target.",
        ]
        _append_best_next(lines)
        lines.append("Next response must be exactly one admissible action string that follows the task.")
        return "\n".join(lines)

    return None


def _build_alfworld_decision_state_block(env_self, avail_actions):

    parsed = _parse_alfworld_task(getattr(env_self, "_task_description", "") or "")
    target = parsed.get("target_class")
    dest = parsed.get("dest_class")
    verb = parsed.get("verb")
    count = parsed.get("count", 1)
    avail = [str(a) for a in (avail_actions or []) if str(a) != "help"]
    held = _extract_held_objects_from_actions(avail, target_class=target)
    held_target = [h for h in held if _target_matches_obj(h, target)]
    held_wrong = [h for h in held if target and not _target_matches_obj(h, target)]
    state_done = _alfworld_state_action_done(env_self, parsed)
    appliance = {"clean": "sinkbasin", "cool": "fridge", "heat": "microwave"}.get(verb)
    placed_records = _alfworld_placed_target_records(env_self, target, dest)
    placed_objs = {obj.lower() for obj, _dest in placed_records}
    preferred_dest_instance = placed_records[0][1] if count and count > 1 and placed_records else ""
    current_obs = str(getattr(env_self, "_current_obs", "") or "")
    closed_open_opts = _current_closed_open_options(current_obs, avail)
    visited_receptacles = {}
    try:
        visited_receptacles = _extract_visited_receptacles(getattr(env_self, "_react_history", []) or [], current_obs)
    except Exception:
        visited_receptacles = {}
    ranked_unvisited_go = (
        _rank_alfworld_unvisited_go_options(
            target,
            avail,
            set(visited_receptacles.keys()),
            limit=10,
            dest_class=dest,
            dest_penalty=(14 if count > 1 and placed_records else None),
        )
        if target else []
    )
    source_focus_classes = _alfworld_source_focus_classes(target, dest)
    focused_unvisited_go = [
        a for a in ranked_unvisited_go
        if _alfworld_object_class(a[6:].strip()) in set(source_focus_classes)
    ] if source_focus_classes else []

    lines = ["[CURRENT TASK STATE — obey this over generic habits]"]
    if target or dest or verb or count:
        req = []
        if verb in ("clean", "cool", "heat"):
            req.append(f"required_state={verb} via {appliance}")
        elif parsed.get("examine_with_desklamp"):
            req.append("required_action=use desklamp while holding target")
        else:
            req.append("required_state=none/place-only")
        lines.append(
            f"Parsed: target_class={target or '?'}; destination={dest or '?'}; "
            f"count={count}; " + "; ".join(req)
        )
    if count and count > 1:
        lines.append(
            f"Count progress: placed {min(len(placed_records), count)}/{count}"
            + (
                f"; already_placed={', '.join(obj for obj, _d in placed_records)}; "
                f"same_destination_instance_required={preferred_dest_instance}"
                if preferred_dest_instance else "; no destination instance chosen yet"
            )
        )
    if held:
        lines.append(f"Likely holding: {', '.join(held)}")
    else:
        lines.append("Likely holding: nothing / hand free")

    if held_wrong:
        drop_opts = []
        for obj in held_wrong:
            drop_opts.extend(_alfworld_exact_options(avail, prefix=f"move {obj.lower()} to ", limit=4))
        lines.append(
            "CRITICAL: you are holding a WRONG object. First drop it with an exact admissible move action; "
            "do not take/clean/heat/cool/move non-target objects."
        )
        if drop_opts:
            lines.append("Exact drop options now: " + " | ".join(drop_opts[:6]))
    elif target and not held_target:
        take_opts = _alfworld_exact_options(avail, prefix=f"take {target.lower()} ", limit=6)
        if placed_objs:
            take_opts = [a for a in take_opts if _alfworld_action_object(a).lower() not in placed_objs]
        if take_opts:
            lines.append("Hand is free and target is visible. NEXT ACTION should be one of: " + " | ".join(take_opts))
        elif count and count > 1 and placed_objs and any(
            _alfworld_action_object(a).lower() in placed_objs
            for a in _alfworld_exact_options(avail, prefix=f"take {target.lower()} ", limit=10)
        ):
            lines.append(
                "Only already-placed target instances are takeable here. Do NOT take them back; "
                "continue searching for a different same-class instance."
            )
        elif closed_open_opts:
            lines.append(
                "Current receptacle is closed and may contain the target. "
                "NEXT ACTION should open it before leaving: " + " | ".join(closed_open_opts[:4])
            )
        else:
            lines.append("Hand is free but target is not directly takeable here. Explore/open a source receptacle; do not take other classes.")
            if verb in ("cool", "heat") and appliance and source_focus_classes:
                lines.append(
                    f"Do NOT go to the {appliance} merely for `{verb}` yet; "
                    f"first find and take an exact `{target}`. The state action comes after holding it."
                )
            if focused_unvisited_go:
                lines.append(
                    "CRITICAL SOURCE PRIORITY: for this target, exhaust these high-yield source classes "
                    f"before any cabinet/drawer/shelf/fridge/microwave: {', '.join(source_focus_classes)}. "
                    f"NEXT ACTION should be: {focused_unvisited_go[0]}"
                )
                lines.append(
                    "Priority unvisited source actions now: "
                    + " | ".join(focused_unvisited_go[:8])
                )
            elif ranked_unvisited_go:
                lines.append(
                    f"NEXT ACTION should be the first sensible unvisited source: {ranked_unvisited_go[0]}"
                )
                lines.append(
                    "High-priority UNVISITED source actions now (choose the first sensible one; do not return to visited places): "
                    + " | ".join(ranked_unvisited_go[:8])
                )
            open_opts = _alfworld_open_options(avail, limit=6)
            nav_opts = _alfworld_navigation_options(avail, limit=10)
            if open_opts:
                if focused_unvisited_go:
                    lines.append(
                        "Other open actions may be admissible, but do NOT open low-priority cabinets/drawers "
                        "until the priority source actions above are exhausted."
                    )
                else:
                    lines.append("Exact open options now: " + " | ".join(open_opts))
            if nav_opts:
                if focused_unvisited_go or ranked_unvisited_go:
                    lines.append(
                        "Other navigation actions are admissible but lower priority; avoid them until the "
                        "listed unvisited source actions have been tried."
                    )
                else:
                    lines.append("Exact navigation options now: " + " | ".join(nav_opts))
    elif held_target:
        target_obj = held_target[0]
        if parsed.get("examine_with_desklamp"):
            lamp_use_opts = [a for a in avail if a.lower().startswith("use desklamp")]
            if lamp_use_opts:
                lines.append(
                    "Holding target for a desklamp task. NEXT ACTION should be to turn on/use the lamp: "
                    + " | ".join(lamp_use_opts[:4])
                )
            else:
                lamp_locs = _known_receptacles_with_object(visited_receptacles, "desklamp", limit=6)
                if lamp_locs:
                    lamp_nav = [a for a in avail if a.startswith("go to ") and a[6:].strip() in set(lamp_locs)]
                    if lamp_nav:
                        lines.append(
                            "Holding target for a desklamp task. Go back to a known desklamp location: "
                            + " | ".join(lamp_nav[:4])
                        )
                    else:
                        lines.append(
                            "Holding target for a desklamp task. A desklamp was seen at: "
                            + ", ".join(lamp_locs)
                            + ". Navigate there until exact `use desklamp ...` is admissible."
                        )
                        lamp_search = _rank_alfworld_lamp_search_go_options(avail, visited_receptacles, limit=8)
                        if lamp_search:
                            lines.append(
                                "If that known lamp location is not directly reachable, keep holding the target and search unvisited lamp candidates: "
                                + " | ".join(lamp_search[:8])
                            )
                else:
                    lamp_nav = [a for a in avail if any(x in a.lower() for x in ("desk", "sidetable", "dresser", "coffeetable")) and a.startswith("go to ")]
                    if lamp_nav:
                        visited_set = set(visited_receptacles.keys())
                        unvisited_lamp_nav = [
                            a for a in lamp_nav
                            if a[6:].strip() not in visited_set
                        ]
                        if unvisited_lamp_nav:
                            already = ", ".join(sorted(visited_set)[:8]) if visited_set else "none"
                            lines.append(
                                "CRITICAL LAMP SEARCH: keep holding the target; do NOT move/drop it. "
                                f"Already visited without usable lamp: {already}. "
                                f"NEXT ACTION should be the first unvisited likely lamp location: {unvisited_lamp_nav[0]}"
                            )
                            lines.append(
                                "Unvisited likely lamp locations now: "
                                + " | ".join(unvisited_lamp_nav[:6])
                            )
                        else:
                            fallback_lamp_nav = _rank_alfworld_lamp_search_go_options(avail, visited_receptacles, limit=8)
                            if fallback_lamp_nav:
                                already = ", ".join(sorted(visited_set)[:8]) if visited_set else "none"
                                lines.append(
                                    "CRITICAL LAMP SEARCH: all desk/sidetable/dresser/coffeetable options here were already visited. "
                                    f"Already visited: {already}. Desklamps can also be on shelves. "
                                    f"NEXT ACTION should search the first unvisited place: {fallback_lamp_nav[0]}"
                                )
                                lines.append(
                                    "All unvisited lamp-search actions now: "
                                    + " | ".join(fallback_lamp_nav[:8])
                                )
                            else:
                                lines.append(
                                    "Holding target for a desklamp task. No unvisited lamp-search navigation remains; "
                                    "do NOT drop the target. Choose any reachable route toward a desklamp until exact `use desklamp ...` appears: "
                                    + " | ".join(lamp_nav[:6])
                                )
                    else:
                        fallback_lamp_nav = _rank_alfworld_lamp_search_go_options(avail, visited_receptacles, limit=8)
                        if fallback_lamp_nav:
                            lines.append(
                                "Holding target for a desklamp task. No named desk/sidetable route is available; "
                                "desklamps can be on shelves or other unvisited furniture. "
                                f"NEXT ACTION should be: {fallback_lamp_nav[0]}"
                            )
                            lines.append(
                                "All unvisited lamp-search actions now: "
                                + " | ".join(fallback_lamp_nav[:8])
                            )
                        else:
                            lines.append("Holding target for a desklamp task. Search for a visible desklamp; do NOT just examine the object.")
        elif verb in ("clean", "cool", "heat") and not state_done:
            state_opts = _alfworld_exact_options(avail, prefix=f"{verb} {target_obj.lower()} with ", limit=4)
            if state_opts:
                lines.append("Holding target but required state is NOT done. NEXT ACTION should be: " + " | ".join(state_opts))
            elif closed_open_opts:
                lines.append(
                    f"Holding target but required state is NOT done. Open the current closed receptacle/appliance first: "
                    + " | ".join(closed_open_opts[:4])
                )
            else:
                lines.append(f"Holding target but required state is NOT done. Go to/open {appliance} until exact `{verb} {target_obj} with ...` is admissible.")
        elif dest:
            if preferred_dest_instance:
                lines.append(
                    f"Count task: use the SAME destination instance as the first placed target: {preferred_dest_instance}."
                )
                move_opts = [
                    a for a in avail
                    if a.lower() == f"move {target_obj.lower()} to {preferred_dest_instance.lower()}"
                ]
            else:
                move_opts = [a for a in avail if a.lower().startswith(f"move {target_obj.lower()} to ") and dest.lower() in a.lower()]
            if move_opts:
                lines.append("Holding target and requirements are done. NEXT ACTION should be destination move: " + " | ".join(move_opts[:4]))
            elif closed_open_opts:
                lines.append("Destination/current receptacle is closed. NEXT ACTION should open it: " + " | ".join(closed_open_opts[:4]))
            else:
                if preferred_dest_instance:
                    lines.append(
                        f"Holding target. Go to/open the SAME destination `{preferred_dest_instance}` until exact "
                        f"`move {target_obj} to {preferred_dest_instance}` is admissible."
                    )
                    nav_opts = [
                        a for a in _alfworld_navigation_options(avail, limit=12)
                        if a[6:].strip().lower() == preferred_dest_instance.lower()
                    ]
                else:
                    lines.append(f"Holding target. Go to/open destination {dest} until exact `move {target_obj} to {dest} ...` is admissible.")
                    nav_opts = [a for a in _alfworld_navigation_options(avail, limit=12) if dest.lower() in a.lower()]
                if nav_opts:
                    lines.append("Exact destination navigation options now: " + " | ".join(nav_opts[:6]))


    exact = []
    exact_note = "Context-appropriate target actions now: "
    if target:
        if parsed.get("examine_with_desklamp"):
            if not held_target:
                exact = _alfworld_exact_options(avail, prefix=f"take {target.lower()} ", limit=6)
                if placed_objs:
                    exact = [a for a in exact if _alfworld_action_object(a).lower() not in placed_objs]
                exact_note = "Context-appropriate target take actions now: "
        elif held_target:
            target_obj = held_target[0]
            if verb in ("clean", "cool", "heat") and not state_done:
                exact = _alfworld_exact_options(avail, prefix=f"{verb} {target_obj.lower()} with ", limit=4)
                exact_note = f"Only valid target state actions now (required={verb}): "
            elif dest:
                if preferred_dest_instance:
                    exact = [
                        a for a in avail
                        if a.lower() == f"move {target_obj.lower()} to {preferred_dest_instance.lower()}"
                    ]
                    exact_note = f"Only valid same-destination move now ({preferred_dest_instance}): "
                else:
                    exact = [a for a in avail if a.lower().startswith(f"move {target_obj.lower()} to ") and dest.lower() in a.lower()]
                    exact_note = "Only valid target destination moves now: "
            else:
                exact = [
                    a for a in avail
                    if _alfworld_action_mentions_target(a, target) and a.startswith("move ")
                ][:6]
        elif not held_wrong:
            exact = _alfworld_exact_options(avail, prefix=f"take {target.lower()} ", limit=6)
            if placed_objs:
                exact = [a for a in exact if _alfworld_action_object(a).lower() not in placed_objs]
            exact_note = "Context-appropriate target take actions now: "
    if exact:
        lines.append(exact_note + " | ".join(exact[:8]))
    elif target:
        if held_target and verb in ("clean", "cool", "heat") and state_done:
            lines.append(
                f"Required `{verb}` state is already done. No further clean/cool/heat target action is allowed; navigate/open until destination move is admissible."
            )
        else:
            lines.append(
                f"No context-appropriate exact `{target}` action is admissible now. "
                f"Do NOT take substring/different classes such as butterknife/pan/etc.; keep searching/opening."
            )
    if parsed.get("examine_with_desklamp"):
        lamp_use_opts = [a for a in avail if a.lower().startswith("use desklamp")]
        if lamp_use_opts:
            lines.append("Desklamp action now: " + " | ".join(lamp_use_opts[:4]))
    lines.append("Admissible-action rule: output exactly one listed action string; if it is not listed, it will fail.")
    return "\n".join(lines)


def _build_alfworld_invalid_action_feedback(env_self, action_str, avail_actions):


    actions_preview = [str(a) for a in (avail_actions or []) if str(a) != "help"][:80]
    lines = [
        f"[INVALID_ACTION] `{action_str}` is not one of the current admissible actions.",
        "Environment state remains the same.",
    ]
    if actions_preview:
        lines.append("Current admissible actions are:")
        lines.extend(f"- {a}" for a in actions_preview)
    lines.append("Choose exactly one current admissible action string.")
    return "\n".join(lines)


def _append_alfworld_neutral_env_feedback(env_self, obs, action_str, info):

    if not _alfworld_env_feedback_enabled():
        return str(obs)

    text = str(obs)
    info = info or {}

    def _as_bool_or_none(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes"}:
                return True
            if s in {"false", "0", "no"}:
                return False
        return None

    def _last_visible_obs():


        candidates = [getattr(env_self, "_current_obs", "")]
        try:
            candidates.extend(obs0 for obs0, _act in reversed(getattr(env_self, "_react_history", []) or []))
        except Exception:
            pass
        for cand in candidates:
            cand_s = str(cand or "").strip()
            if not cand_s:
                continue
            low = cand_s.lower()
            if "nothing happens" in low or low.startswith("[env_feedback]") or low.startswith("[state_unchanged]"):
                continue
            if "[invalid_action]" in low or "[action_blocked" in low or "[no_progress]" in low:
                continue
            return cand_s
        return ""

    action_is_valid = _as_bool_or_none(info.get("action_is_valid"))
    action_is_effective = _as_bool_or_none(info.get("action_is_effective"))
    lower_obs = text.lower()
    no_effect = ("nothing happens" in lower_obs) or (action_is_effective is False)
    feedback = []

    if (action_is_valid is False or no_effect) and _alfworld_invalid_keep_state_enabled():
        prev_visible = _last_visible_obs()
        if prev_visible:


            if len(prev_visible) > 1200:
                prev_visible = prev_visible[:1200].rstrip() + "..."
            text = "[STATE_UNCHANGED] Current visible state remains:\n" + prev_visible
            lower_obs = text.lower()

    if action_is_valid is False:
        failed = str(action_str or "").strip()
        if failed:
            feedback.append(
                f"[ALFWORLD EXECUTION STATUS] Previous action `{failed}` was unavailable "
                "in the state where it was chosen."
            )
        else:
            feedback.append(
                "[ALFWORLD EXECUTION STATUS] Previous action was unavailable "
                "in the state where it was chosen."
            )
    elif no_effect:
        feedback.append(
            "[ALFWORLD EXECUTION STATUS] Previous action produced no visible state change."
        )

    if _alfworld_visit_feedback_enabled():


        try:
            import re
            m = re.match(r"^\s*go to\s+(.+?)\s*$", str(action_str or ""), flags=re.I)
            if m:
                loc = m.group(1).strip().lower()
                prev_visits = 0
                for _obs, prev_act in getattr(env_self, "_react_history", []) or []:
                    pm = re.match(r"^\s*go to\s+(.+?)\s*$", str(prev_act or ""), flags=re.I)
                    if pm and pm.group(1).strip().lower() == loc:
                        prev_visits += 1
                if prev_visits > 0:
                    feedback.append(
                        f"[ALFWORLD EXECUTION STATUS] Location `{loc}` has already been visited "
                        f"{prev_visits} time(s) before this step."
                    )
                elif "you see nothing" in lower_obs:
                    feedback.append(
                        "[ALFWORLD EXECUTION STATUS] This visited location currently shows no objects."
                    )
        except Exception:
            pass

    if not feedback:
        return text
    return text.rstrip() + "\n" + "\n".join(feedback)


def _extract_visited_receptacles(react_history, current_obs):

    visited = {}
    for j, (pre_obs, act) in enumerate(react_history):
        if not act or not str(act).startswith('go to '):
            continue
        recep = str(act)[6:].strip()
        if j + 1 < len(react_history):
            result = react_history[j + 1][0]
        else:
            result = current_obs
        result_str = str(result or '')
        if recep not in visited:
            visited[recep] = {'steps': [], 'last_obs_short': ''}
        visited[recep]['steps'].append(j + 1)
        visited[recep]['last_obs_short'] = result_str[:300].replace('\n', ' ')
    return visited


def _build_alfworld_progress_block(env_self, avail_actions):

    react_history = getattr(env_self, '_react_history', None) or []
    if len(react_history) < 4:
        return ""
    parsed = _parse_alfworld_task(getattr(env_self, '_task_description', '') or '')
    target = parsed.get('target_class')
    if not target:
        return ""
    current_obs = getattr(env_self, '_current_obs', '') or ''
    visited = _extract_visited_receptacles(react_history, current_obs)
    if not visited:
        return ""

    avail_recep = []
    if isinstance(avail_actions, list):
        for a in avail_actions:
            sa = str(a)
            if sa.startswith('go to '):
                avail_recep.append(sa[6:].strip())
    visited_set = set(visited.keys())
    unvisited = [r for r in avail_recep if r not in visited_set]

    lines = ["[PROGRESS CHECK — avoid repeats]"]
    lines.append(f"Target class: {target}")
    if parsed.get('dest_class'):
        lines.append(f"Destination class: {parsed['dest_class']}")
    if parsed.get('source_hint'):
        lines.append(f"Source hint from task: {parsed['source_hint']}")
    if parsed.get('count', 1) > 1:
        lines.append(f"Count required: {parsed['count']}")

    lines.append("")
    lines.append("Visited receptacles:")
    for r, info in sorted(visited.items()):
        steps_str = ', '.join(str(s) for s in info['steps'][-6:])
        obs_short = info['last_obs_short'][:200]
        present = "TARGET PRESENT" if _observation_mentions_object_class(obs_short, target) else "target not present"
        lines.append(f"- {r}: visited at steps {steps_str}; {present}; last saw: {obs_short}")

    if unvisited:
        unv = ', '.join(unvisited[:25])
        lines.append("")
        lines.append(f"Unvisited candidate receptacles (try one of these): {unv}")

    if visited_set and len(visited_set) <= 3:
        lines.append("")
        lines.append(f"Next-action rule: Do NOT keep alternating among {sorted(visited_set)}. Choose an unvisited receptacle.")

    return '\n'.join(lines)


def _build_react_prompt(self, avail_actions):

    from training.react_prompts import ALFWORLD_TEMPLATE, _ALFWORLD_EXAMPLE
    import os

    use_input_fixes = (
        self._task_type == "alfworld"
        and _alfworld_input_fixes_enabled()
    )

    total_history_length = len(self._react_history)
    if use_input_fixes:
        max_hist = _alfworld_history_max()
        if max_hist <= 0:
            recent = []
            start_idx = total_history_length
        else:
            recent = self._react_history[-max_hist:]
            start_idx = total_history_length - len(recent)
    else:
        recent = self._react_history
        start_idx = 0
    history_length = len(recent)

    history_lines = []
    if use_input_fixes:

        def _short_obs(x):
            s = str(x)
            max_chars = _alfworld_history_obs_chars()
            if max_chars > 0 and len(s) > max_chars:
                return s[:max_chars].rstrip() + "..."
            return s

        for j, (pre_obs, act) in enumerate(recent):
            step_num = start_idx + j + 1
            global_j = start_idx + j
            if global_j + 1 < total_history_length:
                result_obs = self._react_history[global_j + 1][0]
            else:
                result_obs = self._current_obs
            history_lines.append(
                f"[Step {step_num}: Action: '{act}' -> Result: '{_short_obs(result_obs)}']"
            )
    else:

        for j, (obs, act) in enumerate(recent):
            step_num = start_idx + j + 1
            history_lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
    action_history = "\n".join(history_lines)


    if use_input_fixes and _alfworld_progress_block_enabled():
        progress_block = _build_alfworld_progress_block(self, avail_actions)
        if progress_block:
            action_history = action_history + "\n\n" + progress_block


    skill_text = ""
    policy_skill_catalog = (
        self._react_skill_catalog_for_policy_action(self._task_description)
        if self.skill_mode == "policy_action" else ""
    )
    if (
        self.skill_mode != "policy_action"
        and self.workspace and self.workspace.size > 0
        and (self._task_type != "webshop" or _webshop_skill_enabled())
    ):
        candidates = self.workspace.retrieve(
            self._task_description, task_type=self._task_type, top_k=1
        )
        for tip in candidates:
            tip_plan = getattr(tip, 'plan', '') or ''
            if tip_plan:
                skill_text += f"- {tip_plan.strip()}\n"

    tip_prefix = (
        f"\n# Learned Strategy (follow this)\n{skill_text.strip()}\n\n"
        if skill_text else ""
    )

    if self._task_type == "webshop":
        rendered = _render_webshop_prompt(
            init=False,
            task_description=self._task_description,
            current_observation=self._current_obs,
            available_actions=avail_actions,
            action_history=action_history,
            step_count=total_history_length,
            history_length=history_length,
            current_step=total_history_length + 1,
            skill_text=skill_text,
        )
        if policy_skill_catalog:
            rendered = policy_skill_catalog + "\n" + rendered
        return _insert_webshop_visible_state_block(rendered, self, avail_actions)
    elif self._task_type == "alfworld":
        rendered = ALFWORLD_TEMPLATE.format(
            example=_ALFWORLD_EXAMPLE,
            task_description=self._task_description,
            step_count=total_history_length,
            history_length=history_length,
            action_history=action_history,
            current_step=total_history_length + 1,
            current_observation=self._current_obs,
            admissible_actions=_format_alfworld_actions(_order_alfworld_actions_for_prompt(avail_actions, self)),
        )
        rendered = _insert_alfworld_skill_trailer(rendered, self._task_description)
        rendered = _insert_alfworld_task_brief_and_memory(rendered, self, avail_actions)
        rendered = _insert_alfworld_skill_apply_block(rendered, self, avail_actions)


        if _alfworld_decision_block_enabled():
            decision_block = _build_alfworld_decision_state_block(self, avail_actions)
            rendered = rendered.replace(
                "Now it's your turn to take an action.",
                _ALFWORLD_DECISION_REMINDERS.strip() + "\n\n"
                + decision_block.strip()
                + "\n\nNow it's your turn to take an action."
            )
        return policy_skill_catalog + "\n" + tip_prefix + rendered
    return policy_skill_catalog + "\n" + tip_prefix + f"Step {len(self._react_history)+1}: {self._current_obs}"


def _format_webshop_actions(avail):

    import re as _re
    actions = []
    if isinstance(avail, dict):
        if avail.get("has_search_bar"):
            actions.append("search[<your query>]")
        clickables = [str(x) for x in avail.get("clickables", [])]
        if _env_flag("WEBSHOP_ORDER_ACTIONS", False):
            def _rank_clickable(x: str):
                lx = x.strip().lower()
                if _re.fullmatch(r"b[0-9a-z]{9}", lx):
                    return (0, lx)          
                if lx in {"buy now"}:
                    return (2, lx)          
                if lx in {"next >", "< prev"}:
                    return (3, lx)          
                if lx in {"back to search", "search", "description", "features", "reviews"}:
                    return (4, lx)          
                return (1, lx)              
            clickables = sorted(clickables, key=_rank_clickable)
        for item in clickables:
            actions.append(f"click[{item}]")
    elif isinstance(avail, list):
        actions = [str(a) for a in avail]
    else:
        return str(avail)

    return "\n".join(f"'{s}'," for s in actions)


def _alfworld_go_options_for_class(avail_actions, cls, specific_instance=None, limit=8):

    out = []
    cls = str(cls or "").lower().strip()
    inst = str(specific_instance or "").lower().strip()
    for a in avail_actions or []:
        s = str(a)
        low = s.lower()
        if not low.startswith("go to "):
            continue
        recep = s[6:].strip()
        if inst:
            if recep.lower() == inst:
                out.append(s)
        elif cls and _alfworld_object_class(recep) == cls:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _alfworld_known_target_source_go_options(env_self, avail_actions, target_class, preferred_dest_instance="", limit=6):

    if not target_class:
        return []
    sources = []
    for _obs, act in getattr(env_self, "_react_history", []) or []:
        obj = _alfworld_action_object(act)
        src = _alfworld_action_take_source(act)
        if obj and src and _target_matches_obj(obj, target_class):
            if preferred_dest_instance and src.lower() == preferred_dest_instance.lower():
                continue
            if src not in sources:
                sources.append(src)
    out = []
    avail_set = {str(a).lower(): str(a) for a in (avail_actions or [])}
    for src in sources:
        cand = f"go to {src}".lower()
        if cand in avail_set and avail_set[cand] not in out:
            out.append(avail_set[cand])
        if len(out) >= limit:
            break
    return out


def _order_alfworld_actions_for_prompt(avail_actions, env_self=None):

    actions = [str(a) for a in (avail_actions or []) if str(a) != "help"]


    if not _env_flag("ALFWORLD_ORDER_ACTIONS", False):
        return actions
    if env_self is None or not actions:
        return actions
    try:
        parsed = _parse_alfworld_task(getattr(env_self, "_task_description", "") or "")
        target = parsed.get("target_class") or ""
        dest = parsed.get("dest_class") or ""
        verb = parsed.get("verb")
        count = int(parsed.get("count") or 1)
        current_obs = str(getattr(env_self, "_current_obs", "") or "")
        held = _extract_held_objects_from_actions(actions, target_class=target)
        held_target = [h for h in held if target and _target_matches_obj(h, target)]
        held_wrong = [h for h in held if target and not _target_matches_obj(h, target)]
        placed_records = _alfworld_placed_target_records(env_self, target, dest) if target else []
        preferred_dest_instance = placed_records[0][1] if count > 1 and placed_records else ""
        state_done = _alfworld_state_action_done(env_self, parsed)
        appliance = {"clean": "sinkbasin", "cool": "fridge", "heat": "microwave"}.get(verb)
        closed_open_opts = _current_closed_open_options(current_obs, actions)

        priority = []
        seen = set()

        def add(seq):
            for x in seq or []:
                sx = str(x)
                if sx in actions and sx not in seen:
                    priority.append(sx)
                    seen.add(sx)

        if held_wrong:
            for obj in held_wrong:
                add(_alfworld_exact_options(actions, prefix=f"move {obj.lower()} to ", limit=8))

        if held_target:
            obj = held_target[0]


            relevant_closed = []
            relevant_classes = {c for c in (appliance, dest) if c}
            if parsed.get("examine_with_desklamp"):
                relevant_classes.add("desklamp")
            for op in closed_open_opts:
                recep = op[5:].strip()
                if _alfworld_object_class(recep) in relevant_classes:
                    relevant_closed.append(op)
            add(relevant_closed)

            if parsed.get("examine_with_desklamp"):
                add([a for a in actions if a.lower().startswith("use desklamp")])
                add(_rank_alfworld_lamp_search_go_options(actions, set(), limit=8))
            elif verb in ("clean", "cool", "heat") and not state_done:
                add(_alfworld_exact_options(actions, prefix=f"{verb} {obj.lower()} with ", limit=8))
                add(_alfworld_go_options_for_class(actions, appliance, limit=8))
            else:
                dest_moves = [
                    a for a in actions
                    if a.lower().startswith(f"move {obj.lower()} to ")
                    and (not dest or _alfworld_object_class(_alfworld_action_move_dest(a)) == dest.lower())
                ]
                if preferred_dest_instance:
                    same_dest = [
                        a for a in dest_moves
                        if _alfworld_action_move_dest(a).lower() == preferred_dest_instance.lower()
                    ]
                    add(same_dest)
                    add(_alfworld_go_options_for_class(actions, "", specific_instance=preferred_dest_instance, limit=4))
                add(dest_moves)
                add(_alfworld_go_options_for_class(actions, dest, limit=8))
        else:

            if target:
                target_take_opts = _alfworld_exact_options(actions, prefix=f"take {target.lower()} ", limit=12)
                placed_objs = {obj.lower() for obj, _dst in placed_records}
                if placed_objs:
                    target_take_opts = [
                        a for a in target_take_opts
                        if _alfworld_action_object(a).lower() not in placed_objs
                    ]
                add(target_take_opts)

            if count > 1 and placed_records and len(placed_records) < count:
                add(_alfworld_known_target_source_go_options(
                    env_self, actions, target,
                    preferred_dest_instance=preferred_dest_instance,
                    limit=6,
                ))

            if target:
                visited = {}
                total = len(getattr(env_self, "_react_history", []) or [])
                for j, (_pre_obs, act) in enumerate(getattr(env_self, "_react_history", []) or []):
                    act_s = str(act or "")
                    if not act_s.startswith("go to "):
                        continue
                    loc = act_s[6:].strip()
                    if j + 1 < total:
                        result = getattr(env_self, "_react_history", [])[j + 1][0]
                    else:
                        result = getattr(env_self, "_current_obs", "") or ""
                    if not str(result).lower().startswith("[state_unchanged]"):
                        visited[loc] = True
                if parsed.get("examine_with_desklamp") and held_target:
                    add(_rank_alfworld_lamp_search_go_options(actions, set(visited.keys()), limit=8))
                else:
                    ranked_go = _rank_alfworld_unvisited_go_options(
                        target,
                        actions,
                        set(visited.keys()),
                        limit=12,
                        dest_class=dest,
                        dest_penalty=(14 if count > 1 and placed_records else None),
                    )
                    source_focus = set(_alfworld_source_focus_classes(target, dest))
                    focused_go = [
                        a for a in ranked_go
                        if a.startswith("go to ")
                        and _alfworld_object_class(a[6:].strip()) in source_focus
                    ] if source_focus else []
                    low_yield_open_classes = {"cabinet", "drawer", "shelf"}
                    immediate_opens = []
                    deferred_opens = []
                    for op in closed_open_opts:
                        cls = _alfworld_object_class(op[5:].strip())
                        if (
                            focused_go and cls in low_yield_open_classes
                            and cls not in source_focus
                            and not target_take_opts
                        ):
                            deferred_opens.append(op)
                        else:
                            immediate_opens.append(op)


                    add(immediate_opens)
                    add(ranked_go)
                    add(deferred_opens)
            else:
                add(closed_open_opts)


        add(actions)
        return priority
    except Exception:
        return actions


def _format_retrieved_skill_memory(skill_text: str) -> str:

    text = (skill_text or "").strip()
    if not text:
        return "No relevant skills found for this task."
    lines = ["### Learned ALFWorld/WebShop Skill"]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-"):
            lines.append(line)
        else:
            lines.append(f"- {line}")
    return "\n".join(lines)


def _format_alfworld_actions(avail):

    if isinstance(avail, list):
        return "\n".join(f"  '{a}'," for a in avail if a != "help")
    return str(avail)


GenericTaskEnvironment.reset_react = _reset_react
GenericTaskEnvironment.react_step = _react_step
