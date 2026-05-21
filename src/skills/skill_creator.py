

from __future__ import annotations

import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

from src.executor.m_exec import MExec
from src.skills.format import SkillEntry, SkillMeta, parse_skill_blocks, assign_skill_id
from src.skills.skill_prompts import (
    GENERATE_SKILL_PROMPT,
    MERGE_SKILL_PROMPT,
    SKILL_REFINE_PROMPT,
    EVOLUTION_SKILL_PROMPT,
    format_trajectory_for_skill_generation,
    format_trajectory_steps_only,
    clean_skill_output,
)
from training.trajectory import Trajectory
from training.flow_metrics import identify_top_importance_steps


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training.backward_policy import BackwardPolicy

logger = logging.getLogger(__name__)


MIN_PLAN_WORDS = 20
MAX_PLAN_WORDS = 200
MAX_NAME_WORDS = 7
EMBEDDING_SIM_THRESHOLD = 0.80   


_VAGUE_PATTERNS = [
    "gather information systematically",
    "collect all relevant data",
]


def _trajectory_outcome_summary(*trajs: Trajectory) -> str:

    parts = []
    for i, traj in enumerate(trajs, 1):
        if traj is None:
            continue
        parts.append(
            f"trajectory_{i}: task_type={traj.task_type}, "
            f"reward={getattr(traj, 'reward', 0.0):.3f}, "
            f"status={'success' if getattr(traj, 'reward', 0.0) >= 0.5 else 'failure'}"
        )
    return "\n".join(parts) if parts else "No outcome metadata available."


class SkillCreator:


    def __init__(
        self,
        m_exec: MExec,
        skill_workspace=None,  
        backward_policy: "Optional[BackwardPolicy]" = None,


    ):
        self.m_exec = m_exec
        self.workspace = skill_workspace
        self.backward_policy = backward_policy

    @staticmethod
    def _extract_field(block: str, field: str) -> Optional[str]:

        import re

        m = re.search(rf'{field}:\s*"([^"]+)"', block)
        if m:
            return m.group(1).strip()

        m = re.search(rf"{field}:\s*'([^']+)'", block)
        if m:
            return m.group(1).strip()

        m = re.search(rf'{field}:\s*\*\*(.+?)\*\*', block)
        if m:
            return m.group(1).strip()

        m = re.search(rf'{field}:\s*(.+)', block)
        if m:
            text = m.group(1).strip().strip('"\'*')
            if len(text) > 5:  
                return text
        return None

    @staticmethod
    def _get_top_importance_indices(traj: Trajectory) -> set:

        if not traj.turns:
            return set()
        import math
        T = len(traj.turns)

        contributions = []
        for i, t in enumerate(traj.turns):
            imp = getattr(t, "step_importance", 1.0)
            c = abs(math.log(max(imp, 1e-300)))
            contributions.append((i, c))
        total_c = sum(c for _, c in contributions)
        if total_c < 1e-10:
            return set()

        uniform_share = total_c / T
        return {idx for idx, c in contributions if c > uniform_share}


    def extract_skills_from_trajectories(
        self,
        high_trajs: List[Trajectory],
        low_trajs: List[Trajectory],
        task_type: str,
        target_count: int = 4,
    ) -> List[SkillEntry]:

        if not high_trajs and not low_trajs:
            return []


        per_traj_skills_raw = []  
        per_traj_skills = []  


        pairs = []
        max_pairs = min(target_count, 4)
        for i in range(max_pairs):
            s = high_trajs[i % len(high_trajs)] if high_trajs else low_trajs[0]
            f = low_trajs[i % len(low_trajs)] if low_trajs else high_trajs[-1]
            pairs.append((s, f))

        for s_traj, f_traj in pairs:

            s_text = format_trajectory_for_skill_generation(s_traj)
            f_text = format_trajectory_for_skill_generation(f_traj)

            outcome_summary = _trajectory_outcome_summary(s_traj, f_traj)

            prompt = GENERATE_SKILL_PROMPT.format(
                task_type=task_type,
                success_trajectory=s_text,
                failure_trajectory=f_text,
                outcome_summary=outcome_summary,
            )

            try:
                response = self.m_exec.execute(prompt, max_tokens=2000, temperature=0.4)
                response = clean_skill_output(response)
                per_traj_skills_raw.append(response)


                parsed = parse_skill_blocks(response)
                if parsed:
                    per_traj_skills.append(parsed[0])
            except Exception as e:
                logger.warning(f"[SkillExtract] Per-trajectory extraction failed: {e}")

        logger.info(
            f"[SkillExtract] {task_type}: {len(per_traj_skills_raw)} raw SOPs, "
            f"{len(per_traj_skills)} parsed SkillEntry"
        )

        if not per_traj_skills_raw:
            return []


        living_doc = self.merge_into_living_document(
            task_type=task_type,
            existing_document="",
            new_skill_contents=per_traj_skills_raw,
        )

        if living_doc:
            logger.info(
                f"[SkillExtract] {task_type}: living document created "
                f"({len(living_doc.split())} words)"
            )

            if self.workspace:
                from src.skills.format import TaskTypeSkillDocument
                doc = TaskTypeSkillDocument(
                    task_type=task_type,
                    consolidated_strategy=living_doc,
                    variant_skill_ids=[],
                    proven_patterns=[],
                )
                self.workspace.update_type_document(doc)


        traj_ids = [t.traj_id for t in (high_trajs[:3] + low_trajs[:2]) if hasattr(t, 'traj_id')]
        for skill in per_traj_skills:
            skill.meta.source = "trajectory_extraction"
            skill.meta.creation_step = 0
            skill.meta.task_types = [task_type]
            skill.meta.trajectory_support = traj_ids


        per_traj_skills = self._quality_gate(per_traj_skills, existing_skills=[])
        logger.info(f"[SkillExtract] {task_type}: {len(per_traj_skills)} skills after quality gate")
        return per_traj_skills[:target_count]


    def extract_atomic_tips(
        self,
        high_trajs: List[Trajectory],
        low_trajs: List[Trajectory],
        task_type: str,
        max_tips: int = 3,
    ) -> List[SkillEntry]:

        from src.skills.skill_prompts_v2 import (
            GENERATE_TIP_PROMPT, parse_tips_yaml
        )
        from src.skills.skill_prompts import (
            format_trajectory_for_skill_generation,
        )

        if not high_trajs and not low_trajs:
            return []


        pairs = []
        n_pairs = min(max_tips, 3)
        for i in range(n_pairs):
            s = high_trajs[i % max(len(high_trajs), 1)] if high_trajs else low_trajs[0]
            f = low_trajs[i % max(len(low_trajs), 1)] if low_trajs else high_trajs[-1]
            if s.traj_id != f.traj_id:  
                pairs.append((s, f))

        all_tips = []
        for s_traj, f_traj in pairs:
            s_text = format_trajectory_for_skill_generation(s_traj)
            f_text = format_trajectory_for_skill_generation(f_traj)
            outcome_summary = _trajectory_outcome_summary(s_traj, f_traj)

            prompt = GENERATE_TIP_PROMPT.format(
                task_type=task_type,
                success_trajectory=s_text,
                failure_trajectory=f_text,
                outcome_summary=outcome_summary,
            )

            try:
                response = self.m_exec.execute(prompt, max_tokens=500, temperature=0.3)
                tips = parse_tips_yaml(response)
                import time as _time
                for trigger, tip_text in tips:
                    ts = _time.time_ns()
                    tt_short = task_type.replace('_', '-')
                    skill_id = f"tip-{tt_short}-{ts}-{len(all_tips)}"
                    entry = SkillEntry(
                        name=skill_id,
                        description=trigger,
                        trigger=trigger,
                        plan=tip_text,
                        pitfall="",
                        constraint="",
                        meta=SkillMeta(
                            skill_id=skill_id,
                            source="atomic_tip",
                            task_types=[task_type],
                            creation_step=0,
                            trajectory_support=[s_traj.traj_id, f_traj.traj_id],
                        ),
                    )
                    all_tips.append(entry)
                    logger.info(
                        f"[TipExtract] {task_type}: "
                        f"\"{tip_text[:80]}\" ({len(tip_text.split())} words)"
                    )
            except Exception as e:
                logger.warning(f"[TipExtract] Failed for {task_type}: {e}")

        logger.info(f"[TipExtract] {task_type}: {len(all_tips)} atomic tips extracted")
        return all_tips[:max_tips]


    def _call_tip_generator(self, prompt: str, max_tokens: int = 1000) -> str:

        import requests, re, time as _time

        api_base = os.environ.get("SKILL_CREATOR_API_BASE", "http://127.0.0.1:3456/v1/messages")
        api_key = os.environ.get("SKILL_CREATOR_API_KEY", "EMPTY")
        model = os.environ.get("SKILL_CREATOR_MODEL", "skill-creator-model")
        max_retries = 5
        backoffs = [5, 15, 45, 120, 300]  
        last_err = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    api_base,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": model,
                        "max_tokens": max_tokens,
                        "stream": False,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=None,
                )
                raw = resp.text

                if '"type":"error"' in raw[:200]:
                    raise RuntimeError(f"Skill Creator error body: {raw[:300]}")
                if "text_delta" in raw:
                    deltas = re.findall(r'"text_delta","text":"(.*?)"', raw)
                    text = "".join(d.replace("\\n", "\n").replace('\\"', '"') for d in deltas)
                else:
                    data = resp.json()
                    text = ""
                    for block in data.get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            break
                if text and len(text) > 20:
                    suffix = f" (attempt {attempt+1})" if attempt > 0 else ""
                    logger.info(f"[TipGen] Skill Creator LLM returned {len(text)} chars{suffix}")
                    return text
                raise RuntimeError(f"Skill Creator returned empty response (resp {len(raw)} chars)")
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < max_retries - 1:
                    wait = backoffs[attempt]
                    logger.warning(
                        f"[TipGen] Skill Creator attempt {attempt+1}/{max_retries} failed ({last_err}), "
                        f"retry after {wait}s"
                    )
                    _time.sleep(wait)
                else:
                    logger.error(f"[TipGen] Skill Creator all {max_retries} attempts failed: {last_err}")
        raise RuntimeError(f"Skill Creator failed after {max_retries} retries: {last_err}")


    _TASK_TOOLS = {
        "multi_hop_qa": "search (find passages), lookup (find detail in passage), fact_verify, decompose",
        "factual_qa": "search (find passages), lookup (find detail), fact_verify",
        "math_reasoning": "python_execute (run Python code), decompose, self_consistency, verify_answer",
        "code_generation": "list_files, search_code (find patterns), view_file (read code), edit_file (modify code), run_tests, python_execute",
        "science_qa": "search, lookup, python_execute, analyze",
        "webshop": "search[query], click[element]",
        "alfworld": "go to X, take X from Y, move X to Y (place item), open X, close X, clean X, heat X, cool X, examine X, inventory, look",
    }

    def curate_and_evolve_tips(
        self,
        task_type: str,
        observations: List[Dict],
        failed_trajectories: List,
        workspace,
        high_flow_trajs=None,
        critical_steps=None,
        dag_comparisons=None,

        bottleneck_diagnoses=None,
        counterfactual_pairs=None,
    ) -> dict:

        from src.skills.skill_prompts_v2 import (
            CURATE_TIPS_PROMPT, GENERATE_SINGLE_TIP_PROMPT,
            parse_curation_verdict, parse_single_tip,
        )

        result = {"added": [], "updated": [], "deleted": [], "skipped": False}
        tool_list = self._TASK_TOOLS.get(task_type, "search, analyze, python_execute")


        existing_skills = workspace.get_by_task_type(task_type) if hasattr(workspace, 'get_by_task_type') else [
            s for s in workspace.get_all() if task_type in (s.meta.task_types or [])
        ]
        if existing_skills:
            existing_text = "\n".join(
                f"[{s.meta.skill_id}] (usage={s.meta.usage_count}, success={s.meta.success_count}, "
                f"flow={s.meta.flow_score:.1f})\n  Description: {s.description}\n  Body: {s.plan}"
                for s in existing_skills
            )
        else:
            existing_text = "(No tips exist yet for this task type.)"


        def _format_traj(traj, label: str) -> str:
            lines = [f"{label} (R={traj.reward:.2f}, {len(traj.turns)} steps):"]
            for i, t in enumerate(traj.turns):
                instr = (getattr(t, 'instruction', '') or '') if hasattr(t, 'instruction') else ''
                obs = (getattr(t, 'observation', '') or '') if hasattr(t, 'observation') else ''
                lines.append(f"  Step {i+1}: [{t.action_type}] {instr}")
                if obs:
                    lines.append(f"    → {obs}")
            return "\n".join(lines)

        success_text = ""
        if high_flow_trajs:
            for st in high_flow_trajs[:3]:
                success_text += _format_traj(st, "Success") + "\n\n"

        fail_text = ""
        if failed_trajectories:
            for ft in failed_trajectories[:3]:
                fail_text += _format_traj(ft, "Failed") + "\n\n"

        critical_text = ""
        if critical_steps:
            for cs in critical_steps:
                critical_text += f"  Step {cs['step']}: {cs['action']} (I={cs['I_t']}, R={cs['from_reward']})\n"
                if cs.get('observation'):
                    critical_text += f"    → {cs['observation']}\n"

        dag_text = ""
        if dag_comparisons:
            for dc in dag_comparisons:
                dag_text += f"  Q: {dc['question']} (reward gap={dc['reward_gap']})\n"
                dag_text += f"    Success path: {' → '.join(dc['success_actions'])}\n"
                dag_text += f"    Failure path: {' → '.join(dc['failure_actions'])}\n"

        evidence_summary = (success_text + fail_text + critical_text + dag_text).strip() or "[no evidence]"


        try:
            if bottleneck_diagnoses:

                from src.skills.skill_prompts_v2 import DIAGNOSE_AND_CURATE_PROMPT
                diag_text = ""
                for i, diag in enumerate(bottleneck_diagnoses):
                    bn = diag.bottleneck
                    diag_text += f"### Bottleneck {i}: steps {bn.step_bucket*2}-{bn.step_bucket*2+1}, "
                    diag_text += f"action={bn.action_type}\n"
                    diag_text += f"  Flow variance: {bn.var_log_It:.2f} "
                    diag_text += f"(mean={bn.mean_log_It:+.2f}, n={bn.n_samples})\n"
                    diag_text += f"  Coverage: {diag.current_skill_coverage}\n"
                    diag_text += f"  Suggested: {diag.suggested_edit_type}\n\n"

                cf_text = ""
                all_cps = counterfactual_pairs or []
                for diag in bottleneck_diagnoses:
                    for cp in diag.counterfactual_pairs[:2]:
                        cf_text += f"  Divergence at step {cp.divergence_step+1} "
                        cf_text += f"(reward gap={cp.reward_gap}):\n"
                        if cp.context_before:
                            last = cp.context_before[-1]
                            cf_text += f"    Last shared: [{last.get('action_type','')}] {last.get('instruction','')[:50]}\n"
                        cf_text += f"    ✅ SUCCESS: [{cp.success_choice.get('action_type','')}] "
                        cf_text += f"{cp.success_choice.get('instruction','')[:60]}\n"
                        cf_text += f"      → R={cp.success_downstream.get('final_reward',0):.2f} "
                        cf_text += f"({cp.success_downstream.get('n_remaining_steps',0)} more steps)\n"
                        cf_text += f"    ❌ FAILURE: [{cp.failure_choice.get('action_type','')}] "
                        cf_text += f"{cp.failure_choice.get('instruction','')[:60]}\n"
                        cf_text += f"      → R={cp.failure_downstream.get('final_reward',0):.2f} "
                        cf_text += f"({cp.failure_downstream.get('n_remaining_steps',0)} more steps)\n\n"

                curation_prompt = DIAGNOSE_AND_CURATE_PROMPT.format(
                    task_type=task_type,
                    existing_tips=existing_text,
                    tool_list=tool_list,
                    bottleneck_diagnoses=diag_text or "[none detected]",
                    counterfactual_evidence=cf_text or "[none]",
                )
                logger.info(f"[Curation] Using 3+2+1 diagnosis prompt for {task_type}")
            else:

                curation_prompt = CURATE_TIPS_PROMPT.format(
                    task_type=task_type,
                    existing_tips=existing_text,
                    tool_list=tool_list,
                    success_evidence=success_text or "[none]",
                    failure_evidence=fail_text or "[none]",
                    critical_steps=critical_text or "[none]",
                    dag_comparisons=dag_text or "[none]",
                )

            curation_response = self._call_tip_generator(curation_prompt, max_tokens=1000)
            verdict = parse_curation_verdict(curation_response)
            logger.info(f"[Curation] {task_type}: {len(verdict['actions'])} actions, needs_new={verdict['needs_new_tip']}")
        except Exception as e:
            logger.warning(f"[Curation] Phase 1 failed for {task_type}: {e}")
            verdict = {"actions": [], "needs_new_tip": True, "new_tip_focus": "generate a useful strategy"}


        from src.skills.format import SkillEntry, SkillMeta
        import time as _time

        for act in verdict.get("actions", []):
            action = act.get("action", "").upper()
            sid = act.get("skill_id", "")

            if action == "DELETE" and sid:
                result["deleted"].append(sid)
                logger.info(f"[Curation] DELETE {sid}: {act.get('reason', '')}")

            elif action == "UPDATE" and sid:
                old_skill = workspace.get_by_id(sid)
                if old_skill:
                    updated = SkillEntry(
                        name=old_skill.name,
                        description=act.get("new_description", old_skill.description),
                        trigger=act.get("new_description", old_skill.trigger),
                        plan=act.get("new_body", old_skill.plan),
                        pitfall=old_skill.pitfall,
                        constraint=old_skill.constraint,
                        meta=old_skill.meta,
                    )
                    result["updated"].append((sid, updated))
                    logger.info(f"[Curation] UPDATE {sid}: {act.get('reason', '')}")

            elif action == "ADD":
                new_body = act.get("new_body", "")
                new_desc = act.get("new_description", "")
                if new_body and new_desc and len(new_body.split()) >= 8:
                    ts = _time.time_ns()
                    tt_short = task_type.replace('_', '-')
                    bn_id = act.get("bottleneck_id", len(result["added"]))
                    skill_id = f"tip-{tt_short}-{ts}-{bn_id}"
                    entry = SkillEntry(
                        name=skill_id,
                        description=new_desc,
                        trigger=new_desc,
                        plan=new_body,
                        pitfall="",
                        constraint="",
                        meta=SkillMeta(
                            skill_id=skill_id,
                            source="llm_3+2+1",
                            task_types=[task_type],
                        ),
                    )
                    result["added"].append(entry)
                    logger.info(f"[Curation] ADD from verdict: \"{new_body[:60]}\" ({len(new_body.split())}w)")
                else:
                    logger.warning(f"[Curation] ADD action has insufficient body: {len((new_body or '').split())}w")


        needs_phase2 = verdict.get("needs_new_tip", False)
        if not result["added"] and not existing_skills and bottleneck_diagnoses:

            needs_phase2 = True
            logger.info(f"[Curation] Forcing Phase 2: workspace empty + bottlenecks detected")

        if needs_phase2 and not result["added"]:
            focus = verdict.get("new_tip_focus", "")
            if not focus:
                focus = "generate a useful tool-calling strategy based on the training evidence"

            remaining_ids = set(s.meta.skill_id for s in existing_skills) - set(result["deleted"])
            remaining_text = "\n".join(
                f"- {s.plan}" for s in existing_skills if s.meta.skill_id in remaining_ids
            ) or "(none)"

            try:
                gen_prompt = GENERATE_SINGLE_TIP_PROMPT.format(
                    task_type=task_type,
                    existing_tips=remaining_text,
                    new_tip_focus=focus,
                    evidence_summary=evidence_summary,
                    tool_list=tool_list,
                )
                gen_response = self._call_tip_generator(gen_prompt, max_tokens=500)
                parsed = parse_single_tip(gen_response)
                if parsed:
                    desc, body = parsed
                    ts = _time.time_ns()
                    tt_short = task_type.replace('_', '-')
                    skill_id = f"tip-{tt_short}-{ts}-0"
                    entry = SkillEntry(
                        name=skill_id,
                        description=desc,
                        trigger=desc,
                        plan=body,
                        pitfall="",
                        constraint="",
                        meta=SkillMeta(
                            skill_id=skill_id,
                            source="llm_curation",
                            task_types=[task_type],
                        ),
                    )
                    result["added"].append(entry)
                    logger.info(f"[Curation] ADD new tip (Phase 2): \"{body[:60]}\" ({len(body.split())}w)")
                else:
                    logger.warning(f"[Curation] Phase 2 parse failed for {task_type}")
            except Exception as e:
                logger.warning(f"[Curation] Phase 2 failed for {task_type}: {e}")
        elif not result["added"] and not result["updated"]:
            result["skipped"] = True
            logger.info(f"[Curation] {task_type}: no changes needed")

        return result

    def generate_tips_from_observations(
        self,
        task_type: str,
        observations: List[Dict],
        failed_trajectories: List[Trajectory],
        max_tips: int = 2,

        high_flow_trajs: Optional[List] = None,
        critical_steps: Optional[List[Dict]] = None,
        dag_comparisons: Optional[List[Dict]] = None,
    ) -> List[SkillEntry]:

        from src.skills.skill_prompts_v2 import (
            GENERATE_TIP_FROM_OBSERVATIONS_PROMPT, parse_tips_yaml
        )


        obs_text = "\n".join(
            f"- [{o['action_type']}] \"{o['instruction']}\" "
            f"(I={o['I_t']:.2f}, step {o['step_position']}/{o['n_steps']}, R={o['traj_reward']})"
            for o in observations[:10]
        )


        fail_text = ""
        if failed_trajectories:
            for ft in failed_trajectories[:2]:
                steps = [
                    f"  [{t.action_type}] {(t.instruction or '')[:50]}"
                    for t in ft.turns[:5]
                ]
                fail_text += f"Failed (R={ft.reward:.2f}):\n" + "\n".join(steps) + "\n"


        _TASK_TOOLS = {
            "multi_hop_qa": "search (find passages), lookup (find detail in passage), fact_verify, decompose (break into sub-questions)",
            "factual_qa": "search (find passages), lookup (find detail), fact_verify",
            "math_reasoning": "python_execute (run Python code), decompose, self_consistency, cross_validate",
            "code_generation": "list_files, search_code (find patterns), view_file (read code), edit_file (modify code), run_tests, python_execute",
            "science_qa": "search, lookup, python_execute",
            "webshop": "search[query] (search items by keywords), click[element] (select product/option/Buy Now)",
            "alfworld": "go to X, take X from Y, put X in Y, put X on Y, open X, close X, clean X with sinkbasin, heat X with microwave, cool X with fridge, examine X, inventory, look",
            "interactive_agent": "search_product, click, act",  
        }
        tool_list = _TASK_TOOLS.get(task_type, "search, analyze, python_execute")


        success_text = ""
        if high_flow_trajs:
            for st in high_flow_trajs[:2]:
                steps = [
                    f"  [{t.action_type}] {(t.instruction or '')[:50]}"
                    for t in st.turns[:5]
                ]
                success_text += f"Success (R={st.reward:.2f}):\n" + "\n".join(steps) + "\n"


        critical_text = ""
        if critical_steps:
            critical_text = "Critical decision points (high I(t) from backward policy):\n"
            for cs in critical_steps[:5]:
                critical_text += f"  Step {cs['step']}: {cs['action']} (I={cs['I_t']}, R={cs['from_reward']})\n"


        dag_text = ""
        if dag_comparisons:
            dag_text = "Same-question trajectory comparisons:\n"
            for dc in dag_comparisons[:3]:
                dag_text += f"  Q: {dc['question'][:50]}... (reward gap={dc['reward_gap']})\n"
                dag_text += f"    Success path: {' → '.join(dc['success_actions'][:4])}\n"
                dag_text += f"    Failure path: {' → '.join(dc['failure_actions'][:4])}\n"


        flow_evidence = ""
        if success_text:
            flow_evidence += f"\nHigh-flow success trajectories:\n{success_text}"
        if critical_text:
            flow_evidence += f"\n{critical_text}"
        if dag_text:
            flow_evidence += f"\n{dag_text}"

        prompt = GENERATE_TIP_FROM_OBSERVATIONS_PROMPT.format(
            task_type=task_type,
            tool_list=tool_list,
            observations=obs_text or "[no observations]",
            failed_contrast=(fail_text or "[none]") + flow_evidence,
        )

        try:
            response = self._call_tip_generator(prompt)
            tips = parse_tips_yaml(response)
            result = []
            import time as _time
            for desc, body in tips[:max_tips]:
                ts = _time.time_ns()
                tt_short = task_type.replace('_', '-')
                skill_id = f"tip-{tt_short}-{ts}-{len(result)}"
                entry = SkillEntry(
                    name=skill_id,
                    description=desc,
                    trigger=desc,
                    plan=body,
                    pitfall="",
                    constraint="",
                    meta=SkillMeta(
                        skill_id=skill_id,
                        source="flow_observation_tip",
                        task_types=[task_type],
                        creation_step=0,
                    ),
                )
                result.append(entry)
                logger.info(
                    f"[TipGen] {task_type}: \"{body[:60]}\" ({len(body.split())}w)"
                )
            return result
        except Exception as e:
            logger.warning(f"[TipGen] Failed for {task_type}: {e}")
            return []

    def _format_trajectory_for_extraction(self, traj: Trajectory) -> str:

        return format_trajectory_for_skill_generation(traj, hide_question=True)

    def _format_trajectory_for_extraction_legacy(self, traj: Trajectory) -> str:

        import math
        top_steps = self._get_top_importance_indices(traj)

        lines = [
            f"Task type: {traj.task_type}",
            f"Question: {traj.question[:200]}",
            "(★ marks steps that dominate the flow balance — where forward and backward "
            "policy disagree most. log_I shows direction: positive = agent decisive, "
            "negative = hindsight says this step was critical)",
            "",
        ]
        for i, turn in enumerate(traj.turns[:8]):
            action = turn.action_type
            instr = (turn.instruction or "")[:100]
            obs = (turn.observation or "")[:150]
            imp = getattr(turn, "step_importance", 1.0)
            log_i = math.log(max(imp, 1e-300))
            if i in top_steps:
                importance_marker = f"★[log_I={log_i:+.1f}] "
            else:
                importance_marker = ""
            lines.append(f"Step {i+1}: {importance_marker}[{action}] {instr}")
            if obs:
                lines.append(f"  Result: {obs}")
            lines.append("")
        lines.append(f"Final answer: {traj.final_answer[:100] if traj.final_answer else '(none)'}")
        lines.append(f"Reward: {traj.reward:.2f}")
        return "\n".join(lines)

    def genesis(
        self,
        seed_questions: List[Dict],
        exploration_trajectories: Optional[List[Trajectory]] = None,
        target_count: int = 12,
    ) -> List[SkillEntry]:

        logger.info(f"Starting genesis with {len(seed_questions)} seed questions")


        task_types = list(set(q.get("task_type", "unknown") for q in seed_questions))
        task_type_examples: Dict[str, List[str]] = {t: [] for t in task_types}

        for q in seed_questions:
            tt = q.get("task_type", "unknown")
            if len(task_type_examples[tt]) < 3:
                task_type_examples[tt].append(str(q.get("question", "")))


        prompt = self._build_genesis_prompt(task_type_examples, exploration_trajectories, target_count)


        all_skills = []
        for attempt in range(3):
            temp = 0.7 - attempt * 0.2  
            logger.info(f"Calling M_exec for genesis skill generation (attempt {attempt+1}, temp={temp})...")
            response = self.m_exec.execute(prompt, max_tokens=6000, temperature=temp)
            logger.info(f"Genesis response length: {len(response)} chars, first 200: {response[:200]}")

            batch = parse_skill_blocks(response)
            logger.info(f"Attempt {attempt+1}: parsed {len(batch)} skills")
            all_skills.extend(batch)

            if len(all_skills) >= target_count:
                break

        skills = all_skills
        logger.info(f"Parsed {len(skills)} total skills from genesis")


        for skill in skills:
            skill.meta.source = "genesis"
            skill.meta.creation_step = 0


        skills = self._quality_gate(skills, existing_skills=[])
        assign_skill_id(skills, start=0)

        logger.info(f"Genesis complete: {len(skills)} valid skills")
        return skills[:target_count]

    def _build_genesis_prompt(
        self,
        task_type_examples: Dict[str, List[str]],
        trajectories: Optional[List[Trajectory]] = None,
        target_count: int = 12,
    ) -> str:

        traj_analysis = ""
        if trajectories:

            success_patterns = []
            failure_patterns = []
            for traj in trajectories[:20]:
                if traj.reward > 0.5:
                    for turn in traj.turns[:3]:
                        if turn.instruction:
                            success_patterns.append(f"- {turn.action_type}: {turn.instruction[:100]}")
                else:
                    if traj.n_parse_errors > 0:
                        failure_patterns.append("- JSON parse errors")

            if success_patterns:
                traj_analysis = f"\nSuccessful action patterns observed:\n" + "\n".join(success_patterns[:10])
            if failure_patterns:
                traj_analysis += f"\nFailure patterns to avoid:\n" + "\n".join(set(failure_patterns[:5]))

        task_section = ""
        for tt, examples in task_type_examples.items():
            task_section += f"\n### {tt}\n"
            for ex in examples:
                task_section += f"  - {ex[:100]}\n"

        return f"""You are designing a skill library for an AI Supervisor agent.

The Supervisor orchestrates task solving through tools and a powerful AI Executor (M_exec).
The Supervisor has these tools (v3, 23 tools):
  Reasoning: think (quick reasoning), plan (M_exec plans), decompose (break into sub-questions)
  Computation: python_execute (M_exec writes+runs code), test_code (write+test function), analyze (M_exec analysis)
  Retrieval: search (BM25+semantic passages), lookup (keyword in prior results), fact_verify (claim verification)
  Answering: ask_llm (M_exec direct answer), self_consistency (3x majority vote)
  Verification: verify_answer (substitution check), check_answer (format check), cross_validate (alternative method)
  Code (SWE): search_code (grep workspace), view_file, edit_file, run_tests
  Environment: act (ALFWorld), search_product (WebShop), click (WebShop)
  Strategy: skill_invoke (invoke a learned skill from the skill library — SkillFlow core mechanism)
  Terminal: accept (submit final answer)

ARCHITECTURE: The Supervisor decides WHICH tool to use and WHAT goal to achieve.
For computation tools (python_execute, test_code), M_exec generates the code and the environment executes it.
For search/fact_verify, the environment searches locally using BM25+embedding hybrid.
For analyze/ask_llm, M_exec does general reasoning.

Skills guide the Supervisor on WHICH TOOLS to use in which order and HOW to decompose problems.

The agent needs skills to handle these task types:
{task_section}
{traj_analysis}

Generate {target_count} high-quality skills in the following YAML format.
IMPORTANT: ALL skills must be specifically for the task type(s) shown above. Do NOT generate skills for other task types.
Each skill should:
1. Cover a distinct reasoning or orchestration strategy FOR THE ABOVE TASK TYPE(S)
2. Be specific enough to be actionable (not vague)
3. Have a clear "plan" with step-by-step instructions using the tools listed above

For each skill, output EXACTLY this format (including the --- delimiters).
Do NOT include a "meta" section — the system will automatically assign IDs and metadata.

---
name: "Skill Name Here"
description: "One sentence describing what this skill does"
trigger: "When to use: specific task patterns, question types"
plan: |
  1. Use [tool_name] to [specific goal]
  2. Use [next tool] to [next step based on result]
  3. Use [verify tool] to confirm, then accept
pitfall: "Common mistake to avoid"
constraint: "Maximum N steps. Stop condition."
---

GOOD skill examples (notice how each uses specific tools):

---
name: "Step-by-Step Math Solver"
description: "Solve competition math via step-by-step coding (SBSC/ToRA pattern: reason → code → verify)"
trigger: "Competition math (AIME, AMC, MATH); equations, inequalities, counting, geometry"
plan: |
  1. Use decompose to break the problem into 2-3 sub-problems
  2. Use python_execute for each sub-problem: 'Use sympy to solve [sub-problem] and print result'
  3. Use python_execute to combine sub-results and verify the final answer numerically
  4. Accept the numeric answer (integer or simplified fraction)
pitfall: "Don't do mental math — always verify with python_execute. For sympy, use 'import sympy as sp'."
constraint: "Maximum 5 steps. Answer must be a number, not an explanation."
---

---
name: "Multi-Hop Passage Chaining"
description: "Decompose multi-hop QA into sequential search calls to chain evidence (IRCoT pattern)"
trigger: "Questions requiring 2+ facts from different passages; 'what X of the Y that Z'; HotpotQA, MuSiQue"
plan: |
  1. Use decompose to split into sub-questions
  2. Use search with SPECIFIC entity names from sub-question answers to find bridge facts
  3. Use lookup to find keywords in previously retrieved passages
  4. Use fact_verify to confirm the chained answer is supported by the passages
  5. Accept the short factual answer (a name, date, or number — NOT a full sentence)
pitfall: "Use SPECIFIC entity names as search queries, not full questions. Chain searches: answer from search 1 → query for search 2."
constraint: "Maximum 5 steps. Answer must be a brief phrase, not a paragraph."
---

---
name: "Code Generate-Test-Repair"
description: "Write, test, and repair iteratively (MapCoder/LDB pattern: code → test → fix with expected vs actual)"
trigger: "Code generation: 'Complete the function', 'Write a function that'; HumanEval, MBPP"
plan: |
  1. Use test_code to have M_exec write the function and run against test cases
  2. If tests fail, read the expected vs actual output to understand the bug
  3. Use python_execute to fix the specific error based on test feedback
  4. Use test_code again to verify the fix passes all tests
  5. Accept the working function code
pitfall: "Don't accept code without testing. Read test feedback carefully — it shows expected vs actual values."
constraint: "Maximum 5 steps. Accept clean function code."
---

Now generate 12 skills for the task types listed above.
IMPORTANT: Generate at least 4 skills for each task type (math_reasoning, code_generation, multi_hop_qa).
Each skill MUST specify which tools to use (e.g. python_execute, test_code, search, fact_verify, decompose, analyze, verify_answer, etc.).
DO NOT generate skills that only use analyze — combine multiple tools for better strategies.

Focus on:
- Math: python_execute with sympy for symbolic + numerical verification (SBSC/ToRA pattern)
- Code: test_code → repair with expected vs actual → re-test iteration (MapCoder/LDB pattern)
- Multi-hop QA: decompose + search chains + lookup + fact_verify (IRCoT pattern)
- Verification: verify_answer, cross_validate, self_consistency for high confidence
- Cross-task: verification and iterative refinement strategies

Output all 12 skills now:"""


    def flow_guided_evolution(
        self,
        low_flow_trajs: List[Trajectory],
        high_flow_trajs: List[Trajectory],
        current_skills: List[SkillEntry],
        skill_marginal_flows: Dict[str, float],
        mined_patterns: Optional[List[Dict]] = None,
        creation_step: int = 0,
        struggling_types: Optional[List[str]] = None,
        proven_experiences: Optional[List[Dict]] = None,
    ) -> List[SkillEntry]:

        if not low_flow_trajs and not high_flow_trajs:
            return []

        logger.info(
            f"Flow evolution: {len(low_flow_trajs)} low-flow, "
            f"{len(high_flow_trajs)} high-flow trajectories"
            + (f", struggling_types={struggling_types}" if struggling_types else "")
            + (f", {len(proven_experiences)} proven experiences" if proven_experiences else "")
        )

        prompt = self._build_evolution_prompt(
            low_flow_trajs, high_flow_trajs,
            current_skills, skill_marginal_flows, mined_patterns or [],
            struggling_types=struggling_types or [],
            proven_experiences=proven_experiences or [],
        )

        response = self.m_exec.execute(prompt, max_tokens=5000, temperature=0.6)

        response = clean_skill_output(response)
        import re as _re
        clean_response = _re.sub(r'\*{1,2}', '', response)
        new_skills = parse_skill_blocks(clean_response)


        if not new_skills and len(response) > 100:
            logger.info(f"[Evolution] Parse failed, asking M_exec to reformat ({len(response)} chars)")
            reformat_prompt = (
                "Convert the following skill descriptions into EXACTLY this YAML format.\n"
                "Each skill MUST be separated by --- on its own line.\n\n"
                "Required format:\n"
                "---\n"
                'name: "Short Name (max 7 words)"\n'
                'description: "One sentence"\n'
                'trigger: "When to use this skill"\n'
                "plan: |\n"
                "  1. First step\n"
                "  2. Second step\n"
                'pitfall: "What to avoid"\n'
                'constraint: "Max steps. Answer format."\n'
                "---\n\n"
                f"Here are the skills to reformat:\n{response[:3000]}\n\n"
                "Output ONLY the reformatted YAML blocks with --- delimiters:"
            )
            reformat_response = self.m_exec.execute(reformat_prompt, max_tokens=3000, temperature=0.1)
            clean_reformat = _re.sub(r'\*{1,2}', '', reformat_response)
            new_skills = parse_skill_blocks(clean_reformat)
            logger.info(f"[Evolution] Reformat retry: parsed {len(new_skills)} skills")

        logger.info(f"Parsed {len(new_skills)} candidate skills from evolution")


        analyzed_traj_ids = list({
            t.traj_id for t in low_flow_trajs + high_flow_trajs
            if hasattr(t, "traj_id") and t.traj_id
        })


        for skill in new_skills:
            skill.meta.source = "flow_evolution"
            skill.meta.creation_step = creation_step

            if not skill.meta.trajectory_support:
                skill.meta.trajectory_support = analyzed_traj_ids[:10]


        new_skills = self._quality_gate(new_skills, existing_skills=current_skills)


        next_id = max(
            (int(s.meta.skill_id.replace("dyn_", "")) for s in current_skills
             if s.meta.skill_id.startswith("dyn_")),
            default=-1,
        ) + 1
        assign_skill_id(new_skills, start=next_id)

        logger.info(f"Evolution produced {len(new_skills)} valid new skills")
        return new_skills

    def _build_evolution_prompt(
        self,
        low_flow_trajs: List[Trajectory],
        high_flow_trajs: List[Trajectory],
        current_skills: List[SkillEntry],
        skill_marginal_flows: Dict[str, float],
        mined_patterns: List[Dict],
        struggling_types: Optional[List[str]] = None,
        proven_experiences: Optional[List[Dict]] = None,
    ) -> str:


        failure_section = ""
        for i, traj in enumerate(low_flow_trajs[:4]):
            full_info = format_trajectory_for_skill_generation(traj)
            failure_section += f"\nFailure {i+1}:\n{full_info}\n"
            failure_section += f"Outcome: reward={traj.reward:.3f}, status={'success' if traj.reward >= 0.5 else 'failure'}\n"


        success_section = ""
        for i, traj in enumerate(high_flow_trajs[:4]):
            full_info = format_trajectory_for_skill_generation(traj)
            success_section += f"\nSuccess {i+1}:\n{full_info}\n"
            success_section += f"Outcome: reward={traj.reward:.3f}, status={'success' if traj.reward >= 0.5 else 'failure'}\n"


        existing_titles = "\n".join(
            f"  - [{s.meta.skill_id}] {s.name}"
            for s in current_skills
        ) if current_skills else "  (none)"


        experience_section = ""
        if proven_experiences:
            experience_section = "\n### Proven Action Rules (from experience store)\n"
            for exp in proven_experiences[:8]:
                experience_section += f"  - When: {exp['condition']}\n    Do: {exp['action']}\n"


        struggling_section = ""
        if struggling_types:
            struggling_section = (
                f"\n### PRIORITY: Struggling Task Types\n"
                f"  Focus new skills on: {', '.join(struggling_types)}\n"
            )

        prompt = EVOLUTION_SKILL_PROMPT.format(
            failure_section=failure_section or "(none)",
            success_section=success_section or "(none)",
            existing_titles=existing_titles,
            experience_section=experience_section,
            struggling_section=struggling_section,
        )
        return prompt


    def merge_into_living_document(
        self,
        task_type: str,
        existing_document: str,
        new_skill_contents: List[str],
    ) -> str:

        if not new_skill_contents:
            return existing_document


        new_skills_text = ""
        for i, content in enumerate(new_skill_contents):
            new_skills_text += f"--- New Skill {i+1} ---\n{content}\n\n"

        prompt = MERGE_SKILL_PROMPT.format(
            task_type=task_type,
            existing_skill=existing_document or "No existing global skill library yet.",
            new_skills=new_skills_text,
        )

        try:
            merged = self.m_exec.execute(prompt, max_tokens=3000, temperature=0.3)
            merged = clean_skill_output(merged)
            if merged and len(merged.split()) > 50:
                logger.info(
                    f"[SkillMerge] {task_type}: living document "
                    f"{'created' if not existing_document else 'updated'} "
                    f"({len(merged.split())} words)"
                )
                return merged
        except Exception as e:
            logger.warning(f"[SkillMerge] Failed for {task_type}: {e}")

        return existing_document

    def refine_living_document(
        self,
        task_type: str,
        document: str,
        word_threshold: int = 1000,
        force_refine: bool = False,
    ) -> str:

        if not document:
            return document

        word_count = len(document.split())

        if word_count < word_threshold and not force_refine:
            logger.debug(
                f"[SkillRefine] {task_type}: already compact "
                f"({word_count} words < {word_threshold}), skipping"
            )
            return document

        logger.info(f"[SkillRefine] {task_type}: starting refinement ({word_count} words)")

        prompt = SKILL_REFINE_PROMPT.format(
            word_count=word_count,
            task_type=task_type,
            skill_content=document,
        )

        try:
            refined = self.m_exec.execute(prompt, max_tokens=2000, temperature=0.3)
            refined = clean_skill_output(refined)
            if refined and len(refined.split()) > 50:
                new_count = len(refined.split())
                logger.info(
                    f"[SkillRefine] {task_type}: {word_count} → {new_count} words"
                )
                return refined
        except Exception as e:
            logger.warning(f"[SkillRefine] Failed for {task_type}: {e}")

        return document


    def cross_trajectory_critique(
        self,
        high_flow_trajs: List[Trajectory],
        low_flow_trajs: List[Trajectory],
    ) -> List[Dict]:

        if not high_flow_trajs or not low_flow_trajs:
            return []


        import math
        success_cases = []
        for traj in high_flow_trajs[:5]:
            top_steps = self._get_top_importance_indices(traj)
            steps = []
            for idx, t in enumerate(traj.turns[:6]):
                if idx in top_steps:
                    log_i = math.log(max(t.step_importance, 1e-300))
                    marker = f"★[log_I={log_i:+.1f}]"
                else:
                    marker = ""
                steps.append(
                    f"  {marker}[{t.action_type}] {(t.instruction or '')[:50]} "
                    f"→ {(t.observation or '')[:50]}"
                )
            success_cases.append(
                f"Task: {traj.task_type} | Reward: {traj.reward:.2f}\n"
                + "\n".join(steps)
            )


        failure_cases = []
        for traj in low_flow_trajs[:5]:
            top_steps = self._get_top_importance_indices(traj)
            steps = []
            for idx, t in enumerate(traj.turns[:6]):
                if idx in top_steps:
                    log_i = math.log(max(t.step_importance, 1e-300))
                    marker = f"★[log_I={log_i:+.1f}]"
                else:
                    marker = ""
                steps.append(
                    f"  {marker}[{t.action_type}] {(t.instruction or '')[:50]} "
                    f"→ {(t.observation or '')[:50]}"
                )
            failure_cases.append(
                f"Task: {traj.task_type} | Reward: {traj.reward:.2f}\n"
                + "\n".join(steps)
            )

        prompt = f"""Compare these SUCCESSFUL trajectories (high flow):
{chr(10).join(success_cases)}

With these FAILED trajectories (low flow):
{chr(10).join(failure_cases)}

Steps marked ★ dominate the flow balance error between forward policy (agent) and backward
policy (hindsight evaluator). The log_I value shows:
  - log_I > 0 (e.g. +18): agent was decisive here but backward policy disagrees — risky choice
  - log_I < 0 (e.g. -10): backward policy sees this as critical but agent wasn't confident
  - Larger |log_I| = stronger disagreement between agent and hindsight
Compare ★ steps in successful vs failed trajectories to find what actually matters.

Extract 3-5 action-level decision rules:
1. At ★ steps, what specific tool/action choices led to success vs failure?
2. What patterns in ★ steps differentiate high-flow from low-flow trajectories?
3. What common mistakes do failed trajectories share at their ★ decision points?

Output EXACTLY this format (one rule per block):
---
condition: "When [specific situation, ≤30 words]"
action: "Do [specific recommended action, ≤30 words]"
task_types: [list of task_type strings, e.g. "multi_hop_qa", "math_reasoning", "code_generation"]
---

Output 3-5 rules now:"""

        response = self.m_exec.execute(prompt, max_tokens=2000, temperature=0.4)
        logger.info(f"[Critique] Response length: {len(response)} chars")
        logger.info(f"[Critique] Response first 500 chars: {response[:500]}")


        experiences = []
        import re
        clean_response = re.sub(r'\*{1,2}', '', response)  
        blocks = re.split(r"---\s*\n", clean_response)
        for block in blocks:
            block = block.strip()
            if not block or "condition:" not in block:
                continue
            cond_text = self._extract_field(block, "condition")
            act_text = self._extract_field(block, "action")
            types_m = re.search(r'task_types:\s*\[([^\]]+)\]', block)

            if cond_text and act_text:
                task_types = []
                if types_m:
                    task_types = [t.strip().strip('"\'') for t in types_m.group(1).split(",")]
                experiences.append({
                    "condition": cond_text,
                    "action": act_text,
                    "task_types": task_types,
                })

        logger.info(f"[Critique] Extracted {len(experiences)} experiences")
        return experiences

    def backward_pattern_mine(
        self,
        trajectory: Trajectory,
        top_k: int = 3,
    ) -> List[Dict]:

        top_steps = identify_top_importance_steps(trajectory, top_k=top_k)
        patterns = []

        for idx, importance in top_steps:

            window_start = max(0, idx - 1)
            window_end = min(len(trajectory.turns), idx + 2)
            window = trajectory.turns[window_start:window_end]

            pattern = {
                "step_idx": idx,
                "importance": round(importance, 4),
                "action_types": [t.action_type for t in window],
                "skills_invoked": [t.skill_id for t in window if t.skill_id],
                "instruction_snippet": (trajectory.turns[idx].instruction or "")[:200],
                "observation_snippet": trajectory.turns[idx].observation[:200],
                "task_type": trajectory.task_type,
            }
            patterns.append(pattern)

        return patterns


    def refine_by_counterfactual(
        self,
        skill: SkillEntry,
        low_flow_invocations: List[Dict],  
        creation_step: int = 0,
    ) -> Optional[SkillEntry]:

        if not low_flow_invocations:
            return None


        counterfactual_analyses: List[Dict] = []

        if self.backward_policy is not None:
            for inv in low_flow_invocations[:5]:
                backward_context = inv.get("backward_context", "")
                action_text = inv.get("action_text", "")
                forward_lp = inv.get("forward_logprob", 0.0)
                backward_lp = inv.get("backward_logprob", 0.0)
                step_imp = inv.get("step_importance", 1.0)
                instruction = inv.get("instruction", "")
                observation = inv.get("observation", "")

                if not backward_context or not action_text:

                    counterfactual_analyses.append({
                        "instruction": instruction[:100],
                        "observation": observation[:100],
                        "reward": inv.get("reward", 0.0),
                        "step_importance": round(step_imp, 3),
                        "log_pi_theta": None,
                        "log_p_phi_actual": None,
                        "log_p_phi_best_alt": None,
                        "best_alternative": None,
                    })
                    continue


                alt_prompt = (
                    f"Context where a skill was invoked but led to poor outcome:\n"
                    f"  State context: {backward_context[:300]}\n"
                    f"  The agent chose: {action_text[:200]}\n"
                    f"  Outcome observation: {observation[:150]}\n"
                    f"  Reward: {inv.get('reward', 0):.2f}\n\n"
                    f"List exactly 3 alternative JSON actions the agent could have taken "
                    f"instead (different action_type or different instruction). "
                    f"Each on its own line, starting with 'ALT:'."
                )
                try:
                    alt_response = self.m_exec.execute(
                        alt_prompt, max_tokens=600, temperature=0.35
                    )
                    alt_actions = [
                        line.replace("ALT:", "").strip()
                        for line in alt_response.split("\n")
                        if line.strip().startswith("ALT:")
                    ][:3]
                except Exception as e:
                    logger.debug(f"M_exec alternative generation failed: {e}")
                    alt_actions = []


                best_alt: Optional[str] = None
                best_alt_lp = backward_lp  

                for alt in alt_actions:
                    try:
                        alt_lp = self.backward_policy._compute_action_logprob(
                            backward_context, alt
                        )
                        if alt_lp > best_alt_lp:
                            best_alt_lp = alt_lp
                            best_alt = alt
                    except Exception as e:
                        logger.debug(f"P_φ scoring failed for alternative: {e}")

                counterfactual_analyses.append({
                    "instruction": instruction[:100],
                    "observation": observation[:100],
                    "reward": inv.get("reward", 0.0),
                    "step_importance": round(step_imp, 3),
                    "log_pi_theta": round(forward_lp, 3),
                    "log_p_phi_actual": round(backward_lp, 3),
                    "log_p_phi_best_alt": round(best_alt_lp, 3),
                    "best_alternative": best_alt,
                })


        if counterfactual_analyses and self.backward_policy is not None:

            analysis_lines = []
            for i, a in enumerate(counterfactual_analyses):
                line = (
                    f"  [{i+1}] instruction='{a['instruction']}'\n"
                    f"       observation='{a['observation']}'\n"
                    f"       reward={a['reward']:.2f}, I(t)={a['step_importance']}\n"
                )
                if a["log_pi_theta"] is not None:
                    line += (
                        f"       log π_θ={a['log_pi_theta']:.3f}, "
                        f"log P_φ(actual)={a['log_p_phi_actual']:.3f}, "
                        f"log P_φ(best_alt)={a['log_p_phi_best_alt']:.3f}\n"
                    )
                if a["best_alternative"]:
                    line += f"       → backward flow prefers: '{a['best_alternative'][:150]}'"
                analysis_lines.append(line)

            analysis_text = "\n".join(analysis_lines)
            analysis_header = (
                "## P_φ Counterfactual Analysis\n"
                "(log π_θ > log P_φ(actual) → forward policy overconfident; "
                "if log P_φ(best_alt) > log P_φ(actual) → backward flow prefers the alternative)\n"
            )
            task_guidance = (
                "Based on the counterfactual analysis above:\n"
                "1. If best_alternative actions show a pattern (different decomposition, "
                "different instruction framing), update the plan to reflect those patterns\n"
                "2. Update pitfall to explicitly warn against actions where "
                "log π_θ >> log P_φ (forward overconfident but backward disagrees)\n"
                "3. If log P_φ(best_alt) consistently exceeds log P_φ(actual), "
                "the current plan leads the agent toward actions the backward flow rates poorly "
                "— rewrite those plan steps to align with the higher-P_φ alternatives\n"
                "4. Keep the same skill_id and name unchanged"
            )
        else:

            analysis_text = "\n".join([
                f"  - instruction='{inv.get('instruction', '')[:100]}', "
                f"observation='{inv.get('observation', '')[:100]}', "
                f"reward={inv.get('reward', 0):.2f}"
                for inv in low_flow_invocations[:5]
            ])
            analysis_header = "## Failure Cases (when this skill was used but failed):\n"
            task_guidance = (
                "Rewrite the skill to fix the identified failure patterns. "
                "Keep the same skill_id and name."
            )

        prompt = f"""Improve this skill based on failure analysis and counterfactual reasoning.

The Supervisor has 23 tools including: skill_invoke, think, plan, decompose, python_execute, test_code, analyze, search, lookup, fact_verify, ask_llm, self_consistency, verify_answer, check_answer, cross_validate, search_code, view_file, edit_file, run_tests, act, search_product, click, accept.
Plan steps MUST specify which tool to use (e.g., "Use search to find X", "Use python_execute to compute Y", "Use verify_answer to check").

## Current Skill
Name: {skill.name}
Plan: {skill.plan}
Pitfall: {skill.pitfall}

{analysis_header}
{analysis_text}

## Task
{task_guidance}

Output the improved skill in this format:

---
name: "{skill.name}"
description: "[improved if needed]"
trigger: "[refined trigger]"
plan: |
  [improved step-by-step plan — each step should instruct M_exec clearly]
pitfall: "[updated pitfall based on counterfactual analysis]"
constraint: "{skill.constraint}"
meta:
  skill_id: {skill.meta.skill_id}
  source: backward_mining
  flow_score: {skill.meta.flow_score}
  creation_step: {creation_step}
  trajectory_support: []
  usage_count: {skill.meta.usage_count}
  last_used_step: {skill.meta.last_used_step}
  success_count: {skill.meta.success_count}
---

Output the refined skill now:"""

        response = self.m_exec.execute(prompt, max_tokens=1500, temperature=0.3)
        skills = parse_skill_blocks(response)

        if skills:
            refined = skills[0]

            refined.meta.skill_id = skill.meta.skill_id
            refined.meta.source = "backward_mining"
            refined.meta.creation_step = creation_step
            refined.meta.usage_count = skill.meta.usage_count
            refined.meta.success_count = skill.meta.success_count
            refined.meta.task_types = skill.meta.task_types  

            support_ids = list({
                inv.get("traj_id", "") for inv in low_flow_invocations
                if inv.get("traj_id")
            })
            refined.meta.trajectory_support = (
                support_ids if support_ids else skill.meta.trajectory_support
            )
            return refined

        return None


    def distill_high_flow_skill(
        self,
        skill: SkillEntry,
        high_flow_trajectories: List[Trajectory],
        skill_marginal_flows: Dict[str, float],
        creation_step: int = 0,
    ) -> Optional[SkillEntry]:

        if not high_flow_trajectories:
            return None


        successful_invocations: List[Dict] = []
        for traj in high_flow_trajectories:
            for t_idx, turn in enumerate(traj.turns):
                if (
                    turn.action_type == "skill_invoke"
                    and turn.skill_id == skill.meta.skill_id
                    and turn.instruction
                ):
                    successful_invocations.append({
                        "traj_id": traj.traj_id,
                        "instruction": turn.instruction[:200],
                        "observation": turn.observation[:150],
                        "reward": traj.reward,
                        "step_importance": turn.step_importance,
                        "edge_flow": turn.edge_flow,
                        "task_type": traj.task_type,
                    })

        if not successful_invocations:

            logger.debug(
                f"distill_high_flow_skill: skill {skill.meta.skill_id} "
                f"not invoked in high-flow trajectories, skipping"
            )
            return None


        successful_invocations.sort(key=lambda x: x["step_importance"], reverse=True)
        top_invocations = successful_invocations[:6]

        examples_text = "\n".join([
            f"  [{i+1}] task_type={inv['task_type']}\n"
            f"       instruction='{inv['instruction']}'\n"
            f"       observation='{inv['observation']}'\n"
            f"       reward={inv['reward']:.2f}, I(t)={inv['step_importance']:.3f}, "
            f"edge_flow={inv['edge_flow']:.4f}"
            for i, inv in enumerate(top_invocations)
        ])

        current_flow = skill_marginal_flows.get(
            skill.meta.skill_id, skill.meta.flow_score
        )

        support_ids = list({
            inv["traj_id"] for inv in successful_invocations if inv.get("traj_id")
        })

        prompt = f"""Distill and strengthen this skill based on its successful high-flow invocations:

## Current Skill
Name: {skill.name}
Description: {skill.description}
Trigger: {skill.trigger}
Plan: {skill.plan}
Pitfall: {skill.pitfall}
Constraint: {skill.constraint}

## High-Flow Successful Invocations (F̂(s)={current_flow:.4f})
(These are the invocations where this skill led to high reward and high flow)
{examples_text}

## Task
Distill the patterns in the successful invocations into a stronger skill definition:
1. Identify recurring instruction patterns across the invocations (e.g., specific decomposition
   style, question framing, executor prompts that led to informative observations)
2. Strengthen the plan to explicitly guide toward those proven instruction patterns
3. Update the trigger to more precisely identify when this skill works best
   (reference specific task_types and question structures from the examples)
4. The improved plan should be MORE SPECIFIC and MORE ACTIONABLE, not more generic
5. Keep name, skill_id, and constraint UNCHANGED

Output the distilled skill in this format:

---
name: "{skill.name}"
description: "[refined description if more precise wording helps]"
trigger: "[refined trigger referencing patterns from successful invocations]"
plan: |
  [distilled, strengthened step-by-step plan with proven instruction patterns]
pitfall: "{skill.pitfall}"
constraint: "{skill.constraint}"
meta:
  skill_id: {skill.meta.skill_id}
  source: flow_evolution
  flow_score: {current_flow}
  creation_step: {creation_step}
  trajectory_support: {support_ids[:3]}
  usage_count: {skill.meta.usage_count}
  last_used_step: {skill.meta.last_used_step}
  success_count: {skill.meta.success_count}
---

Output the distilled skill now:"""

        response = self.m_exec.execute(prompt, max_tokens=1500, temperature=0.2)
        parsed = parse_skill_blocks(response)

        if parsed:
            distilled = parsed[0]

            distilled.meta.skill_id = skill.meta.skill_id
            distilled.meta.source = "flow_evolution"
            distilled.meta.creation_step = creation_step
            distilled.meta.usage_count = skill.meta.usage_count
            distilled.meta.success_count = skill.meta.success_count
            distilled.meta.task_types = skill.meta.task_types  
            distilled.meta.flow_score = current_flow
            distilled.meta.trajectory_support = (
                support_ids[:3] if support_ids else skill.meta.trajectory_support
            )
            return distilled

        return None


    def _quality_gate(
        self,
        new_skills: List[SkillEntry],
        existing_skills: List[SkillEntry],
    ) -> List[SkillEntry]:

        passed = []
        seen_names = set(s.name.lower() for s in existing_skills)

        for skill in new_skills:

            valid, reason = skill.validate()
            if not valid:
                logger.info(f"Skill rejected (validate): {skill.name} — {reason}")
                continue


            if skill.name.lower() in seen_names:
                logger.info(f"Skill rejected (duplicate name): {skill.name}")
                continue


            if skill.meta.source not in ("genesis",) and not skill.meta.trajectory_support:
                logger.info(
                    f"Skill rejected (no trajectory grounding): {skill.name} "
                    f"source={skill.meta.source}, trajectory_support={skill.meta.trajectory_support}"
                )
                continue


            if self._is_too_similar(skill, existing_skills + passed):
                logger.info(f"Skill rejected (too similar): {skill.name}")
                continue

            seen_names.add(skill.name.lower())
            passed.append(skill)

        logger.info(
            f"Quality gate: {len(new_skills)} input → {len(passed)} passed"
        )
        return passed

    @staticmethod
    def _is_too_similar(
        skill: SkillEntry,
        others: List[SkillEntry],
        threshold: float = EMBEDDING_SIM_THRESHOLD,
    ) -> bool:

        if not others:
            return False

        skill_text = f"{skill.name} {skill.trigger} {skill.plan}"
        other_texts = [f"{o.name} {o.trigger} {o.plan}" for o in others]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            all_texts = [skill_text] + other_texts
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,  
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            return bool(np.any(sims > threshold))

        except ImportError:

            logger.warning("sklearn not available; falling back to Jaccard similarity for dedup")

            def jaccard(text1: str, text2: str) -> float:
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                if not words1 or not words2:
                    return 0.0
                return len(words1 & words2) / len(words1 | words2)

            for other_text in other_texts:
                if jaccard(skill_text, other_text) > threshold:
                    return True
            return False
