

from __future__ import annotations

import re
import string
from typing import Callable, Dict, List, Optional, Tuple


REWARD_CAP = 1.0


EPSILON_MIN = 0.1


def normalize_answer(text: str) -> str:

    text = text.lower()
    text = text.replace("-", " ")  
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def token_f1(pred: str, gold: str) -> float:

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def token_f1_multi(pred: str, gold_candidates: List[str]) -> float:

    return max(token_f1(pred, g) for g in gold_candidates)


def extract_math_answer(text: str) -> str:


    if "boxed" in text:

        idx = text.rfind("\\boxed{")
        if idx >= 0:
            start = idx + len("\\boxed{")
            stack = 1
            ans = ""
            for c in text[start:]:
                if c == "{":
                    stack += 1
                    ans += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    ans += c
                else:
                    ans += c
            if ans:

                ans = ans.replace("\\displaystyle", "").replace("\\textstyle", "").strip()
                return ans


    text_stripped = text.strip()
    if len(text_stripped) < 100 and "boxed" not in text_stripped and any(
            cmd in text_stripped for cmd in ["\\frac", "\\sqrt", "\\pi", "\\infty", "\\begin"]):
        return text_stripped


    final_marker = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if final_marker:
        return final_marker.group(1).strip().replace(",", "")


    answer_pattern = re.search(
        r"(?:the\s+)?answer\s+is\s+[:\s]*(-?[\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "")


    _NUM_RE = r"-?[\d,]+(?:\.\d+)?"


    if len(text) < 200:
        numbers = re.findall(_NUM_RE, text)
        if numbers:
            return numbers[-1].replace(",", "")


    last_lines = "\n".join(text.strip().split("\n")[-3:])
    numbers = re.findall(_NUM_RE, last_lines)
    if numbers:
        return numbers[-1].replace(",", "")


    numbers = re.findall(_NUM_RE, text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text


def _strip_math_string(s: str) -> str:

    s = str(s).strip()

    s = s.replace("\\!", "").replace("\\ ", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\displaystyle", "")  
    s = s.replace("\\textstyle", "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("\\circ", "")
    s = s.replace("$", "").replace("\\%", "%")

    if "\\boxed{" in s:
        idx = s.find("\\boxed{")
        start = idx + len("\\boxed{")
        stack, end = 1, len(s)
        for i in range(start, len(s)):
            if s[i] == "{": stack += 1
            elif s[i] == "}":
                stack -= 1
                if stack == 0:
                    end = i
                    break
        s = s[:idx] + s[start:end] + s[end+1:]

    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s).strip()


    for _ in range(5):  
        new_s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s

    for _ in range(5):
        new_s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s

    s = s.replace("\\pi", "pi")

    s = s.replace(" ", "")
    return s


def _symbolic_equal(pred_str: str, gold_str: str) -> bool:

    try:
        from sympy import simplify, N, Rational
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    def _parse(s):
        s = s.strip()
        for parser in [parse_expr]:
            try:
                return parser(s)
            except Exception:
                continue
        return None

    a = _parse(pred_str)
    b = _parse(gold_str)
    if a is None or b is None:
        return False


    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass


    try:
        from math import isclose
        na, nb = float(N(a)), float(N(b))
        if isclose(na, nb, rel_tol=1e-4):
            return True
    except Exception:
        pass

    return False


def exact_match(pred: str, gold: str) -> float:

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)


    if pred_norm == gold_norm:
        return 1.0


    pred_num = extract_math_answer(pred)
    gold_num = extract_math_answer(gold)
    if pred_num and gold_num:
        try:
            p_val = float(pred_num.replace(",", ""))
            g_val = float(gold_num.replace(",", ""))

            if abs(p_val - g_val) < 1e-6:
                return 1.0

            from math import isclose
            for variant in [g_val, g_val / 100, g_val * 100]:
                if isclose(p_val, variant, rel_tol=1e-4):
                    return 1.0
        except (ValueError, OverflowError):
            pass

        if normalize_answer(pred_num) == normalize_answer(gold_num):
            return 1.0


    pred_math = _strip_math_string(pred_num or pred)
    gold_math = _strip_math_string(gold_num or gold)
    if pred_math and gold_math and pred_math == gold_math:
        return 1.0


    if pred_num and gold_num:
        if _symbolic_equal(pred_num, gold_num):
            return 1.0

    if len(pred) < 50 and len(gold) < 50:
        if _symbolic_equal(pred.strip(), gold.strip()):
            return 1.0


    if len(pred) < 100 and len(gold) < 100:
        pred_sympy_ready = _strip_math_string(pred)
        gold_sympy_ready = _strip_math_string(gold)
        if pred_sympy_ready and gold_sympy_ready and pred_sympy_ready != pred and pred_sympy_ready != gold:
            if _symbolic_equal(pred_sympy_ready, gold_sympy_ready):
                return 1.0

        if _symbolic_equal(pred_sympy_ready, gold.strip()) or _symbolic_equal(pred.strip(), gold_sympy_ready):
            return 1.0


    def _try_list_match(a: str, b: str) -> bool:

        def _extract_list_items(s: str) -> list:
            s = s.strip()

            for start, end in [('[', ']'), ('(', ')'), ('{', '}')]:
                if s.startswith(start) and s.endswith(end):
                    s = s[1:-1]
                    break

            if ',' in s:
                return [p.strip() for p in s.split(',') if p.strip()]
            return []

        a_items = _extract_list_items(a)
        b_items = _extract_list_items(b)
        if len(a_items) != len(b_items) or len(a_items) < 2:
            return False
        return all(exact_match(p, g) > 0.5 for p, g in zip(a_items, b_items))

    if _try_list_match(pred, gold) or (pred_num and gold_num and _try_list_match(pred_num, gold_num)):
        return 1.0


    if len(gold) > 100:
        f1 = token_f1(pred, gold)
        if f1 > 0.8:
            return 1.0
        final_match = re.search(r"####\s*(.+?)$", gold.strip(), re.MULTILINE)
        if final_match and pred_num:
            gold_final = final_match.group(1).strip()
            if normalize_answer(pred_num) == normalize_answer(gold_final):
                return 1.0

    return 0.0


def label_accuracy(pred: str, gold: str) -> float:


    label_map = {
        "supports": "supports",
        "supported": "supports",
        "true": "supports",
        "refutes": "refutes",
        "refuted": "refutes",
        "false": "refutes",
        "not_enough_info": "nei",
        "not enough info": "nei",
        "nei": "nei",
        "unknown": "nei",
    }

    pred_norm = label_map.get(normalize_answer(pred), normalize_answer(pred))
    gold_norm = label_map.get(normalize_answer(gold), normalize_answer(gold))
    return 1.0 if pred_norm == gold_norm else 0.0


def _option_label_match(pred: str, gold: str) -> float:

    def _extract_label(text: str) -> str:
        text = text.strip()

        m = re.match(r"^[(\s]*([A-Ea-e])\s*[.):\s]", text)
        if m:
            return m.group(1).upper()

        if len(text) == 1 and text.upper() in "ABCDE":
            return text.upper()

        m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Ea-e])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        if len(text) < 50:
            m = re.search(r"\b([A-Ea-e])\s*[.)\s]*$", text)
            if m:
                return m.group(1).upper()
        return ""

    pred_label = _extract_label(pred)
    gold_label = _extract_label(gold)
    if pred_label and gold_label:
        return 1.0 if pred_label == gold_label else 0.0

    return exact_match(pred, gold)


def rouge_l(pred: str, gold: str) -> float:

    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0


    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def code_test_pass_rate(pred: str, test_cases: List[str]) -> float:

    if not test_cases:
        return 0.0


    code = pred
    code_block = re.search(r"```(?:python)?\n(.*?)```", pred, re.DOTALL)
    if code_block:
        code = code_block.group(1)
    elif "def " in pred:

        def_pos = pred.find("def ")
        code = pred[def_pos:]

    passed = 0
    for test in test_cases:
        try:
            namespace: Dict = {}
            exec(code, namespace)  
            exec(test, namespace)  
            passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


def _swe_bench_reward(pred: str, gold: str, extra: Optional[Dict] = None) -> float:

    if not pred.strip():
        return 0.0
    return _patch_text_similarity(pred, gold)


_sb_api_keys: List[str] = []
_sb_api_key_idx: int = 0
_sb_eval_cache: Dict[str, float] = {}  

def _load_sb_api_keys() -> List[str]:

    global _sb_api_keys
    if _sb_api_keys:
        return _sb_api_keys
    import json, os
    keys = []

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "sb_api_keys.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                data = json.load(f)
            for entry in data.get("keys", []):
                k = entry.get("key", "").strip()
                if k:
                    keys.append(k)
        except Exception:
            pass

    env_key = os.environ.get("SWEBENCH_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    _sb_api_keys = deduped
    return _sb_api_keys


def _swe_bench_api_evaluate(instance_id: str, model_patch: str) -> Optional[float]:

    import requests, time, hashlib, logging
    logger = logging.getLogger(__name__)

    API_BASE = "https://api.swebench.com"
    keys = _load_sb_api_keys()
    if not keys:
        return None


    patch_hash = hashlib.md5(model_patch.encode()).hexdigest()[:10]
    cache_key = f"{instance_id}_{patch_hash}"
    if cache_key in _sb_eval_cache:
        score = _sb_eval_cache[cache_key]
        logger.info(f"[SWE-eval] {instance_id}: API cached={'RESOLVED' if score > 0 else 'unresolved'}")
        return score

    global _sb_api_key_idx
    run_id = f"sf_{instance_id[:20]}_{patch_hash}"

    for attempt in range(len(keys)):
        key = keys[_sb_api_key_idx % len(keys)]
        headers = {"x-api-key": key}

        try:

            submit_payload = {
                "split": "test",
                "subset": "swe-bench_verified",
                "run_id": run_id,
                "prediction": {
                    "instance_id": instance_id,
                    "model_patch": model_patch,
                    "model_name_or_path": "skillflow",
                }
            }
            resp = requests.post(f"{API_BASE}/submit", json=submit_payload, headers=headers, timeout=30)

            if resp.status_code == 429:
                logger.warning(f"[SWE-eval] Key {_sb_api_key_idx} quota exhausted, rotating...")
                _sb_api_key_idx += 1
                run_id = f"sf_{instance_id[:20]}_{patch_hash}_{_sb_api_key_idx}"
                continue
            if resp.status_code == 401:
                logger.warning(f"[SWE-eval] Key {_sb_api_key_idx} auth failed, rotating...")
                _sb_api_key_idx += 1
                continue
            if resp.status_code != 200:
                logger.warning(f"[SWE-eval] Submit {resp.status_code}: {resp.text[:100]}")
                _sb_api_key_idx += 1
                continue

            launch_data = resp.json()
            launched = launch_data.get("launched", False)


            poll_payload = {"run_id": run_id, "subset": "swe-bench_verified", "split": "test"}
            for _ in range(30):
                time.sleep(10)
                poll_resp = requests.get(f"{API_BASE}/poll-jobs", json=poll_payload, headers=headers, timeout=30)
                if poll_resp.status_code != 200:
                    break
                poll_data = poll_resp.json()
                completed = set(poll_data.get("completed", []))
                pending = set(poll_data.get("pending", [])) if "pending" not in poll_data else set()

                running = set(poll_data.get("running", []))
                if instance_id in completed or (not running and not pending):
                    break


            report_payload = {"run_id": run_id, "subset": "swe-bench_verified", "split": "test"}
            report_resp = requests.post(f"{API_BASE}/get-report", json=report_payload, headers=headers, timeout=30)
            if report_resp.status_code == 200:
                report_data = report_resp.json().get("report", {})
                resolved = report_data.get("resolved_instances", 0)
                submitted = report_data.get("submitted_instances", 0)
                score = float(resolved) / max(submitted, 1)
                _sb_eval_cache[cache_key] = score
                logger.info(f"[SWE-eval] {instance_id}: API {'RESOLVED' if score > 0 else 'unresolved'} (key={_sb_api_key_idx})")
                return score
            else:
                logger.warning(f"[SWE-eval] Report failed {report_resp.status_code}: {report_resp.text[:100]}")

        except requests.Timeout:
            logger.warning(f"[SWE-eval] {instance_id}: API timeout (key={_sb_api_key_idx})")
        except Exception as e:
            logger.warning(f"[SWE-eval] {instance_id}: API error: {e}")

        _sb_api_key_idx += 1

    logger.warning(f"[SWE-eval] {instance_id}: all {len(keys)} keys exhausted")
    return None


def _swe_bench_official_reward(pred: str, gold: str, extra: Dict) -> float:

    import logging
    import os
    logger = logging.getLogger(__name__)
    instance_id = extra.get("instance_id", "")

    if not pred.strip() or not instance_id:
        return 0.0


    if os.environ.get("SKILLFLOW_DEFER_SWE_EVAL", "").lower() in {"1", "true", "yes", "on"}:
        return 0.0

    from training.swebench_client import get_client
    client = get_client()


    timeout_s = int(os.environ.get("SWE_EVAL_TIMEOUT", "900"))
    score = client.evaluate(instance_id, pred, timeout=timeout_s)
    if score is not None:
        return score

    logger.warning(f"[SWE-eval] {instance_id}: eval server unavailable, returning 0.0")
    return 0.0


def _normalize_patch(patch: str) -> str:

    lines = []
    for line in patch.split("\n"):
        line = line.rstrip()
        if line.startswith(("diff --git", "index ", "---", "+++", "@@")):
            continue
        if line.startswith(("+", "-")):
            lines.append(line)
    return "\n".join(lines)


def _patch_text_similarity(pred: str, gold: str) -> float:

    import difflib
    pn = _normalize_patch(pred)
    gn = _normalize_patch(gold)
    if not gn:
        return 0.0
    return difflib.SequenceMatcher(None, pn.split("\n"), gn.split("\n")).ratio()


TASK_REWARD_FNS: Dict[str, Callable] = {
    "multi_hop_qa": token_f1,
    "factual_qa": token_f1,
    "fact_checking": label_accuracy,
    "math_reasoning": exact_match,
    "code_generation": _swe_bench_reward,  
    "strategy_qa": exact_match,
    "open_ended": rouge_l,
    "interactive_agent": exact_match,  
    "science_qa": lambda pred, gold: _option_label_match(pred, gold),
}


def _clean_pred_for_reward(pred: str, gold: str, task_type: str) -> str:


    pred = re.sub(r"【[^】]*】", "", pred).strip()
    pred = re.sub(r"\[\d+†[^\]]*\]", "", pred).strip()


    if pred.lstrip().startswith("[PASS]") or pred.lstrip().startswith("[FAIL]"):


        result_match = re.search(r"\[RESULT\]\s*(.+)", pred)
        if result_match:
            pred = result_match.group(1).strip()
        else:
            return ""  
    if pred.lstrip().startswith("[RESULT]"):
        pred = re.sub(r"^\[RESULT\]\s*", "", pred.lstrip()).strip()
    if pred.lstrip().startswith("[Match"):

        pred = re.sub(r"\[Match \d+\]\s*\(score=[\d.]+\)\s*\S*\n?", "", pred).strip()
    if pred.lstrip().startswith("[NO_CONTEXT]"):
        return ""


    if task_type in ("multi_hop_qa", "factual_qa", "strategy_qa") and gold:
        gold_len = max(len(g.strip()) for g in gold.split("|"))
        if len(pred) > gold_len * 5 and gold_len < 100:

            first_line = pred.split("\n")[0].strip()

            if len(first_line) > gold_len * 5:
                sent_end = re.search(r"[.!?]\s", first_line)
                if sent_end:
                    first_line = first_line[:sent_end.end()].strip()
            pred = first_line

    return pred


def compute_answer_reward(
    pred: str,
    gold: str,
    task_type: str,
    kappa: float = REWARD_CAP,
    extra: Optional[Dict] = None,
) -> float:

    if not pred.strip():
        return 0.0


    pred = _clean_pred_for_reward(pred, gold, task_type)
    if not pred:
        return 0.0


    gold_candidates = [g.strip() for g in gold.split("|") if g.strip()]
    if not gold_candidates:
        return 0.0


    _NO_ANSWER_PHRASES = {
        "unanswerable", "no answer", "cannot be determined",
        "not enough information", "cannot be answered",
        "no sufficient information", "insufficient information",
    }
    pred_lower = pred.lower()
    gold_lower = gold.lower()
    pred_is_no_answer = any(p in pred_lower for p in _NO_ANSWER_PHRASES)
    gold_is_no_answer = any(p in gold_lower for p in _NO_ANSWER_PHRASES)
    if pred_is_no_answer and gold_is_no_answer:
        return min(kappa, 1.0)

    reward_fn = TASK_REWARD_FNS.get(task_type, token_f1)


    test_cases = None
    if task_type == "code_generation" and extra:
        if "test_cases" in extra:
            test_cases = extra["test_cases"]
        elif "test" in extra:

            test_str = extra["test"]
            entry_point = extra.get("entry_point", "")
            if test_str.strip():

                if "def check(" in test_str and entry_point:
                    test_cases = [test_str + f"\ncheck({entry_point})"]
                else:

                    test_cases = [t.strip() for t in test_str.strip().split("\n") if t.strip()]
    if test_cases:
        raw = code_test_pass_rate(pred, test_cases)
    elif task_type == "code_generation" and extra and extra.get("instance_id"):

        raw = _swe_bench_official_reward(pred, gold_candidates[0], extra)
    elif len(gold_candidates) > 1:
        raw = max(reward_fn(pred, g) for g in gold_candidates)
    else:
        raw = reward_fn(pred, gold_candidates[0])


    return min(kappa, raw)


TOOL_PROCESS_REWARDS: Dict[str, float] = {
    "think": 0.01,          "plan": 0.03,           "decompose": 0.05,
    "python_execute": 0.03, "test_code": 0.04,      "analyze": 0.03,
    "search": 0.03,         "lookup": 0.02,         "fact_verify": 0.03,
    "ask_llm": 0.03,        "self_consistency": 0.04,
    "verify_answer": 0.03,  "check_answer": 0.02,   "cross_validate": 0.04,
    "list_files": 0.02,     "search_code": 0.02,    "view_file": 0.01,      "edit_file": 0.04,  "run_tests": 0.04,
    "act": 0.03,            "search_product": 0.03,  "click": 0.02,
    "accept": 0.0,

    "passage_search": 0.03, "skill_invoke": 0.05,   "direct_act": 0.03, "reflect": 0.02,
}


def compute_process_reward(turns: list) -> float:

    total = 0.0
    seen_instructions = set()

    for turn in turns:
        if getattr(turn, "parse_error", False):
            total -= 0.10
            continue

        atype = getattr(turn, "action_type", "")
        instr = getattr(turn, "instruction", "") or ""


        total += TOOL_PROCESS_REWARDS.get(atype, 0.0)


        obs = getattr(turn, "observation", "") or ""
        if atype == "edit_file" and "[OK]" in obs:
            total += 0.15  


        instr_key = instr[:100]
        if instr_key and instr_key in seen_instructions:
            total -= 0.05
        seen_instructions.add(instr_key)

    return total


def compute_skill_reward(
    answer_reward: float,
    turns: list,
    eta_skill: float = 0.3,
) -> float:

    if answer_reward <= 0:
        return 0.0

    skill_sub_reward = sum(
        0.05
        for t in turns
        if getattr(t, "action_type", "") == "skill_invoke"
        and not getattr(t, "parse_error", False)
    )
    return min(eta_skill, skill_sub_reward)


def compute_full_reward(
    pred: str,
    gold: str,
    task_type: str,
    turns: list,
    extra: Optional[Dict] = None,
    epsilon_min: float = EPSILON_MIN,
    kappa: float = REWARD_CAP,
    experience_store=None,
    reward_mode: str = "outcome_only",
) -> Tuple[float, float, float, float, float]:

    mode = str(reward_mode).lower()
    paper_mode = mode in {"outcome_only", "paper", "outcome"}

    r_answer = compute_answer_reward(pred, gold, task_type, kappa=kappa, extra=extra)
    r_process = compute_process_reward(turns)


    has_tool_use = any(
        getattr(t, "action_type", "") not in ("answer", "")
        for t in turns
    )
    if (not paper_mode) and not has_tool_use:
        r_answer = 0.0

    r_skill = compute_skill_reward(r_answer, turns)

    if paper_mode:

        r_answer = max(0.0, min(float(r_answer), 1.0))
        r_process = 0.0
        r_skill = 0.0
        r_total = r_answer
    else:

        r_total = r_answer + r_process + r_skill

    r_tilde = max(r_total + epsilon_min, epsilon_min)

    return r_total, r_answer, r_process, r_skill, r_tilde
