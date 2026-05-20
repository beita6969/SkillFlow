"""
SkillFlow 通用奖励函数。

默认奖励与论文附录保持一致：
  R(τ) = outcome metric ∈ [0, 1]
  R̃(τ) = R(τ) + ε_min  （GFlowNet 正支撑平滑）

每个 task_type 有独立的 answer 奖励函数。
"""

from __future__ import annotations

import re
import string
from typing import Callable, Dict, List, Optional, Tuple

# 论文 §3.1：无预定义工具，不导入工具模块

# 奖励上限；默认 outcome-only 路径会 clamp 到 [0,1]。
REWARD_CAP = 1.0

# ε_min：GFlowNet 要求 R̃ > 0
EPSILON_MIN = 0.1


# ──────────────────────────────────────────────────────
# Answer 评估函数
# ──────────────────────────────────────────────────────


def normalize_answer(text: str) -> str:
    """标准化答案文本用于比较（小写、去标点、去冠词、去连字符）"""
    text = text.lower()
    text = text.replace("-", " ")  # 连字符处理
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1（用于 QA 任务）"""
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
    """多候选答案 F1（取 max）"""
    return max(token_f1(pred, g) for g in gold_candidates)


def extract_math_answer(text: str) -> str:
    """从自然语言文本中提取数学答案。

    提取优先级：
    0. 如果文本本身就是 LaTeX 数学表达式，直接返回
    1. \\boxed{...} 格式（MATH 标准，支持嵌套花括号）
    2. "#### N" 格式（GSM8K 标准）
    3. "The answer is N" / "answer is N" 自然语言
    4. 文本最后一行的数字（短文本 <200 chars）
    5. 文本最后出现的数字（长文本 fallback）
    """
    # 1. \boxed{...} — 最高优先级，用 stack 提取（处理嵌套花括号，参考 ToRA parser）
    # 必须在 LaTeX 早返回之前，否则 \boxed{\frac{...}} 会被直接返回而不提取
    if "boxed" in text:
        # 找最后一个 \boxed{ 的位置
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
                # 去掉 \displaystyle 等 LaTeX 修饰
                ans = ans.replace("\\displaystyle", "").replace("\\textstyle", "").strip()
                return ans

    # 1.5. 如果文本本身是 LaTeX 数学表达式（短文本 + 含 \frac 等，无 boxed），直接返回
    text_stripped = text.strip()
    if len(text_stripped) < 100 and "boxed" not in text_stripped and any(
            cmd in text_stripped for cmd in ["\\frac", "\\sqrt", "\\pi", "\\infty", "\\begin"]):
        return text_stripped

    # 2. "#### N" (GSM8K final answer marker)
    final_marker = re.search(r"####\s*(.+?)$", text.strip(), re.MULTILINE)
    if final_marker:
        return final_marker.group(1).strip().replace(",", "")

    # 3. "The answer is N" / "answer is N" / "equals N"
    answer_pattern = re.search(
        r"(?:the\s+)?answer\s+is\s+[:\s]*(-?[\d,]+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    if answer_pattern:
        return answer_pattern.group(1).replace(",", "")

    # 数字匹配模式（小数点后必须有数字，避免句号被当小数点）
    _NUM_RE = r"-?[\d,]+(?:\.\d+)?"

    # 4. 对短文本（<200 chars），取最后的数字
    if len(text) < 200:
        numbers = re.findall(_NUM_RE, text)
        if numbers:
            return numbers[-1].replace(",", "")

    # 5. 对长文本（证明过程），从最后两行中提取数字（避免从中间步骤取错）
    last_lines = "\n".join(text.strip().split("\n")[-3:])
    numbers = re.findall(_NUM_RE, last_lines)
    if numbers:
        return numbers[-1].replace(",", "")

    # 6. 全文 fallback
    numbers = re.findall(_NUM_RE, text)
    if numbers:
        return numbers[-1].replace(",", "")

    return text


def _strip_math_string(s: str) -> str:
    """
    归一化数学答案字符串（参考 ToRA strip_string）。
    处理 LaTeX、分数、单位等格式差异。
    """
    s = str(s).strip()
    # 去掉 LaTeX 噪音（参考 ToRA strip_string 完整列表）
    s = s.replace("\\!", "").replace("\\ ", "").replace("\\,", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\displaystyle", "")  # 关键：去掉 \displaystyle
    s = s.replace("\\textstyle", "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("^{\\circ}", "").replace("\\circ", "")
    s = s.replace("$", "").replace("\\%", "%")
    # 去掉 \boxed{...} — 用 stack 匹配（不能用 rstrip 会破坏 \frac{}{} ）
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
    # 去掉 \text{...} \mathrm{...} 单位
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    s = re.sub(r"\\mathrm\{[^}]*\}", "", s).strip()
    # LaTeX \sqrt{x} → sqrt(x)（先转 sqrt，因为 sqrt 可能在 frac 里）
    # 用循环处理嵌套 \sqrt
    for _ in range(5):  # 最多5层嵌套
        new_s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
        if new_s == s:
            break
        s = new_s
    # LaTeX \frac{a}{b} → (a)/(b)（循环处理嵌套）
    for _ in range(5):
        new_s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s
    # LaTeX \pi → pi
    s = s.replace("\\pi", "pi")
    # 去空格
    s = s.replace(" ", "")
    return s


def _symbolic_equal(pred_str: str, gold_str: str) -> bool:
    """
    用 sympy 判断两个数学表达式是否等价（参考 ToRA grader.symbolic_equal）。
    处理：1/2 == 0.5, sqrt(2)/2 == 2^(-1/2), 等等。
    """
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

    # 代数等价：simplify(a - b) == 0
    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # 数值近似等价
    try:
        from math import isclose
        na, nb = float(N(a)), float(N(b))
        if isclose(na, nb, rel_tol=1e-4):
            return True
    except Exception:
        pass

    return False


def exact_match(pred: str, gold: str) -> float:
    """
    增强版精确匹配（参考 ToRA math_equal 多策略匹配）。

    匹配策略链：
    1. 归一化字符串比较
    2. 数值提取 + 浮点比较（含百分比变体）
    3. 数学字符串归一化比较（LaTeX 等）
    4. sympy 符号等价（1/2 == 0.5, sqrt(2)/2 等）
    5. 元组/列表逐元素比较
    6. 长 gold 文本的 token_f1 fallback
    """
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    # 策略 1: 归一化字符串比较
    if pred_norm == gold_norm:
        return 1.0

    # 策略 2: 数字提取 + 数值比较
    pred_num = extract_math_answer(pred)
    gold_num = extract_math_answer(gold)
    if pred_num and gold_num:
        try:
            p_val = float(pred_num.replace(",", ""))
            g_val = float(gold_num.replace(",", ""))
            # 直接比较
            if abs(p_val - g_val) < 1e-6:
                return 1.0
            # 百分比变体：0.5 vs 50
            from math import isclose
            for variant in [g_val, g_val / 100, g_val * 100]:
                if isclose(p_val, variant, rel_tol=1e-4):
                    return 1.0
        except (ValueError, OverflowError):
            pass
        # 字符串比较
        if normalize_answer(pred_num) == normalize_answer(gold_num):
            return 1.0

    # 策略 3: 数学字符串归一化比较（LaTeX 格式差异）
    pred_math = _strip_math_string(pred_num or pred)
    gold_math = _strip_math_string(gold_num or gold)
    if pred_math and gold_math and pred_math == gold_math:
        return 1.0

    # 策略 4: sympy 符号等价（对短文本直接尝试原始文本比较，避免 extract 破坏分数等表达式）
    if pred_num and gold_num:
        if _symbolic_equal(pred_num, gold_num):
            return 1.0
    # 也对原始短文本尝试 sympy（处理 "0.5" vs "1/2" 这类 extract 会破坏的情况）
    if len(pred) < 50 and len(gold) < 50:
        if _symbolic_equal(pred.strip(), gold.strip()):
            return 1.0

    # 策略 4b: LaTeX 转 sympy 后比较（处理 \frac{a}{b}, \sqrt{x} 等）
    # 先用 _strip_math_string 将 LaTeX 转换为 sympy 可解析的形式
    if len(pred) < 100 and len(gold) < 100:
        pred_sympy_ready = _strip_math_string(pred)
        gold_sympy_ready = _strip_math_string(gold)
        if pred_sympy_ready and gold_sympy_ready and pred_sympy_ready != pred and pred_sympy_ready != gold:
            if _symbolic_equal(pred_sympy_ready, gold_sympy_ready):
                return 1.0
        # 交叉比较：pred 原始形式 vs gold 归一化（反之亦然）
        if _symbolic_equal(pred_sympy_ready, gold.strip()) or _symbolic_equal(pred.strip(), gold_sympy_ready):
            return 1.0

    # 策略 5: 元组/列表逐元素比较（[a,b] vs [c,d]）
    # 修复：也处理 "[1,2,3]" vs "1,2,3" (含/不含括号)
    def _try_list_match(a: str, b: str) -> bool:
        """尝试将 a、b 解析为逗号分隔的列表并逐元素匹配"""
        def _extract_list_items(s: str) -> list:
            s = s.strip()
            # 去掉外层括号
            for start, end in [('[', ']'), ('(', ')'), ('{', '}')]:
                if s.startswith(start) and s.endswith(end):
                    s = s[1:-1]
                    break
            # 分割
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

    # 策略 6: 长 gold fallback
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
    """
    标签分类准确率（用于 FEVER 事实核查）。
    支持 SUPPORTS/REFUTES/NOT_ENOUGH_INFO 等标准标签。
    """
    # 标签归一化
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
    """
    Multiple-choice option label matching (science_qa / GPQA).
    Extracts the option letter (A-E) from both pred and gold.
    Handles: "D", "D.", "(D)", "The answer is D", "Answer: D", etc.
    """
    def _extract_label(text: str) -> str:
        text = text.strip()
        # "A" or "A." or "A. some text" or "(A)" → "A"
        m = re.match(r"^[(\s]*([A-Ea-e])\s*[.):\s]", text)
        if m:
            return m.group(1).upper()
        # Single letter answer
        if len(text) == 1 and text.upper() in "ABCDE":
            return text.upper()
        # "The answer is D" / "answer is D" / "Answer: D"
        m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Ea-e])\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Last single letter in short text (< 50 chars)
        if len(text) < 50:
            m = re.search(r"\b([A-Ea-e])\s*[.)\s]*$", text)
            if m:
                return m.group(1).upper()
        return ""

    pred_label = _extract_label(pred)
    gold_label = _extract_label(gold)
    if pred_label and gold_label:
        return 1.0 if pred_label == gold_label else 0.0
    # Fallback to exact_match
    return exact_match(pred, gold)


def rouge_l(pred: str, gold: str) -> float:
    """
    ROUGE-L F1（最长公共子序列 token level）。
    用于开放生成任务。
    """
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # DP LCS
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
    """
    代码测试通过率（用于 MBPP）。
    安全沙盒：exec 每个测试用例，统计通过率。
    """
    if not test_cases:
        return 0.0

    # 提取代码块：优先 ```python，次选 def 开头的连续行
    code = pred
    code_block = re.search(r"```(?:python)?\n(.*?)```", pred, re.DOTALL)
    if code_block:
        code = code_block.group(1)
    elif "def " in pred:
        # 从第一个 def 开始提取到文本末尾（去掉前面的描述文字）
        def_pos = pred.find("def ")
        code = pred[def_pos:]

    passed = 0
    for test in test_cases:
        try:
            namespace: Dict = {}
            exec(code, namespace)  # noqa: S102
            exec(test, namespace)  # noqa: S102
            passed += 1
        except Exception:
            pass

    return passed / len(test_cases)


def _swe_bench_reward(pred: str, gold: str, extra: Optional[Dict] = None) -> float:
    """
    Local fallback for patch-style code-generation rewards.

    For SWE-bench instances with an ``instance_id``, ``compute_answer_reward``
    routes to ``_swe_bench_official_reward`` instead. This fallback only scores
    the submitted diff text against the training label when no official
    evaluator is available; no evaluator-only artifacts are shown to the policy.
    """
    if not pred.strip():
        return 0.0
    return _patch_text_similarity(pred, gold)


# ── SWE-bench API 多 key 管理 ──────────────────────────────────
_sb_api_keys: List[str] = []
_sb_api_key_idx: int = 0
_sb_eval_cache: Dict[str, float] = {}  # patch_hash → score

def _load_sb_api_keys() -> List[str]:
    """Load API keys from configs/sb_api_keys.json + env var."""
    global _sb_api_keys
    if _sb_api_keys:
        return _sb_api_keys
    import json, os
    keys = []
    # 1) From config file
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
    # 2) From env var (always include, may duplicate but dedup below)
    env_key = os.environ.get("SWEBENCH_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)
    # Dedup preserving order
    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    _sb_api_keys = deduped
    return _sb_api_keys


def _swe_bench_api_evaluate(instance_id: str, model_patch: str) -> Optional[float]:
    """
    Direct SWE-bench API evaluation with multi-key rotation.
    Returns score (0.0 or 1.0) on success, None if all keys exhausted.
    """
    import requests, time, hashlib, logging
    logger = logging.getLogger(__name__)

    API_BASE = "https://api.swebench.com"
    keys = _load_sb_api_keys()
    if not keys:
        return None

    # Cache: same patch → same result
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
            # 1) Submit prediction
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

            # 2) Poll until complete (max 5 min)
            poll_payload = {"run_id": run_id, "subset": "swe-bench_verified", "split": "test"}
            for _ in range(30):
                time.sleep(10)
                poll_resp = requests.get(f"{API_BASE}/poll-jobs", json=poll_payload, headers=headers, timeout=30)
                if poll_resp.status_code != 200:
                    break
                poll_data = poll_resp.json()
                completed = set(poll_data.get("completed", []))
                pending = set(poll_data.get("pending", [])) if "pending" not in poll_data else set()
                # Check if our instance is done
                running = set(poll_data.get("running", []))
                if instance_id in completed or (not running and not pending):
                    break

            # 3) Get report
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
    """SWE-bench 评估 via 自建 Docker 评估服务器（无配额限制）。"""
    import logging
    import os
    logger = logging.getLogger(__name__)
    instance_id = extra.get("instance_id", "")

    if not pred.strip() or not instance_id:
        return 0.0

    # Evaluation-only speed path: let the rollout finish producing all patches
    # first, then score saved predictions in parallel from the eval runner.
    # This does not change agent observations or expose evaluator-only signals; it
    # only avoids blocking generation on each remote Docker job.
    if os.environ.get("SKILLFLOW_DEFER_SWE_EVAL", "").lower() in {"1", "true", "yes", "on"}:
        return 0.0

    from training.swebench_client import get_client
    client = get_client()
    # The remote SWE harness can queue Docker jobs behind other evaluations.
    # Treating a slow but valid evaluation as 0 would corrupt metrics, so use a
    # generous default while keeping it configurable for quick smoke tests.
    timeout_s = int(os.environ.get("SWE_EVAL_TIMEOUT", "900"))
    score = client.evaluate(instance_id, pred, timeout=timeout_s)
    if score is not None:
        return score

    logger.warning(f"[SWE-eval] {instance_id}: eval server unavailable, returning 0.0")
    return 0.0




def _normalize_patch(patch: str) -> str:
    """Normalize patch for comparison: keep only +/- content lines."""
    lines = []
    for line in patch.split("\n"):
        line = line.rstrip()
        if line.startswith(("diff --git", "index ", "---", "+++", "@@")):
            continue
        if line.startswith(("+", "-")):
            lines.append(line)
    return "\n".join(lines)


def _patch_text_similarity(pred: str, gold: str) -> float:
    """Fallback: compare patch text similarity when code_files unavailable."""
    import difflib
    pn = _normalize_patch(pred)
    gn = _normalize_patch(gold)
    if not gn:
        return 0.0
    return difflib.SequenceMatcher(None, pn.split("\n"), gn.split("\n")).ratio()


# 任务类型 → 奖励函数映射
TASK_REWARD_FNS: Dict[str, Callable] = {
    "multi_hop_qa": token_f1,
    "factual_qa": token_f1,
    "fact_checking": label_accuracy,
    "math_reasoning": exact_match,
    "code_generation": _swe_bench_reward,  # SWE-bench: 基于 edit 行为而非文本匹配
    "strategy_qa": exact_match,
    "open_ended": rouge_l,
    "interactive_agent": exact_match,  # ALFWorld/WebShop: binary success
    "science_qa": lambda pred, gold: _option_label_match(pred, gold),
}


# ──────────────────────────────────────────────────────
# 主奖励计算
# ──────────────────────────────────────────────────────


def _clean_pred_for_reward(pred: str, gold: str, task_type: str) -> str:
    """清洗 pred 中的工具输出噪音，提取有效答案部分。"""
    # 1. 去掉执行器引用标记和 markdown 噪音
    pred = re.sub(r"【[^】]*】", "", pred).strip()
    pred = re.sub(r"\[\d+†[^\]]*\]", "", pred).strip()

    # 2. 工具输出前缀清洗（verify_answer / search_context / calculator）
    # [PASS] Code parses successfully (7 lines) → 不是答案
    # [FAIL] No numeric found → 不是答案
    # [RESULT] 2664.0 → 提取数值
    # [Match 1] (score=0.95) Title\nContent → 提取 Content
    # [NO_CONTEXT] ... → 不是答案
    if pred.lstrip().startswith("[PASS]") or pred.lstrip().startswith("[FAIL]"):
        # 这是 verify_answer 工具的输出，不是实际答案
        # 尝试提取 [RESULT] 后的值
        result_match = re.search(r"\[RESULT\]\s*(.+)", pred)
        if result_match:
            pred = result_match.group(1).strip()
        else:
            return ""  # 无法恢复
    if pred.lstrip().startswith("[RESULT]"):
        pred = re.sub(r"^\[RESULT\]\s*", "", pred.lstrip()).strip()
    if pred.lstrip().startswith("[Match"):
        # 去掉所有 [Match N] (score=X.XXX) Title\n 前缀
        pred = re.sub(r"\[Match \d+\]\s*\(score=[\d.]+\)\s*\S*\n?", "", pred).strip()
    if pred.lstrip().startswith("[NO_CONTEXT]"):
        return ""

    # 3. multi_hop_qa / factual_qa: 如果 pred 很长但 gold 很短，提取首行作为答案
    # 避免 pred="Days (film)\nSeven Days () is a 2007..." vs gold="stuntman" 的 F1 稀释
    if task_type in ("multi_hop_qa", "factual_qa", "strategy_qa") and gold:
        gold_len = max(len(g.strip()) for g in gold.split("|"))
        if len(pred) > gold_len * 5 and gold_len < 100:
            # 取第一行或第一句
            first_line = pred.split("\n")[0].strip()
            # 如果第一行也很长，取第一句
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
    """
    计算 outcome reward R_answer。

    默认直接使用各任务的 raw metric，并 clamp 到 [0, 1]，与论文附录的
    outcome-only 标量奖励设置一致。
    """
    if not pred.strip():
        return 0.0

    # 清洗 pred 中的工具输出噪音
    pred = _clean_pred_for_reward(pred, gold, task_type)
    if not pred:
        return 0.0

    # 多候选答案（用 | 分隔）
    gold_candidates = [g.strip() for g in gold.split("|") if g.strip()]
    if not gold_candidates:
        return 0.0

    # "无法回答"等价检测：gold 和 pred 都表示"无答案"时给满分
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

    # 代码生成特殊处理：传入测试用例
    # FlowSteer 用 "test" 字段（HumanEval/MBPP），旧格式用 "test_cases"
    test_cases = None
    if task_type == "code_generation" and extra:
        if "test_cases" in extra:
            test_cases = extra["test_cases"]
        elif "test" in extra:
            # FlowSteer 格式：test 是一个字符串（可能含多个 assert 或 check 函数）
            test_str = extra["test"]
            entry_point = extra.get("entry_point", "")
            if test_str.strip():
                # HumanEval 格式：完整的 check 函数 → 作为单个测试用例
                if "def check(" in test_str and entry_point:
                    test_cases = [test_str + f"\ncheck({entry_point})"]
                else:
                    # MBPP 格式：多个 assert 语句
                    test_cases = [t.strip() for t in test_str.strip().split("\n") if t.strip()]
    if test_cases:
        raw = code_test_pass_rate(pred, test_cases)
    elif task_type == "code_generation" and extra and extra.get("instance_id"):
        # SWE-bench: 官方评估 — 在 conda env 中运行 FAIL_TO_PASS 测试
        raw = _swe_bench_official_reward(pred, gold_candidates[0], extra)
    elif len(gold_candidates) > 1:
        raw = max(reward_fn(pred, g) for g in gold_candidates)
    else:
        raw = reward_fn(pred, gold_candidates[0])

    # 直接使用 raw metric，不做基准分数归一化；所有任务 reward clamp 到 [0,1]。
    return min(kappa, raw)


# v3: 22 tools process reward lookup table
TOOL_PROCESS_REWARDS: Dict[str, float] = {
    "think": 0.01,          "plan": 0.03,           "decompose": 0.05,
    "python_execute": 0.03, "test_code": 0.04,      "analyze": 0.03,
    "search": 0.03,         "lookup": 0.02,         "fact_verify": 0.03,
    "ask_llm": 0.03,        "self_consistency": 0.04,
    "verify_answer": 0.03,  "check_answer": 0.02,   "cross_validate": 0.04,
    "list_files": 0.02,     "search_code": 0.02,    "view_file": 0.01,      "edit_file": 0.04,  "run_tests": 0.04,
    "act": 0.03,            "search_product": 0.03,  "click": 0.02,
    "accept": 0.0,
    # Legacy aliases
    "passage_search": 0.03, "skill_invoke": 0.05,   "direct_act": 0.03, "reflect": 0.02,
}


def compute_process_reward(turns: list) -> float:
    """
    计算 R_process（论文 §4.3 过程奖励，per-step shaping）。

    R_process = Σ r_t^step

    v3: 使用 TOOL_PROCESS_REWARDS 查表，不再硬编码。
      -0.10 PARSE_ERROR
      -0.05 重复相同 instruction（防止无效循环）

    Args:
        turns: List of Turn-like objects (with action_type, parse_error, instruction fields)
    """
    total = 0.0
    seen_instructions = set()

    for turn in turns:
        if getattr(turn, "parse_error", False):
            total -= 0.10
            continue

        atype = getattr(turn, "action_type", "")
        instr = getattr(turn, "instruction", "") or ""

        # Lookup table reward
        total += TOOL_PROCESS_REWARDS.get(atype, 0.0)

        # 成功 edit_file 额外奖励（code_generation 的核心行为）
        obs = getattr(turn, "observation", "") or ""
        if atype == "edit_file" and "[OK]" in obs:
            total += 0.15  # 成功编辑是 SWE-bench 最有价值的动作

        # 重复 instruction 惩罚
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
    """
    计算 R_skill（仅当 answer_reward > 0 时触发，§Eq.12）。

    R_skill = I{R_answer > 0} · min(η_skill, Σ r_t^skill)
    r_t^skill = +0.05 per effective skill_invoke
    """
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
    """
    论文对齐 reward 设计。

    默认 `outcome_only` 对齐论文附录：R(τ)=R_answer ∈ [0,1]，
    R̃(τ)=max(R(τ)+ε_min, ε_min)。

    兼容模式 `shaped` 保留早期工程实现：
    R(τ) = R_answer + R_process + R_skill
    R̃(τ) = max(R(τ) + ε_min, ε_min)

    `shaped` 仅用于兼容旧实验；论文对齐配置使用 `outcome_only`。

    Returns:
        (R_total, R_answer, R_process, R_skill, R_tilde)
    """
    mode = str(reward_mode).lower()
    paper_mode = mode in {"outcome_only", "paper", "outcome"}

    r_answer = compute_answer_reward(pred, gold, task_type, kappa=kappa, extra=extra)
    r_process = compute_process_reward(turns)

    # Evidence Gate: 没使用工具就不给 answer reward
    # 论文 §4.1: system prompt 要求 "ALWAYS use tools"，reward 与 prompt 一致
    has_tool_use = any(
        getattr(t, "action_type", "") not in ("answer", "")
        for t in turns
    )
    if (not paper_mode) and not has_tool_use:
        r_answer = 0.0

    r_skill = compute_skill_reward(r_answer, turns)

    if paper_mode:
        # 论文附录：outcome-based scalar in [0,1]。
        r_answer = max(0.0, min(float(r_answer), 1.0))
        r_process = 0.0
        r_skill = 0.0
        r_total = r_answer
    else:
        # 兼容旧实验：answer + process + skill shaping。
        r_total = r_answer + r_process + r_skill
    # GFlowNet 正支撑：R̃ = R + ε_min，保证 R̃ > 0
    r_tilde = max(r_total + epsilon_min, epsilon_min)

    return r_total, r_answer, r_process, r_skill, r_tilde
