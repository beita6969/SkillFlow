

from __future__ import annotations

import ast
import io
import math
import multiprocessing
import re
import signal
import sys
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional


TOOL_EXEC_TIMEOUT = 30


TOOL_ACTION_TYPES = {"python_exec", "calculator", "search_context", "verify_answer", "test_code"}


TOOL_SUCCESS_REWARDS: Dict[str, float] = {
    "python_exec": 0.05,
    "calculator": 0.04,
    "search_context": 0.04,
    "verify_answer": 0.03,
    "test_code": 0.06,
}


TOOL_FAILURE_PENALTY = -0.03


def execute_tool(
    action_type: str,
    tool_args: Dict[str, Any],
    context: Optional[List] = None,
    extra: Optional[Dict] = None,
) -> str:

    dispatch = {
        "python_exec": _python_exec,
        "calculator": _calculator,
        "search_context": _search_context,
        "verify_answer": _verify_answer,
        "test_code": _test_code,
    }

    fn = dispatch.get(action_type)
    if fn is None:
        return f"[TOOL_ERROR] Unknown tool: {action_type}"

    try:
        result = fn(tool_args, context or [], extra or {})
        return result
    except Exception as e:
        return f"[TOOL_ERROR] {action_type} failed: {type(e).__name__}: {e}"


def is_tool_success(observation: str) -> bool:

    return not any(tag in observation for tag in ("[ERROR]", "[TOOL_ERROR]", "[FAIL]", "[EXCEPTION]"))


def _exec_code_in_process(code: str, result_queue: multiprocessing.Queue) -> None:

    old_stdout, old_stderr = sys.stdout, sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        sys.stdout = captured_out
        sys.stderr = captured_err
        namespace: Dict[str, Any] = {"__builtins__": __builtins__}
        exec(code, namespace)
        stdout = captured_out.getvalue()
        stderr = captured_err.getvalue()
        result_parts = []
        if stdout:
            result_parts.append(f"[STDOUT]\n{stdout}")
        if stderr:
            result_parts.append(f"[STDERR]\n{stderr}")
        if not result_parts:
            result_parts.append("[OK] Code executed successfully (no output)")
        result_queue.put("\n".join(result_parts))
    except Exception:
        stderr = captured_err.getvalue()
        tb = traceback.format_exc()
        parts = []
        if stderr:
            parts.append(f"[STDERR]\n{stderr}")
        parts.append(f"[EXCEPTION]\n{tb}")
        result_queue.put("\n".join(parts))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _python_exec(args: Dict, context: List, extra: Dict) -> str:

    code = args.get("code", "")
    if not code.strip():
        return "[ERROR] No code provided"

    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_code_in_process, args=(code, result_queue))
    proc.start()
    proc.join(timeout=TOOL_EXEC_TIMEOUT)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        return f"[ERROR] Code execution timed out after {TOOL_EXEC_TIMEOUT}s. Simplify your code or use a different approach."

    if not result_queue.empty():
        return result_queue.get()
    return "[ERROR] Code execution produced no result"


_CALC_NAMESPACE = {
    k: getattr(math, k) for k in dir(math) if not k.startswith("_")
}
_CALC_NAMESPACE.update({
    "abs": abs, "round": round, "min": min, "max": max,
    "int": int, "float": float, "sum": sum, "pow": pow,
    "True": True, "False": False, "len": len,
})


def _calculator(args: Dict, context: List, extra: Dict) -> str:

    expression = args.get("expression", "")
    if not expression.strip():
        return "[ERROR] No expression provided"

    try:
        tree = ast.parse(expression, mode="eval")
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, _CALC_NAMESPACE)
        return f"[RESULT] {expression} = {result}"
    except Exception as e:
        return f"[ERROR] Cannot evaluate '{expression}': {type(e).__name__}: {e}"


def _tokenize(text: str) -> List[str]:

    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "of", "in", "to", "for", "and", "or", "but", "on", "at",
                  "by", "with", "from", "that", "this", "it", "as", "not"}
    tokens = [
        w.lower().strip(".,;:!?\"'()[]{}—–-")
        for w in text.split()
        if len(w) > 1
    ]
    return [t for t in tokens if t and t not in stop_words]


def _extract_embedded_passages(question_text: str) -> List[Dict]:


    pattern = re.compile(r"\[([^\]]+)\]\s*(.+?)(?=\[[^\]]+\]|\Z)", re.DOTALL)
    matches = pattern.findall(question_text)
    if matches and len(matches) >= 2:
        return [{"title": title.strip(), "text": text.strip()} for title, text in matches if text.strip()]


    paragraphs = [p.strip() for p in question_text.split("\n\n") if len(p.strip()) > 50]
    if len(paragraphs) >= 2:
        return [{"title": "", "text": p} for p in paragraphs]

    return []


def _search_context(args: Dict, context: List, extra: Dict) -> str:

    query = args.get("query", "")
    top_k = int(args.get("top_k", 3))

    if not query.strip():
        return "[ERROR] No query provided"

    if not context:

        question_text = extra.get("question_text", args.get("question_text", ""))
        if question_text:
            context = _extract_embedded_passages(question_text)
        if not context:
            return "[NO_CONTEXT] No context passages available for this task"


    passages = []
    for ctx in context:
        if isinstance(ctx, dict):
            text = ctx.get("text", ctx.get("content", ctx.get("paragraph", str(ctx))))
            title = ctx.get("title", "")
            passages.append({"text": text, "title": title})
        else:
            passages.append({"text": str(ctx), "title": ""})

    if not passages:
        return "[NO_CONTEXT] No readable passages in context"


    passage_embs = extra.get("passage_embeddings")  
    embed_model = extra.get("embed_model")  

    if passage_embs is not None and embed_model is not None:
        import numpy as np
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        scores = (q_emb @ passage_embs.T)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(ranked[:top_k]):
            p = passages[idx]
            title = p["title"]
            header = f"[Match {rank+1}] (score={float(score):.3f})"
            if title:
                header += f" {title}"
            results.append(f"{header}\n{p['text']}")
        return "\n\n---\n\n".join(results) if results else "[NO_MATCH] No passages matched"


    query_tokens = _tokenize(query)
    if not query_tokens:
        return "[ERROR] Query has no searchable tokens"

    doc_freq: Counter = Counter()
    doc_tokens_list = []
    for p in passages:
        tokens = _tokenize(p["text"])
        doc_tokens_list.append(tokens)
        doc_freq.update(set(tokens))

    n_docs = len(passages)
    scored = []
    for i, (p, doc_tokens) in enumerate(zip(passages, doc_tokens_list)):
        if not doc_tokens:
            continue
        tf = Counter(doc_tokens)
        score = sum(
            tf[qt] / len(doc_tokens) * math.log(1 + n_docs / (1 + doc_freq.get(qt, 0)))
            for qt in query_tokens if qt in tf
        )
        scored.append((score, i, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for rank, (score, idx, p) in enumerate(scored[:top_k]):
        title = p["title"]
        header = f"[Match {rank+1}] (score={score:.3f})"
        if title:
            header += f" {title}"
        results.append(f"{header}\n{p['text']}")

    if not results:
        return "[NO_MATCH] No passages matched the query"

    return "\n\n---\n\n".join(results)


def _verify_answer(args: Dict, context: List, extra: Dict) -> str:

    answer = args.get("answer", "")
    check_type = args.get("check_type", "format")
    task_type = extra.get("task_type", "")

    if not answer.strip():
        return "[FAIL] Answer is empty"

    checks = {
        "numeric": _check_numeric,
        "label": _check_label,
        "code": _check_code,
        "non_empty": _check_non_empty,
        "format": lambda a, e: _check_format(a, e, task_type),
    }

    fn = checks.get(check_type, checks["format"])
    return fn(answer, extra)


def _check_numeric(answer: str, extra: Dict) -> str:

    numbers = re.findall(r"-?\d+\.?\d*", answer)

    has_math = bool(re.search(r"[=<>]|\\frac|\\sqrt|[a-z]\s*=", answer))
    if not numbers and not has_math:
        return f"[FAIL] No numeric or math expression found in answer: '{answer[:200]}'"
    if numbers:
        return f"[PASS] Numeric value(s) found: {', '.join(numbers[:5])}"
    return f"[PASS] Math expression found: '{answer.strip()[:100]}'"


def _check_label(answer: str, extra: Dict) -> str:
    valid_labels = {
        "supports", "refutes", "not enough info", "nei",
        "true", "false", "yes", "no",
        "entailment", "contradiction", "neutral",
    }
    normalized = answer.strip().lower()
    if normalized in valid_labels:
        return f"[PASS] Valid label: '{normalized}'"
    for label in valid_labels:
        if label in normalized:
            return f"[PASS] Label detected: '{label}' in answer"
    return f"[FAIL] No valid label found in: '{answer[:200]}'. Expected one of: {', '.join(sorted(valid_labels))}"


def _check_code(answer: str, extra: Dict) -> str:
    code = answer
    code_block = re.search(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    if code_block:
        code = code_block.group(1)
    try:
        ast.parse(code)
        return f"[PASS] Code parses successfully ({len(code.splitlines())} lines)"
    except SyntaxError as e:
        return f"[FAIL] Syntax error at line {e.lineno}: {e.msg}"


def _check_non_empty(answer: str, extra: Dict) -> str:
    words = answer.split()
    if len(words) < 1:
        return "[FAIL] Answer is empty"
    if len(words) < 3 and len(answer) < 10:
        return f"[WARN] Answer very short ({len(words)} words): '{answer}'"
    return f"[PASS] Answer has {len(words)} words, {len(answer)} chars"


def _check_format(answer: str, extra: Dict, task_type: str) -> str:
    if task_type in ("math_reasoning",):
        return _check_numeric(answer, extra)
    elif task_type in ("fact_checking",):
        return _check_label(answer, extra)
    elif task_type in ("code_generation",):
        return _check_code(answer, extra)
    elif task_type in ("multi_hop_qa",):

        if answer.strip().startswith("[Match") or answer.strip().startswith("[NO_"):
            return "[FAIL] Answer looks like raw tool output, not an extracted answer. State the answer directly."
        return _check_non_empty(answer, extra)
    else:
        return _check_non_empty(answer, extra)


def _test_code(args: Dict, context: List, extra: Dict) -> str:

    code = args.get("code", "")
    if not code.strip():
        return "[ERROR] No code provided"


    code_block = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    if code_block:
        code = code_block.group(1)


    test_cases = args.get("test_cases", [])
    if isinstance(test_cases, str):
        test_cases = [t.strip() for t in test_cases.strip().split("\n") if t.strip()]

    if not test_cases:
        test_str = extra.get("test_list", extra.get("test", ""))
        if isinstance(test_str, str) and test_str:
            test_cases = [t.strip() for t in test_str.strip().split("\n") if t.strip()]
        elif isinstance(test_str, list):
            test_cases = test_str

    if not test_cases:

        try:
            compile(code, "<test>", "exec")
            return "[WARN] No test cases available. Code compiles successfully."
        except SyntaxError as e:
            return f"[FAIL] No test cases, and code has syntax error at line {e.lineno}: {e.msg}"


    passed = 0
    total = len(test_cases)
    details = []

    def _run_single_test(code_str: str, test_str: str, q: multiprocessing.Queue) -> None:
        try:
            namespace: Dict[str, Any] = {}
            exec(code_str, namespace)
            exec(test_str, namespace)
            q.put(("PASS", ""))
        except AssertionError as e:
            q.put(("FAIL", str(e)))
        except Exception as e:
            q.put(("ERROR", f"{type(e).__name__}: {e}"))

    for i, test in enumerate(test_cases):
        q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_run_single_test, args=(code, test, q))
        proc.start()
        proc.join(timeout=TOOL_EXEC_TIMEOUT)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
            details.append(f"  Test {i+1}: TIMEOUT ({TOOL_EXEC_TIMEOUT}s)")
        elif not q.empty():
            status_str, msg = q.get()
            if status_str == "PASS":
                passed += 1
                details.append(f"  Test {i+1}: PASS")
            elif status_str == "FAIL":
                details.append(f"  Test {i+1}: FAIL (assertion: {msg})")
            else:
                details.append(f"  Test {i+1}: ERROR ({msg})")
        else:
            details.append(f"  Test {i+1}: ERROR (no result)")

    rate = passed / total
    status = "PASS" if rate == 1.0 else "PARTIAL" if rate > 0 else "FAIL"

    result_lines = [
        f"[{status}] {passed}/{total} tests passed ({rate:.0%})",
        *details,
    ]
    return "\n".join(result_lines)
