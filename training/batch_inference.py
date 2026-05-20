

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


_thread_local = threading.local()


_sglang_pause_event = threading.Event()
_pause_wait_start_times: Dict[int, float] = {}  


_adapter_name_lock = threading.Lock()
_current_adapter_name: str = "theta_live"  


def get_current_adapter_name() -> str:

    with _adapter_name_lock:
        return _current_adapter_name


def set_current_adapter_name(name: str) -> None:

    global _current_adapter_name
    with _adapter_name_lock:
        _current_adapter_name = name


def _resolve_model(model: str) -> str:

    if ":" in model:
        base = model.split(":", 1)[0]
    else:
        base = model

    if base == "supervisor_theta":
        return f"{base}:{get_current_adapter_name()}"
    return model


def _wait_if_paused():

    if _sglang_pause_event.is_set():
        tid = threading.get_ident()
        _pause_wait_start_times[tid] = time.time()
        while _sglang_pause_event.is_set():
            time.sleep(0.05)
        dt = time.time() - _pause_wait_start_times.pop(tid, time.time())
        if dt > 1.0:  
            logger.debug(f"[sglang_pause] worker {tid} waited {dt:.1f}s")


def _resolve_api_key(api_key: str = "") -> str:

    return api_key or os.environ.get("SUPERVISOR_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY")


def _get_client(api_base: str, api_key: str = "") -> OpenAI:

    import httpx
    api_key = _resolve_api_key(api_key)
    key = f"{api_base}_{api_key}"
    clients = getattr(_thread_local, "clients", None)
    if clients is None:
        clients = {}
        _thread_local.clients = clients
    if key not in clients:
        clients[key] = OpenAI(
            base_url=api_base, api_key=api_key, timeout=300.0,
            http_client=httpx.Client(proxy=None, timeout=300.0),
        )
    return clients[key]


SUPERVISOR_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "skill_invoke",
            "description": "Invoke a learned skill by ID to receive its strategy before continuing with regular tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "The ID of the learned skill to invoke"},
                },
                "required": ["skill_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": "Have the AI executor create a detailed step-by-step plan for solving the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The goal to plan for"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decompose",
            "description": "Break a complex problem into 2-3 simpler sub-questions that can be solved step by step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The complex problem to decompose"},
                },
                "required": ["problem"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Describe a computation in natural language; M_exec writes and runs the Python. Do not pass source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "NL description of what to compute"},
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "test_code",
            "description": "Have the executor write a Python function, then run it against test cases. Returns detailed pass/fail results with expected vs actual values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "Describe the function to implement"},
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": "Have the AI executor analyze data or reason about a specific aspect of the problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "What to analyze or reason about"},
                    "data": {"type": "string", "description": "Optional data context to analyze"},
                },
                "required": ["instruction"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search context passages using hybrid BM25+semantic matching. Returns top matches with scores. IMPORTANT: Use specific entity names, not generic phrases. If results say [REPEATED], all results were already seen — use 'lookup' to find details in existing results, or provide your answer. If [NO_MATCH], try different keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — use specific entity names or keywords"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a keyword in previously retrieved documents. Searches through all observations from prior search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "The keyword or phrase to look up"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fact_verify",
            "description": "Verify a factual claim against context passages. Returns SUPPORTED/NOT_SUPPORTED/PARTIAL with evidence excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The factual claim to verify"},
                },
                "required": ["claim"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "ask_llm",
            "description": "Ask a powerful LLM to directly answer the question based on all evidence and computations gathered so far. Best when you have enough information and just need the final reasoning step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to answer, including all relevant context and evidence gathered"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "self_consistency",
            "description": "Generate multiple independent solutions and return the majority answer. Uses 3 parallel attempts with different reasoning paths. Best for math problems where you want high confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "The problem to solve independently multiple times"},
                },
                "required": ["instruction"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "verify_answer",
            "description": "Verify a candidate answer by substituting it back into the original problem constraints. For math: checks the answer satisfies equations. For code: runs final tests. For QA: cross-checks against passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The candidate answer to verify"},
                    "method": {"type": "string", "description": "Verification method: 'substitute' (math), 'test' (code), or 'crosscheck' (QA)"},
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_answer",
            "description": "Quick sanity check on an answer's format and plausibility. Checks if the answer format matches the expected type (number, label, code, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The answer to check"},
                },
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cross_validate",
            "description": "Solve the problem using a different method and compare with the candidate answer. Useful for catching errors by approaching from a different angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The candidate answer to cross-validate"},
                },
                "required": ["answer"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the code workspace with their sizes. Use this FIRST to discover available files before searching or editing.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for patterns in the code workspace. Supports regex and multi-keyword OR matching. Returns file:line matches. IMPORTANT: Use list_files first to see available files. If [NO_MATCH], try simpler keywords or use view_file to read the file directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern for source code (supports regex). Use concise symbols/error strings; repeated exact queries return cached evidence."},
                    "file_pattern": {"type": "string", "description": "Optional file/path filter (basename, directory, or glob; e.g. 'models.py', 'django/db', '*.py')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View contents of a source file. Returns line-numbered source; use concrete paths from search/list results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Concrete file path to view"},
                    "start_line": {"type": "integer", "description": "Start line number (1-based, default 1)"},
                    "end_line": {"type": "integer", "description": "End line number (inclusive, default end of file)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Describe a source code change in natural language; M_exec generates and applies the exact edit. Cite function/class and a unique local code anchor/condition so the edit location is unambiguous, and phrase insertions where referenced local variables are already defined. Preserve local API style (keyword args, property assignment vs method call) and avoid unnecessary signature/caller plumbing. Do not provide replacement source or diff text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "instruction": {"type": "string", "description": "Natural-language semantic change request grounded in viewed source; include function/class and local code anchor/behavior, preserve keyword-argument/property API style, not a raw patch."},
                },
                "required": ["path", "instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run test commands in the code workspace. Returns stdout/stderr output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_cmd": {"type": "string", "description": "The test command to execute (e.g. 'pytest tests/')"},
                },
                "required": ["test_cmd"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the repository directory. Use for: searching code (grep -rn), navigating files (find, ls, cat), running tests (python -m pytest), reproducing bugs (python script.py). Output is truncated to 10000 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "Custom editing tool for viewing, creating and editing files.\n"
                "* If `path` is a file, `view` displays the file with line numbers. If `path` is a directory, `view` lists files up to 2 levels deep.\n"
                "* The `create` command cannot be used if the file already exists.\n"
                "* If output is long, it will be truncated with `<response clipped>`.\n"
                "* The `undo_edit` command reverts the last edit to the file at `path`.\n\n"
                "Notes for `str_replace` command:\n"
                "* `old_str` must match EXACTLY one or more consecutive lines from the file. Be mindful of whitespace!\n"
                "* If `old_str` is not unique, the replacement will not be performed. Include enough context to make it unique.\n"
                "* `new_str` contains the replacement lines.\n"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run: view, create, str_replace, insert, undo_edit.",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for `create`: content of the new file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for `str_replace`: exact string in `path` to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "For `str_replace`: replacement string. For `insert`: string to insert.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for `insert`: line number after which to insert `new_str`.",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional for `view` on files: [start_line, end_line]. Use [start, -1] for start to end.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "verify_fix",
            "description": "After editing, describe what to verify; M_exec writes a test script (exit 0 = pass). [PASS] terminates the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "NL description of what to check"},
                },
                "required": ["description"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "act",
            "description": "Execute an action in ALFWorld household environment. ALWAYS choose actions from the 'Admissible actions' list shown in the observation. Common patterns: 'go to [location]' to navigate, 'take [object] from [location]' to pick up, 'move [object] to [location]' to place, 'open/close [object]' for containers. If [INVALID], the action had no effect — pick a different one from the admissible list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "The action to execute in the environment"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product",
            "description": "Search for products in the WebShop environment. Returns product IDs (like B078GWRC1J) and names. After search, use click with a product ID to select it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click on an element in the WebShop page. Workflow: search_product → click[product_id] → click[option] → click[Buy Now]. IMPORTANT: Only click elements visible on the current page. If [FAILED], the element is not on this page — use search_product to find the right product first. After selecting options, click 'Buy Now' to complete purchase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "The element to click on"},
                },
                "required": ["element"],
            },
        },
    },


    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Submit your final answer to end the episode. Call this ONLY after you have gathered enough evidence via other tools. The 'response' argument contains your concise final answer (e.g., a number for math, a name for QA, A/B/C/D/E for multiple choice).",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The final answer (brief). Numbers: just the value. QA: the name/fact. Multi-choice: single letter.",
                    },
                },
                "required": ["response"],
            },
        },
    },
]


_QWEN_FUNC_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>", re.DOTALL
)
_QWEN_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)


def parse_qwen_tool_call(content: str) -> Optional[Tuple[str, Dict]]:

    if not content:
        return None


    func_match = _QWEN_FUNC_RE.search(content)
    if func_match:
        func_name = func_match.group(1)
        body = func_match.group(2)
        params = {}
        for pm in _QWEN_PARAM_RE.finditer(body):
            params[pm.group(1)] = pm.group(2).strip()
        return func_name, params


    json_match = re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL
    )
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            func_name = str(data.get("name", "")).strip()
            args = data.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            if func_name:
                return func_name, args
        except (json.JSONDecodeError, Exception):
            pass


    bare_json = re.search(r'\{\s*"name"\s*:', content)
    if bare_json:

        brace_start = bare_json.start()

        for pattern in [r"\{[^{}]*\{[^{}]*\}[^{}]*\}", r"\{[^{}]*\}"]:
            m = re.search(pattern, content[brace_start:], re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                    func_name = str(data.get("name", "")).strip()
                    args = data.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    if func_name:
                        return func_name, args
                except (json.JSONDecodeError, Exception):
                    continue


    trunc_match = re.search(r"<function=(\w+)>", content)
    if trunc_match:
        func_name = trunc_match.group(1)

        params = {}
        for pm in _QWEN_PARAM_RE.finditer(content[trunc_match.start():]):
            params[pm.group(1)] = pm.group(2).strip()
        if not params:

            after = content[trunc_match.end():].strip()

            after = re.sub(r"<[^>]+>", "", after).strip()
            if after:
                params["instruction"] = after[:500]
        return func_name, params

    return None


def supervisor_call(
    messages: List[Dict],
    api_base: str,
    model: str,
    tools: Optional[List[Dict]] = None,
    api_key: str = "",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    enable_thinking: bool = False,
) -> Tuple[str, Optional[str], Optional[Dict]]:


    _wait_if_paused()

    client = _get_client(api_base, api_key)
    effective_tools = tools if tools is not None else SUPERVISOR_TOOLS

    resolved_model = _resolve_model(model)

    extra = {
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if enable_thinking:
        extra["chat_template_kwargs"]["thinking_budget"] = 512

    resp = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        tools=effective_tools,
        max_tokens=max_tokens + (512 if enable_thinking else 0),
        temperature=temperature,
        extra_body=extra,
    )

    msg = resp.choices[0].message
    content = (getattr(msg, "content", "") or "").strip()
    reasoning_content = (getattr(msg, "reasoning_content", "") or "").strip()
    finish_reason = getattr(resp.choices[0], "finish_reason", "stop")
    native_tool_calls = getattr(msg, "tool_calls", None)


    if enable_thinking and not content and not native_tool_calls and reasoning_content:


        ans_in_think = re.search(r"(?:answer|result|m\s*\+\s*n)\s*(?:is|=|:)\s*[\\$]*(\d+)", reasoning_content, re.IGNORECASE)
        if ans_in_think:
            content = ans_in_think.group(1).strip()
        else:

            last_lines = [l.strip() for l in reasoning_content.split('\n') if l.strip()]
            if last_lines:
                content = last_lines[-1][:100]


    if native_tool_calls:
        tc = native_tool_calls[0]
        tool_name = tc.function.name
        try:
            tool_args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, Exception):
            tool_args = {"instruction": tc.function.arguments}
        logger.debug(f"[Supervisor] Native tool call: {tool_name}({list(tool_args.keys())})")
        return content, tool_name, tool_args


    parsed = parse_qwen_tool_call(content)
    if parsed is not None:
        tool_name, tool_args = parsed
        logger.debug(f"[Supervisor] Parsed tool call: {tool_name}({list(tool_args.keys())})")
        return content, tool_name, tool_args


    if finish_reason == "length":
        logger.debug(f"[Supervisor] Truncated (length), retrying with tool-call prompt")
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Your response was too long and got truncated. "
             "Call a tool immediately with a short instruction. "
             "If you have the final answer, call the 'answer' tool with response=<your answer>."},
        ]
        try:
            resp2 = client.chat.completions.create(
                model=resolved_model, messages=retry_messages, tools=effective_tools,
                max_tokens=512, temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            msg2 = resp2.choices[0].message
            content2 = (getattr(msg2, "content", "") or "").strip()


            native_tc2 = getattr(msg2, "tool_calls", None)
            if native_tc2:
                tc = native_tc2[0]
                try:
                    args2 = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, Exception):
                    args2 = {"instruction": tc.function.arguments}
                return content2, tc.function.name, args2


            parsed2 = parse_qwen_tool_call(content2)
            if parsed2 is not None:
                return content2, parsed2[0], parsed2[1]
        except Exception:
            pass


        logger.warning(f"[Supervisor] Truncated retry also failed")


    logger.debug(f"[Supervisor] Direct answer: {content[:100]}...")
    return content, None, None


def vllm_generate_batch(
    messages_list: List[List[Dict]],
    api_base: str,
    model: str,
    tools: Optional[List[Dict]] = None,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    api_key: str = "",
) -> List[Tuple[str, Optional[str], Optional[Dict]]]:

    results = []
    for messages in messages_list:
        try:
            out = supervisor_call(
                messages=messages,
                api_base=api_base,
                model=model,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )
            results.append(out)
        except Exception as e:
            logger.warning(f"vLLM batch call failed: {e}")
            results.append(("", None, None))
    return results


def react_call(
    prompt: str,
    api_base: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.8,
    max_tokens: int = 256,
) -> Tuple[str, Optional[str]]:


    _wait_if_paused()

    client = _get_client(api_base, api_key)

    resolved_model = _resolve_model(model)

    resp = client.chat.completions.create(
        model=resolved_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    msg = resp.choices[0].message
    content = (getattr(msg, "content", "") or "").strip()


    content_lower = content.lower()
    start_tag = "<action>"
    end_tag = "</action>"
    start_idx = content_lower.find(start_tag)
    end_idx = content_lower.find(end_tag)

    if start_idx != -1 and end_idx != -1:
        action_str = content[start_idx + len(start_tag):end_idx].strip()
        if action_str:
            return content, action_str


    import re as _re

    m = _re.search(r'(search|click|skill_invoke)\[([^\]]+)\]', content)
    if m:
        action_str = f"{m.group(1)}[{m.group(2)}]"
        logger.info(f"[ReAct] Extracted raw action (no tags): {action_str[:80]}")
        return content, action_str

    import re as _re2
    for line in content.split("\n"):
        line = line.strip().strip('"').strip("'").strip()
        if line.startswith(">"):
            line = line[1:].strip()
        if line.lower().startswith("action:"):
            line = line[7:].strip()

        if line and _re2.match(r'(?:go to|take|put|move|open|close|use|clean|heat|cool|examine|toggle|turn|inventory|look|pick up|throw)\b', line, _re2.IGNORECASE):
            logger.info(f"[ReAct] Extracted ALFWorld action: {line[:80]}")
            return content, line

    logger.warning(f"[ReAct] No action found in ({len(content)}c): {content[:150]}")
    return content, None
