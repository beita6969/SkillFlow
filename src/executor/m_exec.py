"""
M_exec — 通用 LLM 执行器（完全冻结，不参与训练）。

SkillFlow 中 M_exec 是唯一的执行后端：
- 接收 Supervisor 的自然语言 instruction
- 无工具调用，无函数签名约束
- 返回文本结果（observation）

支持 gpt-oss-120b（本地 vLLM）或任意 OpenAI 兼容 API。
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

from openai import OpenAI, APIError

logger = logging.getLogger(__name__)

# M_exec 的系统 prompt：简洁、强调准确执行
_SYSTEM_PROMPT = (
    "You are a capable AI assistant. "
    "Execute the given instruction accurately and completely. "
    "Be concise but thorough. Do not add unnecessary caveats or disclaimers."
)


class MExec:
    """
    通用 LLM 执行后端。

    Usage:
        exec = MExec(api_base="http://localhost:8004/v1", model_name="gpt-oss-120b")
        result = exec.execute("What is the capital of France?")
        # → "The capital of France is Paris."
    """

    def __init__(
        self,
        api_base: str,
        model_name: str = "gpt-oss-120b",
        api_key: str = "",
        default_temperature: float = 0.1,
        default_max_tokens: int = 8192,  # v5.3: 4096→8192, 复杂 math/code 需要更多 token
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self._api_base = api_base
        self._api_key = api_key or os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY")
        self._thread_local = threading.local()  # Thread-local client storage
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 统计
        self._total_calls = 0
        self._total_errors = 0

        # Singleton 引用（供 experience_store 的 LLM 去重使用）
        MExec._instance = self

    @property
    def client(self) -> OpenAI:
        """Thread-local OpenAI client（每个线程独立连接，避免连接池竞争）"""
        c = getattr(self._thread_local, "client", None)
        if c is None:
            c = OpenAI(base_url=self._api_base, api_key=self._api_key)
            self._thread_local.client = c
        return c

    def execute(
        self,
        instruction: str,
        context: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "",
    ) -> str:
        """
        执行任意自然语言 instruction，返回文本结果。

        Args:
            instruction: Supervisor 发出的指令（由各工具 handler 构造）
            context:     可选辅助上下文（任务背景、文档段落等）
            temperature: 覆盖默认温度
            max_tokens:  覆盖默认最大 token
            task_type:   任务类型标签（用于日志）

        Returns:
            M_exec 的输出字符串。失败时返回 "[EXECUTION_ERROR] ..." 格式字符串。
        """
        messages = self._build_messages(instruction, context)
        self._total_calls += 1

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.default_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                msg = resp.choices[0].message
                result = msg.content or ""
                result = self._strip_thinking(result)

                # Fallback: 如果 content 为空但有 reasoning_content（thinking 模式耗尽 token）
                if not result.strip():
                    reasoning = getattr(msg, "reasoning_content", None) or ""
                    if reasoning:
                        logger.warning(
                            f"[MExec] content empty, using reasoning_content "
                            f"({len(reasoning)} chars) as fallback"
                        )
                        result = reasoning

                logger.debug(
                    f"[MExec] {task_type or 'unknown'} | "
                    f"instr={instruction[:60]}... | "
                    f"result={result[:80]}..."
                )
                return result

            except APIError as e:
                self._total_errors += 1
                logger.warning(f"[MExec] API error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                self._total_errors += 1
                logger.error(f"[MExec] Unexpected error: {e}")
                return f"[EXECUTION_ERROR] {type(e).__name__}: {e}"

        return "[EXECUTION_ERROR] Max retries exceeded"

    def execute_batch(
        self,
        instructions: list[str],
        contexts: Optional[list[str]] = None,
        **kwargs,
    ) -> list[str]:
        """批量执行（顺序执行，不并行）"""
        results = []
        for i, instr in enumerate(instructions):
            ctx = contexts[i] if contexts else ""
            results.append(self.execute(instr, context=ctx, **kwargs))
        return results

    def test_connectivity(self) -> bool:
        """测试 M_exec API 连通性"""
        try:
            result = self.execute("What is 2+2? Answer with just the number.", max_tokens=512)
            return "4" in result
        except Exception:
            return False

    @property
    def stats(self) -> dict:
        """返回调用统计"""
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_calls),
        }

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _build_messages(self, instruction: str, context: str) -> list[dict]:
        """构建 OpenAI messages 列表"""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if context:
            # 先给上下文，模拟文档检索场景
            messages.append({
                "role": "user",
                "content": f"Here is some relevant context:\n\n{context}"
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have read the context and will use it to answer."
            })

        messages.append({"role": "user", "content": instruction})
        return messages

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """去除模型输出中的 <think> 块（Qwen3 思维链）"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
