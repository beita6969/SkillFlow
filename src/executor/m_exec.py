

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

from openai import OpenAI, APIError

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a capable AI assistant. "
    "Execute the given instruction accurately and completely. "
    "Be concise but thorough. Do not add unnecessary caveats or disclaimers."
)


class MExec:


    def __init__(
        self,
        api_base: str,
        model_name: str = "gpt-oss-120b",
        api_key: str = "",
        default_temperature: float = 0.1,
        default_max_tokens: int = 8192,  
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self._api_base = api_base
        self._api_key = api_key or os.environ.get("MEXEC_API_KEY") or os.environ.get("SGLANG_API_KEY", "EMPTY")
        self._thread_local = threading.local()  
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay


        self._total_calls = 0
        self._total_errors = 0


        MExec._instance = self

    @property
    def client(self) -> OpenAI:

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

        results = []
        for i, instr in enumerate(instructions):
            ctx = contexts[i] if contexts else ""
            results.append(self.execute(instr, context=ctx, **kwargs))
        return results

    def test_connectivity(self) -> bool:

        try:
            result = self.execute("What is 2+2? Answer with just the number.", max_tokens=512)
            return "4" in result
        except Exception:
            return False

    @property
    def stats(self) -> dict:

        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(1, self._total_calls),
        }


    def _build_messages(self, instruction: str, context: str) -> list[dict]:

        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if context:

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

        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
