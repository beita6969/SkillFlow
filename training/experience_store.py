

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experience:

    condition: str          
    action: str             
    task_types: List[str]   
    reward_signal: float    
    usage_count: int = 0
    source_step: int = 0    
    exp_id: str = ""


class ExperienceStore:


    def __init__(
        self,
        max_entries: int = 120,
        dedup_threshold: float = 0.70,
        save_path: Optional[Path] = None,
    ):
        self.entries: List[Experience] = []
        self.embeddings: Optional[np.ndarray] = None  
        self.max_entries = max_entries
        self.dedup_threshold = dedup_threshold
        self.save_path = save_path
        self._embed_model = None
        self._next_id = 0

    @property
    def size(self) -> int:
        return len(self.entries)


    def retrieve(
        self,
        query: str,
        task_type: str = "",
        top_k: int = 3,
    ) -> List[Experience]:

        if not self.entries:
            return []

        model = self._get_embed_model()
        if model is None:

            matched = [e for e in self.entries if not task_type or task_type in e.task_types]
            matched.sort(key=lambda e: e.reward_signal, reverse=True)
            return matched[:top_k]

        q_emb = model.encode([query], normalize_embeddings=True)
        if self.embeddings is None:
            self._rebuild_embeddings()
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        sims = (q_emb @ self.embeddings.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([float(sims)])


        scored = []
        for i, exp in enumerate(self.entries):
            type_bonus = 0.15 if task_type and task_type in exp.task_types else 0.0
            score = float(sims[i]) + type_bonus
            scored.append((score, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for score, exp in scored[:top_k] if score > 0.1]


    def add(self, exp: Experience) -> bool:


        if self._is_duplicate(exp):
            logger.debug(f"[ExpStore] Duplicate, merging: {exp.condition[:40]}")
            return False

        exp.exp_id = f"exp_{self._next_id:03d}"
        self._next_id += 1
        self.entries.append(exp)
        self.embeddings = None  


        if len(self.entries) > self.max_entries:
            self._evict_lowest()

        logger.info(
            f"[ExpStore] Added {exp.exp_id}: {exp.condition[:40]}... "
            f"(r={exp.reward_signal:.2f}, total={self.size})"
        )
        return True

    def add_batch(self, experiences: List[Experience]) -> int:

        added = sum(1 for exp in experiences if self.add(exp))
        return added


    def _is_duplicate(self, new_exp: Experience) -> bool:

        if not self.entries:
            return False

        model = self._get_embed_model()
        if model is None:
            for exp in self.entries:
                if exp.condition == new_exp.condition:
                    exp.reward_signal = 0.7 * exp.reward_signal + 0.3 * new_exp.reward_signal
                    exp.usage_count += 1
                    return True
            return False

        new_text = f"{new_exp.condition} {new_exp.action}"
        new_emb = model.encode([new_text], normalize_embeddings=True)

        if self.embeddings is None:
            self._rebuild_embeddings()
        if self.embeddings is None or len(self.embeddings) == 0:
            return False

        sims = (new_emb @ self.embeddings.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([float(sims)])

        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        existing = self.entries[max_idx]


        if max_sim < 0.50:
            return False


        if max_sim > 0.92:
            existing.reward_signal = 0.7 * existing.reward_signal + 0.3 * new_exp.reward_signal
            existing.usage_count += 1
            logger.debug(f"[ExpStore] Exact duplicate (sim={max_sim:.2f}): {new_exp.condition[:40]}")
            return True


        is_dup = self._llm_judge_duplicate(new_exp, existing, max_sim)
        if is_dup:
            existing.reward_signal = 0.7 * existing.reward_signal + 0.3 * new_exp.reward_signal
            existing.usage_count += 1
        return is_dup

    def _llm_judge_duplicate(self, new_exp: Experience, existing: Experience, sim: float) -> bool:

        try:
            from src.executor.m_exec import MExec
            m_exec = MExec._instance if hasattr(MExec, '_instance') and MExec._instance else None
            if m_exec is None:

                return sim > self.dedup_threshold

            prompt = f"""Are these two action rules saying the SAME strategy, or are they DIFFERENT strategies?

Rule A:
  Condition: {existing.condition}
  Action: {existing.action}

Rule B:
  Condition: {new_exp.condition}
  Action: {new_exp.action}

Answer ONLY "SAME" or "DIFFERENT". They are SAME only if both the trigger condition AND the recommended action are essentially identical. If they apply to similar situations but recommend different actions or tools, answer DIFFERENT."""

            response = m_exec.execute(prompt, max_tokens=10, temperature=0.0)
            is_same = "SAME" in response.upper() and "DIFFERENT" not in response.upper()
            logger.info(
                f"[ExpStore] LLM dedup (sim={sim:.2f}): "
                f"{'SAME → reject' if is_same else 'DIFFERENT → keep'} | "
                f"A='{existing.condition[:30]}' vs B='{new_exp.condition[:30]}'"
            )
            return is_same
        except Exception as e:
            logger.debug(f"[ExpStore] LLM dedup failed ({e}), fallback to threshold")
            return sim > self.dedup_threshold

    def _evict_lowest(self):

        if len(self.entries) <= self.max_entries:
            return

        self.entries.sort(
            key=lambda e: e.reward_signal * math.log(e.usage_count + 2),
            reverse=True,
        )
        removed = self.entries[self.max_entries:]
        self.entries = self.entries[:self.max_entries]
        self.embeddings = None
        logger.info(f"[ExpStore] Evicted {len(removed)} low-quality experiences")


    def compute_experience_reward(
        self,
        turns: list,
        task_type: str,
        max_bonus: float = 0.1,
    ) -> float:

        if not self.entries:
            return 0.0


        relevant = [e for e in self.entries if task_type in e.task_types]
        if not relevant:
            return 0.0

        bonus = 0.0
        for turn in turns:
            atype = getattr(turn, "action_type", "")
            for exp in relevant:

                if atype and atype in exp.action.lower():
                    bonus += 0.02 * min(exp.reward_signal, 1.5)

        return min(bonus, max_bonus)


    def format_for_prompt(self, task_type: str = "", query: str = "", top_k: int = 3) -> str:

        exps = self.retrieve(query, task_type=task_type, top_k=top_k)
        if not exps:
            return ""

        bullets = "\n".join(
            f"• [{exp.exp_id}] {exp.condition} → {exp.action}"
            for exp in exps
        )
        return (
            f"Here are practical tips gathered from similar problems:\n\n"
            f"{bullets}\n\n"
            f"These highlight common patterns. Consider applying them when matching situations arise."
        )


    def save(self, path: Optional[Path] = None):

        p = path or self.save_path
        if p is None:
            return
        data = [asdict(e) for e in self.entries]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.debug(f"[ExpStore] Saved {len(data)} experiences to {p}")

    def load(self, path: Optional[Path] = None):

        p = path or self.save_path
        if p is None or not p.exists():
            return
        data = json.loads(p.read_text())
        self.entries = [Experience(**d) for d in data]
        self._next_id = max((int(e.exp_id.split("_")[-1]) for e in self.entries if e.exp_id), default=-1) + 1
        self.embeddings = None
        logger.info(f"[ExpStore] Loaded {len(self.entries)} experiences from {p}")


    def _get_embed_model(self):
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
            except Exception:
                self._embed_model = "FAILED"
        return self._embed_model if self._embed_model != "FAILED" else None

    def _rebuild_embeddings(self):
        model = self._get_embed_model()
        if model is None or not self.entries:
            self.embeddings = None
            return
        texts = [f"{e.condition} {e.action}" for e in self.entries]
        self.embeddings = model.encode(texts, normalize_embeddings=True)
