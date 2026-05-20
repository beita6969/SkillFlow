

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from src.skills.format import (
    SkillEntry, TaskTypeSkillDocument, GENERAL_SKILL,
    load_skills_from_dir, assign_skill_id,
)

logger = logging.getLogger(__name__)


_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}
_EMBEDDING_MODEL_LOCK = threading.Lock()

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def _get_embedding_model(model_name: str = _DEFAULT_EMBEDDING_MODEL):

    with _EMBEDDING_MODEL_LOCK:
        if model_name not in _EMBEDDING_MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {model_name} ...")
                model = SentenceTransformer(model_name, device="cpu")
                _EMBEDDING_MODEL_CACHE[model_name] = model
                logger.info(f"Embedding model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model {model_name}: {e}")
                _EMBEDDING_MODEL_CACHE[model_name] = None
        return _EMBEDDING_MODEL_CACHE[model_name]


class SkillWorkspace:


    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        max_skills: int = 60,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        embedding_weight: float = 0.6,
        flow_weight: float = 0.4,
    ):
        self._skills: Dict[str, SkillEntry] = {}
        self._lock = threading.RLock()
        self.skills_dir = skills_dir
        self.max_skills = max_skills


        self._embedding_model_name = embedding_model
        self._embedding_weight = embedding_weight  
        self._flow_weight = flow_weight              

        self._skill_embeddings_cache: Optional[Dict] = None


        self._type_documents: Dict[str, TaskTypeSkillDocument] = {}

        if skills_dir:
            skills_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()


    def add(self, skill: SkillEntry) -> bool:

        with self._lock:
            if len(self._skills) >= self.max_skills and skill.meta.skill_id not in self._skills:
                logger.warning(
                    f"Workspace full ({len(self._skills)}/{self.max_skills}), "
                    f"cannot add {skill.meta.skill_id}"
                )
                return False

            self._skills[skill.meta.skill_id] = skill
            self._skill_embeddings_cache = None  
            if self.skills_dir:
                self._save_skill_to_disk(skill)

            logger.debug(f"Added skill {skill.meta.skill_id}: {skill.name}")
            return True

    def add_batch(self, skills: List[SkillEntry]) -> int:

        seen_ids: dict[str, int] = {}  
        added = 0
        for s in skills:
            sid = s.meta.skill_id
            if sid in seen_ids:
                seen_ids[sid] += 1
                logger.warning(
                    f"Batch ID collision: '{sid}' appeared {seen_ids[sid]} times "
                    f"(skill '{s.name}' overwrites previous). "
                    f"This indicates assign_skill_id() was not called or failed."
                )
            else:
                seen_ids[sid] = 1
            if self.add(s):
                added += 1

        unique_count = len(seen_ids)
        if unique_count < len(skills):
            logger.warning(
                f"add_batch: {len(skills)} skills submitted but only "
                f"{unique_count} unique IDs — {len(skills) - unique_count} overwrites occurred"
            )
        return unique_count

    def remove(self, skill_id: str) -> bool:

        with self._lock:
            if skill_id not in self._skills:
                return False
            del self._skills[skill_id]
            self._skill_embeddings_cache = None  
            if self.skills_dir:
                path = self.skills_dir / f"{skill_id}.md"
                if path.exists():
                    path.unlink()
            logger.info(f"Removed skill {skill_id}")
            return True

    def update_flow_score(self, skill_id: str, new_score: float, alpha: float = 0.3) -> None:

        with self._lock:
            if skill_id in self._skills:
                old = self._skills[skill_id].meta.flow_score
                updated = alpha * new_score + (1 - alpha) * old
                self._skills[skill_id].meta.flow_score = updated

    def update_reward_variance(self, skill_id: str, new_variance: float, alpha: float = 0.3) -> None:

        with self._lock:
            if skill_id in self._skills:
                old = self._skills[skill_id].meta.reward_variance
                updated = alpha * new_variance + (1 - alpha) * old
                self._skills[skill_id].meta.reward_variance = updated

    def record_usage(self, skill_id: str, step: int, success: bool = False) -> None:

        with self._lock:
            if skill_id in self._skills:
                self._skills[skill_id].meta.usage_count += 1
                self._skills[skill_id].meta.last_used_step = step
                if success:
                    self._skills[skill_id].meta.success_count += 1


    def get_type_document(self, task_type: str) -> Optional[TaskTypeSkillDocument]:

        return self._type_documents.get(task_type)

    def update_type_document(self, doc: TaskTypeSkillDocument) -> None:

        with self._lock:
            self._type_documents[doc.task_type] = doc
            logger.debug(f"Updated type document for {doc.task_type}: {len(doc.variant_skill_ids)} skills")

    def get_skills_by_task_type(self, task_type: str) -> List[SkillEntry]:

        with self._lock:
            return [
                s for s in self._skills.values()
                if task_type in (s.meta.task_types or [])
            ]


    def retrieve(
        self,
        query: str,
        task_type: str = "",
        top_k: int = 4,
    ) -> List[SkillEntry]:

        with self._lock:
            if not self._skills:
                return []


            enhanced_query = self._enhance_query(query, task_type)


            result = self._embedding_retrieve(enhanced_query, top_k, task_type=task_type)
            if result is not None:
                return result


            logger.debug("Embedding unavailable, falling back to keyword matching")
            skills = list(self._skills.values())
            scored = [
                (skill.meta.flow_score, self._keyword_similarity(query, skill, task_type), skill)
                for skill in skills
            ]
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [s for _, _, s in scored[:top_k]]

    def get_by_id(self, skill_id: str) -> Optional[SkillEntry]:

        with self._lock:
            if skill_id == "general":
                return GENERAL_SKILL
            return self._skills.get(skill_id)

    def get_all(self) -> List[SkillEntry]:

        with self._lock:
            skills = list(self._skills.values())
            skills.sort(key=lambda s: s.meta.flow_score, reverse=True)
            return skills

    def get_by_task_type(self, task_type: str) -> List[SkillEntry]:

        with self._lock:
            return [s for s in self._skills.values() if task_type in (s.meta.task_types or [])]

    def get_all_ids(self) -> List[str]:

        with self._lock:
            return list(self._skills.keys())

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._skills)


    def format_skills_for_prompt(
        self,
        query: str = "",
        task_type: str = "",
        top_k: int = 4,
    ) -> str:

        if not self._skills:
            return ""

        if query:
            skills = self.retrieve(query, task_type=task_type, top_k=top_k)  
        else:
            skills = self.get_all()[:top_k]

        if not skills:
            return ""

        lines = ["## Learned Strategies (follow these patterns directly with your tools; no invoke needed)"]
        for skill in skills:
            lines.append(skill.format_for_prompt(include_pitfall=True))
            lines.append("---")

        return "\n".join(lines)


    def save_all(self) -> None:

        if not self.skills_dir:
            return
        with self._lock:
            for skill in self._skills.values():
                self._save_skill_to_disk(skill)

    def _load_from_disk(self) -> None:

        if not self.skills_dir:
            return
        skills = load_skills_from_dir(self.skills_dir)
        for skill in skills:
            self._skills[skill.meta.skill_id] = skill
        logger.info(f"Loaded {len(skills)} skills from {self.skills_dir}")

    def _save_skill_to_disk(self, skill: SkillEntry) -> None:

        if not self.skills_dir:
            return
        path = self.skills_dir / f"{skill.meta.skill_id}.md"
        try:
            skill.save_to_file(path)
        except Exception as e:
            logger.warning(f"Failed to save skill {skill.meta.skill_id}: {e}")


    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    @staticmethod
    def _enhance_query(query: str, task_type: str = "") -> str:

        parts = []
        if task_type:
            parts.append(f"Task type: {task_type}.")
        parts.append(f"Question: {query}")
        context_query = " ".join(parts)
        return SkillWorkspace._BGE_QUERY_PREFIX + context_query

    @staticmethod
    def _skill_to_text(skill: SkillEntry) -> str:

        parts = [skill.name]
        if skill.trigger:
            parts.append(skill.trigger)
        if skill.description:
            parts.append(skill.description)
        if skill.plan:
            plan_text = skill.plan if len(skill.plan) <= 200 else skill.plan[:200]
            parts.append(plan_text)
        return " ".join(parts)

    @staticmethod
    def _skill_to_multi_text(skill: SkillEntry) -> List[str]:

        texts = [

            (skill.name + " " + (skill.description or "")).strip(),

            skill.trigger or skill.name,

            (skill.plan or "")[:200] if skill.plan else skill.name,
        ]
        return texts

    def _compute_skill_embeddings(self) -> Optional[Dict]:

        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        model = _get_embedding_model(self._embedding_model_name)
        if model is None:
            return None

        if not self._skills:
            return None

        skill_ids = list(self._skills.keys())

        try:

            multi_embeddings = []
            for sid in skill_ids:
                texts = self._skill_to_multi_text(self._skills[sid])
                vecs = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64,
                )
                multi_embeddings.append(vecs)  

            cache = {
                "skill_ids": skill_ids,
                "embeddings": np.array(multi_embeddings, dtype=np.float32),  
            }
            self._skill_embeddings_cache = cache
            logger.debug(
                f"Computed multi-vector embeddings for {len(skill_ids)} skills "
                f"(shape: {cache['embeddings'].shape})"
            )
            return cache
        except Exception as e:
            logger.warning(f"Failed to compute skill embeddings: {e}")
            return None

    def _embedding_retrieve(self, query: str, top_k: int, task_type: str = "") -> Optional[List[SkillEntry]]:

        cache = self._compute_skill_embeddings()
        if cache is None:
            return None

        model = _get_embedding_model(self._embedding_model_name)
        if model is None:
            return None

        try:

            query_emb = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_emb = np.array(query_emb, dtype=np.float32)  


            all_sims = np.einsum('ijk,mk->ij', cache["embeddings"], query_emb)  
            max_sims = np.max(all_sims, axis=1)  

            skill_ids = cache["skill_ids"]


            same_type_scored = []
            cross_type_scored = []

            for i, sid in enumerate(skill_ids):
                skill = self._skills.get(sid)
                if skill is None:
                    continue

                cos_sim = float(max_sims[i])
                norm_flow = self._normalize_flow(skill.meta.flow_score)
                usage = skill.meta.usage_count
                success_rate = skill.meta.success_count / max(usage, 1)

                is_same_type = task_type and task_type in (skill.meta.task_types or [])

                if is_same_type:


                    usage_conf = min(usage / 10.0, 1.0)
                    effective_sr = usage_conf * success_rate + (1.0 - usage_conf) * 0.3
                    combined = 0.7 * cos_sim + 0.2 * norm_flow + 0.1 * effective_sr
                    same_type_scored.append((combined, cos_sim, skill))
                else:

                    combined = 0.4 * cos_sim + 0.3 * norm_flow + 0.3 * success_rate
                    cross_type_scored.append((combined, cos_sim, skill))


            same_type_scored.sort(key=lambda x: x[0], reverse=True)
            cross_type_scored.sort(key=lambda x: x[0], reverse=True)


            result = [s for _, _, s in same_type_scored[:top_k]]
            remaining = top_k - len(result)
            if remaining > 0:
                result.extend([s for _, _, s in cross_type_scored[:remaining]])

            if result:
                top1 = result[0]
                top1_idx = skill_ids.index(top1.meta.skill_id)
                dim_sims = all_sims[top1_idx]
                best_dim = int(np.argmax(dim_sims))
                dim_names = ["功能", "触发条件", "计划"]
                is_same = task_type and task_type in (top1.meta.task_types or [])
                logger.debug(
                    f"Two-stage retrieve: query='{query[:50]}...', "
                    f"same_type={len(same_type_scored)}, cross_type={len(cross_type_scored)}, "
                    f"top1=[{top1.meta.skill_id}] same_type={is_same} "
                    f"best_dim={dim_names[best_dim]}"
                )

            return result

        except Exception as e:
            logger.warning(f"Embedding retrieve failed: {e}")
            return None

    _STOP_WORDS_SET = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other",
        "some", "such", "than", "too", "very", "just", "about", "up",
        "out", "if", "then", "that", "this", "these", "those", "it", "its",
        "what", "which", "who", "whom", "how", "when", "where", "why",
    })

    @staticmethod
    def _keyword_similarity(query: str, skill: SkillEntry, task_type: str) -> float:


        q_words = set(query.lower().split()) - SkillWorkspace._STOP_WORDS_SET
        if not q_words:
            return 0.3  


        skill_text = f"{skill.trigger} {skill.description} {skill.name}".lower()
        s_words = set(skill_text.split()) - SkillWorkspace._STOP_WORDS_SET


        if not s_words:
            return 0.0

        common = q_words & s_words
        base_score = len(common) / (len(q_words | s_words) + 1e-8)


        task_bonus = 0.2 if task_type and task_type.replace("_", " ") in skill_text else 0.0

        return min(1.0, base_score + task_bonus)

    @staticmethod
    def _normalize_flow(flow_score: float) -> float:

        import math

        x = max(min(flow_score * 0.5, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-x))

    def get_next_skill_id(self) -> str:

        with self._lock:
            existing_ids = set(self._skills.keys())
            i = 0
            while f"dyn_{i:03d}" in existing_ids:
                i += 1
            return f"dyn_{i:03d}"

    def prune_low_flow(
        self,
        prune_threshold: float = -10.0,
        min_usage: int = 20,
        current_step: int = -1,
        recency_window: int = 10,
        evolution_phase_steps: int = 10,
    ) -> List[str]:

        with self._lock:
            to_prune = []
            age_pruned = []

            for sid, skill in self._skills.items():

                if skill.meta.usage_count == 0 and current_step > 0:
                    age = current_step - skill.meta.creation_step
                    min_age = 2 * evolution_phase_steps  
                    if age >= min_age:
                        age_pruned.append(sid)
                        logger.info(
                            f"Prune (age) {sid} '{skill.name}': "
                            f"0 usage after {age} steps (min_age={min_age})"
                        )
                        continue


                if not (skill.meta.flow_score < prune_threshold
                        and skill.meta.usage_count >= min_usage):
                    continue


                if current_step > 0 and skill.meta.last_used_step >= current_step - recency_window:
                    logger.info(
                        f"Prune skip {sid}: recently used (last={skill.meta.last_used_step}, "
                        f"current={current_step}, window={recency_window})"
                    )
                    continue


                success_rate = (skill.meta.success_count / max(1, skill.meta.usage_count))
                if success_rate >= 0.3:
                    logger.info(
                        f"Prune skip {sid}: high success rate ({success_rate:.2f}, "
                        f"{skill.meta.success_count}/{skill.meta.usage_count})"
                    )
                    continue

                to_prune.append(sid)

            all_pruned = to_prune + age_pruned
            for sid in all_pruned:
                self.remove(sid)

            if all_pruned:
                logger.info(
                    f"Pruned {len(all_pruned)} skills: "
                    f"{len(to_prune)} low-flow + {len(age_pruned)} age-based = {all_pruned}"
                )

            return all_pruned

    def summary(self) -> Dict:

        with self._lock:
            skills = list(self._skills.values())
            return {
                "total_skills": len(skills),
                "avg_flow_score": sum(s.meta.flow_score for s in skills) / max(1, len(skills)),
                "total_usage": sum(s.meta.usage_count for s in skills),
                "skill_ids": [s.meta.skill_id for s in skills],
                "top_skills": [
                    {
                        "id": s.meta.skill_id,
                        "name": s.name,
                        "flow": s.meta.flow_score,
                        "var": s.meta.reward_variance,
                    }
                    for s in sorted(skills, key=lambda x: x.meta.flow_score, reverse=True)[:5]
                ],
            }
