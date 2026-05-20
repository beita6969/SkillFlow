"""
SkillWorkspace — 技能库管理，支持 embedding + flow 排序检索。

职责：
  1. 存储所有 skills（内存 + 磁盘持久化）
  2. 按 embedding 语义相似度 + F̂(s) flow_score 检索相关 skills
  3. 跟踪 usage 统计（供 lifecycle manager 使用）
  4. 线程安全（训练中可能并发读写）
  5. 支持动态增删技能时自动失效 embedding 缓存
"""

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

# ── Embedding 模型全局缓存（避免重复加载） ──
_EMBEDDING_MODEL_CACHE: Dict[str, Any] = {}
_EMBEDDING_MODEL_LOCK = threading.Lock()

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def _get_embedding_model(model_name: str = _DEFAULT_EMBEDDING_MODEL):
    """懒加载 embedding 模型，全局单例。"""
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
    """
    技能库（S），支持增删查改和 flow-aware 排序。

    内部存储：Dict[skill_id → SkillEntry]
    持久化：skills_dir/*.md 文件
    """

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

        # Embedding 检索配置
        self._embedding_model_name = embedding_model
        self._embedding_weight = embedding_weight  # 语义相似度权重
        self._flow_weight = flow_weight              # flow_score 权重
        # Embedding 缓存：技能变更时置 None
        self._skill_embeddings_cache: Optional[Dict] = None
        # cache format: {"skill_ids": List[str], "embeddings": np.ndarray (N, D)}

        # Per-task-type Skill Documents（XSkill Living Document）
        self._type_documents: Dict[str, TaskTypeSkillDocument] = {}

        if skills_dir:
            skills_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ──────────────────────────────────────────────
    # 增删
    # ──────────────────────────────────────────────

    def add(self, skill: SkillEntry) -> bool:
        """
        添加技能。如果 skill_id 冲突，覆盖旧版本。

        Returns:
            True if added, False if rejected (over capacity)
        """
        with self._lock:
            if len(self._skills) >= self.max_skills and skill.meta.skill_id not in self._skills:
                logger.warning(
                    f"Workspace full ({len(self._skills)}/{self.max_skills}), "
                    f"cannot add {skill.meta.skill_id}"
                )
                return False

            self._skills[skill.meta.skill_id] = skill
            self._skill_embeddings_cache = None  # 失效缓存
            if self.skills_dir:
                self._save_skill_to_disk(skill)

            logger.debug(f"Added skill {skill.meta.skill_id}: {skill.name}")
            return True

    def add_batch(self, skills: List[SkillEntry]) -> int:
        """批量添加，返回实际新增的唯一技能数量。

        如果批内存在 skill_id 冲突（多个 skill 共享同一 ID），
        会记录警告日志。调用者应在调用前确保 ID 唯一（通过 assign_skill_id）。
        """
        seen_ids: dict[str, int] = {}  # skill_id -> count in this batch
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
        """删除技能（含磁盘文件）"""
        with self._lock:
            if skill_id not in self._skills:
                return False
            del self._skills[skill_id]
            self._skill_embeddings_cache = None  # 失效缓存
            if self.skills_dir:
                path = self.skills_dir / f"{skill_id}.md"
                if path.exists():
                    path.unlink()
            logger.info(f"Removed skill {skill_id}")
            return True

    def update_flow_score(self, skill_id: str, new_score: float, alpha: float = 0.3) -> None:
        """EMA 更新 F̂(s)"""
        with self._lock:
            if skill_id in self._skills:
                old = self._skills[skill_id].meta.flow_score
                updated = alpha * new_score + (1 - alpha) * old
                self._skills[skill_id].meta.flow_score = updated

    def update_reward_variance(self, skill_id: str, new_variance: float, alpha: float = 0.3) -> None:
        """EMA 更新 Var_skill(s)（论文 Eq.11）"""
        with self._lock:
            if skill_id in self._skills:
                old = self._skills[skill_id].meta.reward_variance
                updated = alpha * new_variance + (1 - alpha) * old
                self._skills[skill_id].meta.reward_variance = updated

    def record_usage(self, skill_id: str, step: int, success: bool = False) -> None:
        """记录技能被调用"""
        with self._lock:
            if skill_id in self._skills:
                self._skills[skill_id].meta.usage_count += 1
                self._skills[skill_id].meta.last_used_step = step
                if success:
                    self._skills[skill_id].meta.success_count += 1

    # ──────────────────────────────────────────────
    # Per-Task-Type Documents（XSkill Living Document）
    # ──────────────────────────────────────────────

    def get_type_document(self, task_type: str) -> Optional[TaskTypeSkillDocument]:
        """获取某 task_type 的 Living Document"""
        return self._type_documents.get(task_type)

    def update_type_document(self, doc: TaskTypeSkillDocument) -> None:
        """更新 per-task-type Living Document"""
        with self._lock:
            self._type_documents[doc.task_type] = doc
            logger.debug(f"Updated type document for {doc.task_type}: {len(doc.variant_skill_ids)} skills")

    def get_skills_by_task_type(self, task_type: str) -> List[SkillEntry]:
        """获取属于某 task_type 的所有 skills"""
        with self._lock:
            return [
                s for s in self._skills.values()
                if task_type in (s.meta.task_types or [])
            ]

    # ──────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        task_type: str = "",
        top_k: int = 4,
    ) -> List[SkillEntry]:
        """
        检索最相关的 top_k 技能。

        排序公式（参考 SkillRL §3.4）：
            score = α * cosine_sim(query, skill) + β * normalized_flow(skill)

        其中：
        - cosine_sim 由 embedding 模型计算（bge-base-en-v1.5）
        - normalized_flow = sigmoid(flow_score) 归一化到 [0, 1]
        - α = embedding_weight, β = flow_weight

        训练初期 flow_score 全为 0 时，排序完全由语义相似度决定；
        随着训练进行，flow_score 逐渐区分高质量技能。

        Embedding 不可用时回退到关键词匹配。
        """
        with self._lock:
            if not self._skills:
                return []

            # 构造增强 query：加入 task_type 上下文 + BGE instruction prefix
            enhanced_query = self._enhance_query(query, task_type)

            # 尝试两阶段 embedding 检索（task-type primary + F̂(s) ranking）
            result = self._embedding_retrieve(enhanced_query, top_k, task_type=task_type)
            if result is not None:
                return result

            # Fallback: 关键词匹配
            logger.debug("Embedding unavailable, falling back to keyword matching")
            skills = list(self._skills.values())
            scored = [
                (skill.meta.flow_score, self._keyword_similarity(query, skill, task_type), skill)
                for skill in skills
            ]
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return [s for _, _, s in scored[:top_k]]

    def get_by_id(self, skill_id: str) -> Optional[SkillEntry]:
        """按 ID 获取技能（含 general 兜底 skill）"""
        with self._lock:
            if skill_id == "general":
                return GENERAL_SKILL
            return self._skills.get(skill_id)

    def get_all(self) -> List[SkillEntry]:
        """获取所有技能（按 flow_score 降序）"""
        with self._lock:
            skills = list(self._skills.values())
            skills.sort(key=lambda s: s.meta.flow_score, reverse=True)
            return skills

    def get_by_task_type(self, task_type: str) -> List[SkillEntry]:
        """获取指定 task_type 的所有技能"""
        with self._lock:
            return [s for s in self._skills.values() if task_type in (s.meta.task_types or [])]

    def get_all_ids(self) -> List[str]:
        """获取所有 skill_id"""
        with self._lock:
            return list(self._skills.keys())

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._skills)

    # ──────────────────────────────────────────────
    # Prompt 格式化
    # ──────────────────────────────────────────────

    def format_skills_for_prompt(
        self,
        query: str = "",
        task_type: str = "",
        top_k: int = 4,
    ) -> str:
        """生成注入 Supervisor prompt 的技能列表"""
        if not self._skills:
            return ""

        if query:
            skills = self.retrieve(query, task_type=task_type, top_k=top_k)  # F̂(s)-first
        else:
            skills = self.get_all()[:top_k]

        if not skills:
            return ""

        lines = ["## Learned Strategies (follow these patterns directly with your tools; no invoke needed)"]
        for skill in skills:
            lines.append(skill.format_for_prompt(include_pitfall=True))
            lines.append("---")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────

    def save_all(self) -> None:
        """将所有技能保存到磁盘"""
        if not self.skills_dir:
            return
        with self._lock:
            for skill in self._skills.values():
                self._save_skill_to_disk(skill)

    def _load_from_disk(self) -> None:
        """从磁盘加载技能"""
        if not self.skills_dir:
            return
        skills = load_skills_from_dir(self.skills_dir)
        for skill in skills:
            self._skills[skill.meta.skill_id] = skill
        logger.info(f"Loaded {len(skills)} skills from {self.skills_dir}")

    def _save_skill_to_disk(self, skill: SkillEntry) -> None:
        """保存单个技能到磁盘"""
        if not self.skills_dir:
            return
        path = self.skills_dir / f"{skill.meta.skill_id}.md"
        try:
            skill.save_to_file(path)
        except Exception as e:
            logger.warning(f"Failed to save skill {skill.meta.skill_id}: {e}")

    # ──────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────

    # ──────────────────────────────────────────────
    # Embedding 检索
    # ──────────────────────────────────────────────

    # BGE instruction prefix（官方推荐用于非对称检索，只加在 query 端）
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    @staticmethod
    def _enhance_query(query: str, task_type: str = "") -> str:
        """
        增强 query 以提升 embedding 检索效果。

        核心问题：skill 文本是抽象策略描述，query 是具体问题，语义空间不对称。
        解决方案：
        1. 加入 task_type 上下文（如 "Task type: multi_hop_qa"）拉近语义距离
        2. 加入 BGE instruction prefix 激活非对称检索能力

        实验数据（multi_hop_qa top-1 cosine similarity）：
        - 基线（无 prefix）：0.30~0.33
        - + BGE prefix only：0.24~0.30（更差，因为 skill 不是传统 passage）
        - + task_type + prefix：0.50~0.52（+70% 提升，匹配准确率 100%）
        """
        parts = []
        if task_type:
            parts.append(f"Task type: {task_type}.")
        parts.append(f"Question: {query}")
        context_query = " ".join(parts)
        return SkillWorkspace._BGE_QUERY_PREFIX + context_query

    @staticmethod
    def _skill_to_text(skill: SkillEntry) -> str:
        """
        将 SkillEntry 转换为单一可 embed 文本（用于外部调用和 cross-encoder 等场景）。
        """
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
        """
        将 SkillEntry 拆分为 3 个语义维度的文本，分别编码。

        Multi-vector 方法：每个 skill 编码 3 个向量，query 与最相关维度匹配（取 max），
        避免不同维度的语义互相稀释。

        实验数据（10 queries, 12 skills）：
        - 单向量 (A): Hit@1=100%
        - 多向量 (B): Hit@1=100%，strategy_qa 匹配更准确（直接命中 Conditional Chain Deduction）

        3 个维度：
        1. name + description — 技能的核心功能描述
        2. trigger — 何时使用（最接近 SkillRL 的 when_to_apply）
        3. plan 摘要 — 具体执行步骤
        """
        texts = [
            # 维度 1: 功能描述
            (skill.name + " " + (skill.description or "")).strip(),
            # 维度 2: 触发条件（最关键的匹配维度）
            skill.trigger or skill.name,
            # 维度 3: 执行计划摘要
            (skill.plan or "")[:200] if skill.plan else skill.name,
        ]
        return texts

    def _compute_skill_embeddings(self) -> Optional[Dict]:
        """
        预计算所有技能的 multi-vector embeddings，带缓存。
        缓存在 add/remove/add_batch 时自动失效。

        Multi-vector：每个 skill 生成 3 个 embedding（功能 / 触发条件 / 计划），
        检索时 query 与 3 个维度分别计算相似度，取 max。

        Returns:
            {"skill_ids": List[str], "embeddings": np.ndarray (N, 3, D)}
            如果模型不可用返回 None
        """
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        model = _get_embedding_model(self._embedding_model_name)
        if model is None:
            return None

        if not self._skills:
            return None

        skill_ids = list(self._skills.keys())

        try:
            # 为每个 skill 编码 3 个维度
            multi_embeddings = []
            for sid in skill_ids:
                texts = self._skill_to_multi_text(self._skills[sid])
                vecs = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=64,
                )
                multi_embeddings.append(vecs)  # (3, D)

            cache = {
                "skill_ids": skill_ids,
                "embeddings": np.array(multi_embeddings, dtype=np.float32),  # (N, 3, D)
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
        """
        两阶段检索（论文 §3.4："skills ranked by Ĥ(s)"）。

        Stage 1: 按 task_type 过滤（primary key）
        Stage 2: 同 type 内按 F̂(s) 综合排序
          - Within same task_type: 0.6 * norm_flow + 0.3 * success_rate + 0.1 * cos_sim
          - Cross-type fallback: 0.4 * cos_sim + 0.3 * norm_flow + 0.3 * success_rate

        Returns:
            排序后的 top_k skills，如果 embedding 不可用返回 None
        """
        cache = self._compute_skill_embeddings()
        if cache is None:
            return None

        model = _get_embedding_model(self._embedding_model_name)
        if model is None:
            return None

        try:
            # 编码 query
            query_emb = model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_emb = np.array(query_emb, dtype=np.float32)  # (1, D)

            # Multi-vector cosine similarity
            all_sims = np.einsum('ijk,mk->ij', cache["embeddings"], query_emb)  # (N, 3)
            max_sims = np.max(all_sims, axis=1)  # (N,)

            skill_ids = cache["skill_ids"]

            # Stage 1: 分离 same-type 和 cross-type skills
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
                    # Stage 2a: same-type — cos_sim 主导，区分子类（代数 vs 概率 vs 几何）
                    # 实验验证：cos=0.7 权重使代数题选代数skill而非高sr的概率skill
                    # Bayesian prior: 新 skill (usage=0) 假设 sr=0.3，避免冷启动惩罚
                    usage_conf = min(usage / 10.0, 1.0)
                    effective_sr = usage_conf * success_rate + (1.0 - usage_conf) * 0.3
                    combined = 0.7 * cos_sim + 0.2 * norm_flow + 0.1 * effective_sr
                    same_type_scored.append((combined, cos_sim, skill))
                else:
                    # Stage 2b: cross-type fallback — embedding 权重更高
                    combined = 0.4 * cos_sim + 0.3 * norm_flow + 0.3 * success_rate
                    cross_type_scored.append((combined, cos_sim, skill))

            # 各组内降序排列
            same_type_scored.sort(key=lambda x: x[0], reverse=True)
            cross_type_scored.sort(key=lambda x: x[0], reverse=True)

            # 优先 same-type，不足时补充 cross-type
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
        """
        简单关键词匹配相似度（无需外部库）。

        综合比较 query 与 skill 的 trigger + description。
        """
        # 提取 query 关键词
        q_words = set(query.lower().split()) - SkillWorkspace._STOP_WORDS_SET
        if not q_words:
            return 0.3  # 空 query → 默认中等分

        # skill 文本
        skill_text = f"{skill.trigger} {skill.description} {skill.name}".lower()
        s_words = set(skill_text.split()) - SkillWorkspace._STOP_WORDS_SET

        # Jaccard-like 得分
        if not s_words:
            return 0.0

        common = q_words & s_words
        base_score = len(common) / (len(q_words | s_words) + 1e-8)

        # task_type 匹配奖励
        task_bonus = 0.2 if task_type and task_type.replace("_", " ") in skill_text else 0.0

        return min(1.0, base_score + task_bonus)

    @staticmethod
    def _normalize_flow(flow_score: float) -> float:
        """将 log-space flow_score 归一化到 [0, 1]（sigmoid）。

        flow_score 现在是 log F̂(s)，范围 [-20, 20]。
        sigmoid 中心点设为 0（log F̂ = 0 → F̂ = 1.0，中等 flow）。
        缩放因子 0.5 使区分度更好（[-20,20] 映射到 ~[0,1]）。
        """
        import math
        # clamp 防止 overflow
        x = max(min(flow_score * 0.5, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-x))

    def get_next_skill_id(self) -> str:
        """生成下一个可用的 dyn_NNN ID"""
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
        """
        剪枝低效技能。两种剪枝策略：

        策略 A（原有）：低 flow + 高使用量
          log F̂(s) < δ_prune AND usage_count >= N_prune
          AND last_used_step < current_step - recency_window
          AND success_rate < 0.3

        策略 B（新增）：Age-based 剪枝
          usage_count == 0 AND 存活超过 2 个 evolution cycle
          → 技能从未被选用，说明不匹配任何任务模式，应清除
        """
        with self._lock:
            to_prune = []
            age_pruned = []

            for sid, skill in self._skills.items():
                # ── 策略 B：Age-based 剪枝（0-usage 长期未用） ──
                if skill.meta.usage_count == 0 and current_step > 0:
                    age = current_step - skill.meta.creation_step
                    min_age = 2 * evolution_phase_steps  # 至少给 2 个进化周期的机会
                    if age >= min_age:
                        age_pruned.append(sid)
                        logger.info(
                            f"Prune (age) {sid} '{skill.name}': "
                            f"0 usage after {age} steps (min_age={min_age})"
                        )
                        continue

                # ── 策略 A：低 flow + 高使用量 ──
                if not (skill.meta.flow_score < prune_threshold
                        and skill.meta.usage_count >= min_usage):
                    continue

                # 保护：最近使用过的技能不剪枝（可能只是暂时 flow 低）
                if current_step > 0 and skill.meta.last_used_step >= current_step - recency_window:
                    logger.info(
                        f"Prune skip {sid}: recently used (last={skill.meta.last_used_step}, "
                        f"current={current_step}, window={recency_window})"
                    )
                    continue

                # 保护：成功率高的技能不剪枝
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
        """返回 workspace 统计摘要"""
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
