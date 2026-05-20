"""
SkillFlow Trajectory — 轨迹数据结构，含 flow 字段。

轨迹 τ = (q, s_0, a_1, o_1, s_1, ..., a_T, o_T, r)
每步携带：
  - action_type: one of 23 tools (skill_invoke, think, plan, search, accept, etc.)
  - forward/backward logprobs（训练时填充）
  - step_importance I(t)（flow 计算后填充）
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────
# Think-token 分离工具
# ──────────────────────────────────────────────────────

_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


def split_think_and_action(supervisor_output: str) -> Tuple[str, str]:
    """
    将 Qwen3-8B 的输出拆分为 (think_part, json_part)。

    论文 §3.2 区分：
      a_t^think  — 内部思考过程（<think>...</think>），不计入 TTB loss
      a_t^out    — 实际结构化动作（JSON），是 logprob target

    TTB loss 只对 a_t = (α_t, a_t^out) 计算 log π_θ / log P_φ。
    think 部分在 teacher-forcing 时扩展进 context（模型 forward pass 能看到），
    但不进入 log-sum，从而避免 thinking token 污染 TTB balance error。

    Returns:
        (think_part, json_part)：
        - think_part: 从输出开头到 </think>（含标签），保留原始 token 序列
        - json_part:  </think> 之后去掉前导空白的内容（JSON 动作）

    边界情况：
        - 无 </think> 且有 <think>：think 块被截断，think_part=全部，json_part=""
          → 调用方应降级：对整个 action_text 计算（包含 think）
        - 无任何 think 标签：无思考模式输出，think_part=""，json_part=全部文本
    """
    close_pos = supervisor_output.find(_THINK_CLOSE_TAG)

    if close_pos == -1:
        if _THINK_OPEN_TAG in supervisor_output:
            # 有开标签但无闭标签（截断）：全部视为思考，JSON 部分为空
            return supervisor_output, ""
        else:
            # 完全无 think 标签（无思考模式 / base model 直接输出 JSON）
            return "", supervisor_output

    # 正常路径：切分
    end_of_think = close_pos + len(_THINK_CLOSE_TAG)
    think_part = supervisor_output[:end_of_think]
    # lstrip() 去掉 </think> 和 JSON 之间可能的换行/空格
    json_part = supervisor_output[end_of_think:].lstrip()
    return think_part, json_part


@dataclass
class Turn:
    """单个交互轮次（Supervisor 的一步）

    v4: Multi-turn tool calling format.
      - action_type: tool_name from API (or "answer" for direct text answer)
      - supervisor_output: raw content string from model
      - tool_args: parsed tool arguments dict (from API)
      - answer: final answer text (when action_type == "answer")
    """

    # 输入 / 输出
    supervisor_input: str       # H_{t-1}（legacy: prompt string; v4: empty, messages-based）
    supervisor_output: str      # Supervisor 的原始输出 content
    action_type: str            # v4: tool_name or "answer" for direct text response

    # action 内容
    skill_id: Optional[str] = None      # action_type==skill_invoke 时
    instruction: Optional[str] = None   # 发给 M_exec 的指令 / primary arg text
    answer: Optional[str] = None        # action_type==answer 时的最终答案
    tool_args: Dict[str, Any] = field(default_factory=dict)  # v4: parsed tool arguments

    # M_exec 结果
    observation: str = ""

    # Flow 相关字段（训练时填充）
    forward_logprob: float = 0.0       # log π_θ(a_t | H_{t-1})
    backward_logprob: float = 0.0      # log P_φ(a_t | H_t)
    action_token_count: int = 0        # |a_t|_tokens — action 的 token 数（用于 H_flow 归一化）
    step_importance: float = 0.0       # I(t) = exp(fwd - bwd)
    state_flow: float = 0.0            # log F(s_t)
    edge_flow: float = 0.0             # log F(s_{t-1} → s_t)

    # Multi-turn messages snapshot（v6: 用于 logprob 计算时的 chat template 上下文）
    messages_snapshot: Optional[List[Dict]] = None

    # 元数据
    timestamp: float = field(default_factory=time.time)
    parse_error: bool = False
    raw_action: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """完整的 episode 轨迹"""

    # 任务信息
    traj_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    gold_answer: str = ""
    task_type: str = ""
    task_type_id: int = 0

    # 轨迹步骤
    turns: List[Turn] = field(default_factory=list)

    # 结果
    final_answer: str = ""
    reward: float = 0.0
    answer_reward: float = 0.0
    process_reward: float = 0.0
    skill_reward: float = 0.0
    r_tilde: float = 0.0               # R̃(τ) = R(τ) + ε_min（GFlowNet 正支撑）

    # Flow 字段（batch 完成后填充）
    log_z: float = 0.0                 # Z_θ(q) 分区函数对数
    flow_weight: float = 1.0           # F(τ)/Σ F(τ') — 归一化流量（用于 entropy）

    # 统计
    completed: bool = False
    truncated: bool = False            # 超过 max_episode_steps
    n_skill_invoke: int = 0
    n_direct_act: int = 0
    n_tool_calls: int = 0
    n_parse_errors: int = 0
    skills_invoked: List[str] = field(default_factory=list)  # skill_id 列表（按序）
    tools_used: List[str] = field(default_factory=list)      # 工具名列表（按序）

    def add_turn(self, turn: Turn) -> None:
        """添加一个轮次并更新统计。"""
        self.turns.append(turn)
        if turn.action_type == "skill_invoke":
            self.n_skill_invoke += 1
            if turn.skill_id:
                self.skills_invoked.append(turn.skill_id)
        elif turn.action_type in ("direct_act", "analyze"):
            self.n_direct_act += 1
        if turn.parse_error:
            self.n_parse_errors += 1
        # v4: Track all tool usage ("answer" = direct text, not a tool call)
        if turn.action_type not in ("accept", "answer", "parse_error") and not turn.parse_error:
            self.n_tool_calls += 1
            self.tools_used.append(turn.action_type)

    @property
    def n_steps(self) -> int:
        return len(self.turns)

    @property
    def unique_skills(self) -> List[str]:
        return list(dict.fromkeys(self.skills_invoked))

    # ──────────────────────────────────────────────
    # 训练用：构建 tokenizer 输入
    # ──────────────────────────────────────────────

    def to_forward_text(self) -> str:
        """
        构建 forward policy 的训练文本。

        格式：concatenation of (supervisor_input, supervisor_output) per turn.
        后续 tokenizer 会对齐 action_mask，只计算 supervisor_output 部分的 loss。
        """
        chunks = []
        for turn in self.turns:
            chunks.append(turn.supervisor_input)
            chunks.append(turn.supervisor_output)
        return "\n".join(chunks)

    def to_backward_text_per_turn(self, t: int) -> str:
        """
        构建 backward policy 在第 t 步的输入文本。

        论文 P_φ(a_t | H_{t-1} ⊕ o_t)：
          context = H_{t-1} + o_t（不含 a_t，否则预测 a_t 变为平凡任务）
          target  = supervisor_output（a_t）

        语义："给定前一状态 H_{t-1} 和本步执行结果 o_t，反推 Supervisor 做了什么决策"
        这让 P_φ 学习到哪些动作真正有效（高回报观测→应该被"回溯"到）。
        """
        turn = self.turns[t]
        parts = [turn.supervisor_input]
        if turn.observation:
            parts.append(f"Observation: {turn.observation}")
        return "\n".join(parts)

    def get_spans(self) -> List[tuple[str, str]]:
        """
        返回 (context_text, action_text) 对，用于计算 action_mask。

        context_text: 发给 Supervisor 的 prompt（不计 loss）
        action_text: Supervisor 输出（计 loss）
        """
        return [(turn.supervisor_input, turn.supervisor_output) for turn in self.turns]

    # ──────────────────────────────────────────────
    # 序列化
    # ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict（用于 JSONL 日志）"""
        return {
            "traj_id": self.traj_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "task_type": self.task_type,
            "final_answer": self.final_answer,
            "reward": self.reward,
            "answer_reward": self.answer_reward,
            "r_tilde": self.r_tilde,
            "log_z": self.log_z,
            "n_steps": self.n_steps,
            "n_skill_invoke": self.n_skill_invoke,
            "n_direct_act": self.n_direct_act,
            "n_tool_calls": self.n_tool_calls,
            "n_parse_errors": self.n_parse_errors,
            "skills_invoked": self.skills_invoked,
            "tools_used": self.tools_used,
            "completed": self.completed,
            "truncated": self.truncated,
            "turns": [
                {
                    "action_type": t.action_type,
                    "skill_id": t.skill_id,
                    "instruction": t.instruction or "",
                    "observation": t.observation or "",
                    "forward_logprob": t.forward_logprob,
                    "backward_logprob": t.backward_logprob,
                    "action_token_count": t.action_token_count,
                    "step_importance": t.step_importance,
                    "parse_error": t.parse_error,
                }
                for t in self.turns
            ],
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trajectory":
        """从 dict 反序列化"""
        traj = cls(
            traj_id=d.get("traj_id", str(uuid.uuid4())[:8]),
            question=d.get("question", ""),
            gold_answer=d.get("gold_answer", ""),
            task_type=d.get("task_type", ""),
            final_answer=d.get("final_answer", ""),
            reward=d.get("reward", 0.0),
            answer_reward=d.get("answer_reward", 0.0),
            r_tilde=d.get("r_tilde", 0.0),
            log_z=d.get("log_z", 0.0),
            completed=d.get("completed", False),
            truncated=d.get("truncated", False),
        )
        for td in d.get("turns", []):
            turn = Turn(
                supervisor_input="",
                supervisor_output="",
                action_type=td.get("action_type", "think"),
                skill_id=td.get("skill_id"),
                instruction=td.get("instruction"),
                observation=td.get("observation", ""),
                forward_logprob=td.get("forward_logprob", 0.0),
                backward_logprob=td.get("backward_logprob", 0.0),
                action_token_count=td.get("action_token_count", 0),
                step_importance=td.get("step_importance", 0.0),
                parse_error=td.get("parse_error", False),
            )
            traj.turns.append(turn)
        # 重建统计
        traj.n_skill_invoke = d.get("n_skill_invoke", 0)
        traj.n_direct_act = d.get("n_direct_act", 0)
        traj.n_tool_calls = d.get("n_tool_calls", 0)
        traj.n_parse_errors = d.get("n_parse_errors", 0)
        traj.skills_invoked = d.get("skills_invoked", [])
        traj.tools_used = d.get("tools_used", [])
        return traj
