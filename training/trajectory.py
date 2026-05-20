

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"


def split_think_and_action(supervisor_output: str) -> Tuple[str, str]:

    close_pos = supervisor_output.find(_THINK_CLOSE_TAG)

    if close_pos == -1:
        if _THINK_OPEN_TAG in supervisor_output:

            return supervisor_output, ""
        else:

            return "", supervisor_output


    end_of_think = close_pos + len(_THINK_CLOSE_TAG)
    think_part = supervisor_output[:end_of_think]

    json_part = supervisor_output[end_of_think:].lstrip()
    return think_part, json_part


@dataclass
class Turn:


    supervisor_input: str       
    supervisor_output: str      
    action_type: str            


    skill_id: Optional[str] = None      
    instruction: Optional[str] = None   
    answer: Optional[str] = None        
    tool_args: Dict[str, Any] = field(default_factory=dict)  


    observation: str = ""


    forward_logprob: float = 0.0       
    backward_logprob: float = 0.0      
    action_token_count: int = 0        
    step_importance: float = 0.0       
    state_flow: float = 0.0            
    edge_flow: float = 0.0             


    messages_snapshot: Optional[List[Dict]] = None


    timestamp: float = field(default_factory=time.time)
    parse_error: bool = False
    raw_action: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:


    traj_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = ""
    gold_answer: str = ""
    task_type: str = ""
    task_type_id: int = 0


    turns: List[Turn] = field(default_factory=list)


    final_answer: str = ""
    reward: float = 0.0
    answer_reward: float = 0.0
    process_reward: float = 0.0
    skill_reward: float = 0.0
    r_tilde: float = 0.0               


    log_z: float = 0.0                 
    flow_weight: float = 1.0           


    completed: bool = False
    truncated: bool = False            
    n_skill_invoke: int = 0
    n_direct_act: int = 0
    n_tool_calls: int = 0
    n_parse_errors: int = 0
    skills_invoked: List[str] = field(default_factory=list)  
    tools_used: List[str] = field(default_factory=list)      

    def add_turn(self, turn: Turn) -> None:

        self.turns.append(turn)
        if turn.action_type == "skill_invoke":
            self.n_skill_invoke += 1
            if turn.skill_id:
                self.skills_invoked.append(turn.skill_id)
        elif turn.action_type in ("direct_act", "analyze"):
            self.n_direct_act += 1
        if turn.parse_error:
            self.n_parse_errors += 1

        if turn.action_type not in ("accept", "answer", "parse_error") and not turn.parse_error:
            self.n_tool_calls += 1
            self.tools_used.append(turn.action_type)

    @property
    def n_steps(self) -> int:
        return len(self.turns)

    @property
    def unique_skills(self) -> List[str]:
        return list(dict.fromkeys(self.skills_invoked))


    def to_forward_text(self) -> str:

        chunks = []
        for turn in self.turns:
            chunks.append(turn.supervisor_input)
            chunks.append(turn.supervisor_output)
        return "\n".join(chunks)

    def to_backward_text_per_turn(self, t: int) -> str:

        turn = self.turns[t]
        parts = [turn.supervisor_input]
        if turn.observation:
            parts.append(f"Observation: {turn.observation}")
        return "\n".join(parts)

    def get_spans(self) -> List[tuple[str, str]]:

        return [(turn.supervisor_input, turn.supervisor_output) for turn in self.turns]


    def to_dict(self) -> Dict[str, Any]:

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

        traj.n_skill_invoke = d.get("n_skill_invoke", 0)
        traj.n_direct_act = d.get("n_direct_act", 0)
        traj.n_tool_calls = d.get("n_tool_calls", 0)
        traj.n_parse_errors = d.get("n_parse_errors", 0)
        traj.skills_invoked = d.get("skills_invoked", [])
        traj.tools_used = d.get("tools_used", [])
        return traj
