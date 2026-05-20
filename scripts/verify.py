"""
SkillFlow 验证脚本 — 按计划中的 5 个验证方案依次检查。

用法：
  cd /path/to/skillflow
  python scripts/verify.py [--m-exec-only] [--flow-only] [--all]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# 加入项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_skill_format() -> bool:
    """测试技能格式序列化/反序列化"""
    print("\n=== Test 1: Skill Format ===")
    from src.skills.format import SkillEntry, SkillMeta

    meta = SkillMeta(skill_id="dyn_001", source="genesis", flow_score=0.5)
    skill = SkillEntry(
        name="Multi-Hop Fact Chaining",
        description="Chain multiple facts to answer complex questions",
        trigger="Questions requiring 2+ facts; bridge entity questions",
        plan="1. Identify bridge entities\n2. Find connecting facts\n3. Chain to final answer",
        pitfall="Don't ask for final answer in step 1",
        constraint="Maximum 5 steps",
        meta=meta,
    )

    # 验证
    valid, reason = skill.validate()
    print(f"  validate(): {valid} ({reason})")

    # 序列化/反序列化
    md = skill.to_markdown()
    skill2 = SkillEntry.from_markdown(md)
    assert skill2.name == skill.name, f"Name mismatch: {skill2.name} != {skill.name}"
    assert skill2.meta.skill_id == "dyn_001"
    print(f"  Serialize/Deserialize: OK")

    # prompt 格式
    prompt_text = skill.format_for_prompt()
    assert "dyn_001" in prompt_text
    print(f"  format_for_prompt(): OK")

    print("  [PASS] Skill Format")
    return True


def test_m_exec_connectivity(api_base: str, model: str) -> bool:
    """测试 M_exec 连通性"""
    print(f"\n=== Test 2: M_exec Connectivity ({api_base}) ===")
    from src.executor.m_exec import MExec

    m_exec = MExec(api_base=api_base, model_name=model)
    try:
        ok = m_exec.test_connectivity()
        print(f"  test_connectivity(): {'OK' if ok else 'FAILED'}")
        if ok:
            # 实际调用
            result = m_exec.execute("What is 7 times 8? Answer with just the number.")
            print(f"  7*8 = '{result[:50]}'")
            assert "56" in result, f"Expected '56' in result: {result}"
            print("  [PASS] M_exec Connectivity")
        return ok
    except Exception as e:
        print(f"  Error: {e}")
        print("  [SKIP] M_exec not available")
        return False


def test_flow_metrics() -> bool:
    """测试 flow entropy 触发逻辑"""
    print("\n=== Test 3: Flow Entropy Trigger ===")
    from training.trajectory import Trajectory, Turn
    from training.flow_metrics import compute_flow_entropy

    def make_traj(reward: float, log_z: float = 0.0) -> Trajectory:
        traj = Trajectory(question="test", gold_answer="ans", task_type="factual_qa")
        turn = Turn(
            supervisor_input="q",
            supervisor_output='{"action_type":"accept","answer":"ans"}',
            action_type="accept",
        )
        turn.forward_logprob = -0.5
        turn.backward_logprob = -0.3
        turn.state_flow = log_z + turn.forward_logprob - turn.backward_logprob
        traj.turns.append(turn)
        traj.reward = reward
        traj.r_tilde = reward + 0.1
        traj.log_z = log_z
        return traj

    # Case 1: 极不均匀 flow → Ĥ < δ_H → 触发进化
    trajs_low_entropy = [make_traj(1.0, log_z=5.0)] + [make_traj(0.1, log_z=0.0)] * 7
    h_low = compute_flow_entropy(trajs_low_entropy)
    print(f"  Low-diversity Ĥ_flow = {h_low:.3f}")

    # Case 2: 均匀 flow → Ĥ ≈ log|B| → 不触发
    trajs_high_entropy = [make_traj(0.5, log_z=1.0)] * 8
    h_high = compute_flow_entropy(trajs_high_entropy)
    print(f"  High-diversity Ĥ_flow = {h_high:.3f}")

    assert h_low < h_high, f"Expected h_low < h_high: {h_low} vs {h_high}"

    delta_h = 2.0
    print(f"  δ_H = {delta_h}")
    print(f"  Low-entropy triggers evolution: {h_low < delta_h}")
    print(f"  High-entropy suppresses evolution: {h_high >= delta_h}")

    print("  [PASS] Flow Metrics")
    return True


def test_reward_functions() -> bool:
    """测试奖励函数"""
    print("\n=== Test 4: Reward Functions ===")
    from training.reward import token_f1, exact_match, label_accuracy, compute_answer_reward

    assert token_f1("Paris is the capital", "Paris") > 0.3, "token_f1 failed"
    assert exact_match("42", "42") == 1.0
    assert exact_match("41", "42") == 0.0
    assert label_accuracy("supports", "SUPPORTS") == 1.0
    assert label_accuracy("refutes", "SUPPORTS") == 0.0

    # Multi-candidate
    r = compute_answer_reward("Paris", "Paris|paris|PARIS", "factual_qa")
    assert r > 0, f"Multi-candidate failed: {r}"

    print(f"  token_f1('Paris is the capital', 'Paris') = {token_f1('Paris is the capital', 'Paris'):.3f}")
    print(f"  label_accuracy('supports', 'SUPPORTS') = {label_accuracy('supports', 'SUPPORTS')}")
    print(f"  multi-candidate reward('Paris', 'Paris|paris|PARIS') = {r:.3f}")
    print("  [PASS] Reward Functions")
    return True


def test_action_parsing() -> bool:
    """测试 Supervisor 动作解析（<tool_call> 格式 + 旧 JSON 兼容）"""
    print("\n=== Test 5: Action Parsing ===")
    from training.environment import parse_supervisor_action, parse_tool_call_output

    # ── A. Qwen3 原生 <tool_call> 格式 ──
    print("  --- <tool_call> format ---")

    # skill_invoke via <tool_call>
    tc1 = parse_supervisor_action(
        '<tool_call>\n{"name": "skill_invoke", "arguments": {"skill_id": "dyn_001", "instruction": "Find X"}}\n</tool_call>'
    )
    assert tc1.action_type == "skill_invoke"
    assert tc1.skill_id == "dyn_001"
    assert tc1.instruction == "Find X"
    assert not tc1.parse_error
    print(f"  <tool_call> skill_invoke: {tc1.action_type} [{tc1.skill_id}]")

    # direct_act via <tool_call>
    tc2 = parse_supervisor_action(
        '<tool_call>\n{"name": "direct_act", "arguments": {"instruction": "Calculate 2+2"}}\n</tool_call>'
    )
    assert tc2.action_type == "direct_act"
    assert tc2.instruction == "Calculate 2+2"
    print(f"  <tool_call> direct_act: {tc2.action_type}")

    # accept via <tool_call>
    tc3 = parse_supervisor_action(
        '<tool_call>\n{"name": "accept", "arguments": {"answer": "Paris"}}\n</tool_call>'
    )
    assert tc3.action_type == "accept"
    assert tc3.answer == "Paris"
    print(f"  <tool_call> accept: {tc3.action_type}, answer={tc3.answer}")

    # <tool_call> with broken JSON → parse_error (no fallback)
    tc4 = parse_supervisor_action(
        '<tool_call>\n{"name": "accept", "arguments": {"answer": "trunc\n</tool_call>'
    )
    assert tc4.parse_error
    print(f"  <tool_call> broken JSON → parse_error: {tc4.parse_error}")

    # <tool_call> with unknown function name → parse_error
    tc5 = parse_supervisor_action(
        '<tool_call>\n{"name": "unknown_func", "arguments": {}}\n</tool_call>'
    )
    assert tc5.parse_error
    print(f"  <tool_call> unknown func → parse_error: {tc5.parse_error}")

    # <tool_call> with think field in arguments
    tc6 = parse_supervisor_action(
        '<tool_call>\n{"name": "direct_act", "arguments": {"think": "reasoning here", "instruction": "do X"}}\n</tool_call>'
    )
    assert tc6.action_type == "direct_act"
    assert tc6.think == "reasoning here"
    print(f"  <tool_call> with think: OK")

    # ── B. 旧 JSON 格式向后兼容 ──
    print("  --- Legacy JSON format (backward compat) ---")

    # skill_invoke (legacy)
    action1 = parse_supervisor_action(
        '{"think": "I need to search", "action_type": "skill_invoke", "skill_id": "dyn_001", "instruction": "Find X"}'
    )
    assert action1.action_type == "skill_invoke"
    assert action1.skill_id == "dyn_001"
    print(f"  legacy skill_invoke: {action1.action_type} [{action1.skill_id}]")

    # direct_act (legacy)
    action2 = parse_supervisor_action(
        '{"action_type": "direct_act", "instruction": "Calculate 2+2"}'
    )
    assert action2.action_type == "direct_act"
    print(f"  legacy direct_act: {action2.action_type}")

    # accept (legacy)
    action3 = parse_supervisor_action(
        '{"action_type": "accept", "answer": "Paris"}'
    )
    assert action3.action_type == "accept"
    assert action3.answer == "Paris"
    print(f"  legacy accept: {action3.action_type}, answer={action3.answer}")

    # ── C. 完全无效输出 ──
    action4 = parse_supervisor_action("This is not JSON at all")
    assert action4.parse_error
    print(f"  garbage → parse_error: {action4.parse_error}")

    # ── D. parse_tool_call_output 返回 None 时触发 fallback ──
    assert parse_tool_call_output("no tool call here") is None
    print(f"  no <tool_call> → returns None (triggers fallback)")

    # ── E. Fix 2: dyn_xxx 自动映射到 skill_invoke ──
    tc_dyn = parse_supervisor_action(
        '<tool_call>\n{"name": "dyn_014", "arguments": {"titles": ["Paper A"]}}\n</tool_call>'
    )
    assert tc_dyn.action_type == "skill_invoke"
    assert tc_dyn.skill_id == "dyn_014"
    print(f"  dyn_014 → skill_invoke [dyn_014]: OK")

    # ── F. Fix 3: 纯文本答案 auto-wrap 为 accept ──
    tc_plain = parse_supervisor_action(
        "The context provided does not mention anything about the specific population. Based on the available information, the answer is approximately 50,000."
    )
    assert tc_plain.action_type == "accept"
    assert "50,000" in (tc_plain.answer or "")
    print(f"  plain text → accept: OK")

    # 短 garbage 仍然是 parse_error
    tc_short = parse_supervisor_action("I don't know")
    assert tc_short.parse_error
    print(f"  short garbage → parse_error: OK")

    print("  [PASS] Action Parsing")
    return True


def test_ttb_loss() -> bool:
    """测试 TTB 损失计算"""
    print("\n=== Test 6: TTB Loss ===")
    try:
        import torch
        from training.gflownet_trainer import compute_ttb_loss

        B = 4
        log_z = torch.tensor([1.0, 0.5, 2.0, 1.5], requires_grad=True)
        fwd_sums = torch.tensor([-2.0, -1.5, -3.0, -2.5], requires_grad=True)
        bwd_sums = torch.tensor([-1.5, -1.2, -2.5, -2.0], requires_grad=True)
        r_tilde = torch.tensor([0.8, 0.3, 1.2, 0.6])

        loss = compute_ttb_loss(log_z, fwd_sums, bwd_sums, r_tilde, beta=1.0)
        print(f"  TTB loss = {loss.item():.4f}")
        assert not torch.isnan(loss), "TTB loss is NaN!"
        assert loss.item() > 0, "TTB loss should be positive"
        print("  Gradients check...")
        loss.backward()
        print(f"  log_z.grad norm = {log_z.grad.norm().item():.4f}")
        assert log_z.grad is not None, "log_z has no grad"
        print("  [PASS] TTB Loss")
    except ImportError:
        print("  [SKIP] PyTorch not available")
        return False
    return True


def test_workspace() -> bool:
    """测试 SkillWorkspace"""
    print("\n=== Test 7: SkillWorkspace ===")
    import tempfile
    from src.skills.format import SkillEntry, SkillMeta
    from src.skills.workspace import SkillWorkspace

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = SkillWorkspace(skills_dir=Path(tmpdir), max_skills=10)

        for i in range(3):
            meta = SkillMeta(skill_id=f"dyn_{i:03d}", source="test")
            meta.flow_score = float(i)
            skill = SkillEntry(
                name=f"Test Skill {i}",
                description=f"Description {i}",
                trigger=f"When task type is type_{i}",
                plan=f"1. First step {i}\n2. Second step {i}\n3. Third step {i} and more words here",
                pitfall=f"Pitfall {i}",
                constraint=f"Max {i+2} steps",
                meta=meta,
            )
            ws.add(skill)

        assert ws.size == 3
        print(f"  Added 3 skills, size={ws.size}")

        # 检索
        results = ws.retrieve("type_1 task", top_k=2)
        print(f"  Retrieved {len(results)} relevant skills")
        assert len(results) > 0

        # Flow score update
        ws.update_flow_score("dyn_000", 5.0, alpha=1.0)
        updated = ws.get_by_id("dyn_000")
        assert updated.meta.flow_score == 5.0
        print(f"  Flow score update: OK")

        # 磁盘持久化验证
        ws2 = SkillWorkspace(skills_dir=Path(tmpdir), max_skills=10)
        assert ws2.size == 3
        print(f"  Disk persistence: OK ({ws2.size} skills loaded)")

    print("  [PASS] SkillWorkspace")
    return True


def test_tools() -> bool:
    """测试 5 个真实工具的执行"""
    print("\n=== Test 8: Tool Execution ===")
    from training.tools import execute_tool, is_tool_success, TOOL_ACTION_TYPES

    all_ok = True

    # ── python_exec ──
    print("  --- python_exec ---")
    out = execute_tool("python_exec", {"code": "print(7 * 8)"}, [], {})
    print(f"    7*8: {out.strip()}")
    assert "56" in out, f"Expected 56 in output: {out}"
    assert is_tool_success(out)

    out_err = execute_tool("python_exec", {"code": "1/0"}, [], {})
    print(f"    1/0: {out_err[:80]}")
    assert "ZeroDivisionError" in out_err
    assert not is_tool_success(out_err)  # EXCEPTION → not success

    out_empty = execute_tool("python_exec", {"code": ""}, [], {})
    assert "[ERROR]" in out_empty
    print("    [OK] python_exec")

    # ── calculator ──
    print("  --- calculator ---")
    out = execute_tool("calculator", {"expression": "2**10 + sqrt(144)"}, [], {})
    print(f"    2**10+sqrt(144): {out}")
    assert "1036" in out
    assert is_tool_success(out)

    out_pi = execute_tool("calculator", {"expression": "pi * 2"}, [], {})
    assert "6.28" in out_pi
    print(f"    pi*2: {out_pi}")

    out_bad = execute_tool("calculator", {"expression": "import os"}, [], {})
    assert "[ERROR]" in out_bad
    print("    [OK] calculator")

    # ── search_context ──
    print("  --- search_context ---")
    ctx = [
        {"text": "Paris is the capital of France. It is known for the Eiffel Tower.", "title": "France"},
        {"text": "Berlin is the capital of Germany. It has the Brandenburg Gate.", "title": "Germany"},
        {"text": "Tokyo is the capital of Japan. Mount Fuji is nearby.", "title": "Japan"},
    ]
    out = execute_tool("search_context", {"query": "capital France Eiffel"}, ctx, {})
    print(f"    search 'capital France Eiffel': {out[:100]}...")
    assert "France" in out or "Paris" in out
    assert is_tool_success(out)

    out_empty = execute_tool("search_context", {"query": "test"}, [], {})
    assert "[NO_CONTEXT]" in out_empty

    # Fix 1: search_context with embedded passages in question_text
    question_with_passages = (
        "[Mabel Pines] Mabel is a character from Gravity Falls animated series. "
        "She loves arts and crafts. "
        "[Dipper Pines] Dipper is Mabel's twin brother. He is fascinated by mysteries."
    )
    out_embed = execute_tool(
        "search_context", {"query": "animated series character"},
        [], {"question_text": question_with_passages}
    )
    assert is_tool_success(out_embed), f"Embedded passage search failed: {out_embed}"
    assert "Mabel" in out_embed or "Gravity Falls" in out_embed
    print(f"    embedded passages fallback: OK")
    print("    [OK] search_context")

    # ── verify_answer ──
    print("  --- verify_answer ---")
    out = execute_tool("verify_answer", {"answer": "42", "check_type": "numeric"}, [], {})
    assert "[PASS]" in out
    print(f"    numeric '42': {out}")

    out = execute_tool("verify_answer", {"answer": "supports", "check_type": "label"}, [], {})
    assert "[PASS]" in out

    out = execute_tool("verify_answer", {"answer": "def f(): pass", "check_type": "code"}, [], {})
    assert "[PASS]" in out

    out = execute_tool("verify_answer", {"answer": ""}, [], {})
    assert "[FAIL]" in out

    out_fmt = execute_tool("verify_answer", {"answer": "42", "check_type": "format"}, [], {"task_type": "math_reasoning"})
    assert "[PASS]" in out_fmt
    print("    [OK] verify_answer")

    # ── test_code ──
    print("  --- test_code ---")
    code = "def add(a, b): return a + b"
    tests = "assert add(1, 2) == 3\nassert add(-1, 1) == 0"
    out = execute_tool("test_code", {"code": code, "test_cases": tests}, [], {})
    print(f"    test add(): {out.splitlines()[0]}")
    assert "[PASS]" in out
    assert "2/2" in out

    bad_code = "def add(a, b): return a - b"
    out = execute_tool("test_code", {"code": bad_code, "test_cases": tests}, [], {})
    assert "[FAIL]" in out or "[PARTIAL]" in out
    print("    [OK] test_code")

    # ── v3 tool_call parsing ──
    print("  --- v3 tool_call parsing ---")
    from training.environment import parse_supervisor_action, _VALID_ACTION_TYPES

    # think via <tool_call>
    tc = parse_supervisor_action(
        '<tool_call>\n{"name": "think", "arguments": {"thought": "Let me reason about this"}}\n</tool_call>'
    )
    assert tc.action_type == "think"
    print(f"    <tool_call> think: OK")

    # search via <tool_call>
    tc2 = parse_supervisor_action(
        '<tool_call>\n{"name": "search", "arguments": {"query": "capital of France"}}\n</tool_call>'
    )
    assert tc2.action_type == "search"
    print(f"    <tool_call> search: OK")

    # python_execute via <tool_call>
    tc3 = parse_supervisor_action(
        '<tool_call>\n{"name": "python_execute", "arguments": {"instruction": "compute 2+3 using sympy"}}\n</tool_call>'
    )
    assert tc3.action_type == "python_execute"
    print(f"    <tool_call> python_execute: OK")

    # verify_answer via <tool_call>
    tc4 = parse_supervisor_action(
        '<tool_call>\n{"name": "verify_answer", "arguments": {"answer": "42", "method": "substitute"}}\n</tool_call>'
    )
    assert tc4.action_type == "verify_answer"
    print(f"    <tool_call> verify_answer: OK")

    # test_code via <tool_call>
    tc5 = parse_supervisor_action(
        '<tool_call>\n{"name": "test_code", "arguments": {"instruction": "implement add function"}}\n</tool_call>'
    )
    assert tc5.action_type == "test_code"
    print(f"    <tool_call> test_code: OK")

    # All 23 v3 tools are in _VALID_ACTION_TYPES
    for tool in ["skill_invoke", "think", "plan", "decompose", "python_execute",
                  "test_code", "analyze", "search", "lookup", "fact_verify",
                  "ask_llm", "self_consistency", "verify_answer", "check_answer",
                  "cross_validate", "search_code", "view_file", "edit_file",
                  "run_tests", "act", "search_product", "click", "accept"]:
        assert tool in _VALID_ACTION_TYPES, f"{tool} not in _VALID_ACTION_TYPES"
    print(f"    All 23 v3 tools in _VALID_ACTION_TYPES: OK")

    print("  [PASS] Tool Execution")
    return True


def test_tool_rewards() -> bool:
    """测试 Tool-R1 R_exec 奖励计算"""
    print("\n=== Test 9: Tool-R1 Rewards ===")
    from training.reward import compute_process_reward
    from training.trajectory import Turn

    # v3: 模拟工具使用轨迹
    turns = [
        Turn(supervisor_input="", supervisor_output="", action_type="search",
             instruction="find info", observation="[Match 1] Paris is the capital"),
        Turn(supervisor_input="", supervisor_output="", action_type="python_execute",
             instruction="compute 2+3", observation="[Result] 5"),
        Turn(supervisor_input="", supervisor_output="", action_type="verify_answer",
             instruction="check: 42", observation="VERIFIED"),
        Turn(supervisor_input="", supervisor_output="", action_type="think",
             instruction="let me reason", observation="[Thought] reasoning"),
    ]

    r = compute_process_reward(turns)
    print(f"  R_process (search+python_execute+verify_answer+think) = {r:.3f}")
    # Expected: search(0.03) + python_execute(0.03) + verify_answer(0.03) + think(0.01) = 0.10
    assert r > 0, f"Expected positive reward, got {r}"

    # v3: 对比 skill_invoke 轨迹
    turns_skill = [
        Turn(supervisor_input="", supervisor_output="", action_type="skill_invoke",
             instruction="apply strategy", observation="[Strategy: ...]"),
    ]
    r_skill = compute_process_reward(turns_skill)
    print(f"  R_process (1 skill_invoke) = {r_skill:.3f}")
    # Expected: skill_invoke(0.05)
    assert r_skill > 0

    print("  [PASS] Tool-R1 Rewards")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SkillFlow components")
    parser.add_argument("--m-exec-only", action="store_true")
    parser.add_argument("--flow-only", action="store_true")
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument(
        "--executor-api",
        type=str,
        default="http://localhost:8004/v1",
        help="M_exec API base URL",
    )
    parser.add_argument(
        "--executor-model",
        type=str,
        default="gpt-oss-120b",
    )
    args = parser.parse_args()

    results = {}

    # 总是运行不依赖外部服务的测试
    results["skill_format"] = test_skill_format()
    results["action_parsing"] = test_action_parsing()
    results["reward_functions"] = test_reward_functions()
    results["flow_metrics"] = test_flow_metrics()
    results["ttb_loss"] = test_ttb_loss()
    results["workspace"] = test_workspace()
    results["tools"] = test_tools()
    results["tool_rewards"] = test_tool_rewards()

    # 外部服务测试（可选）
    if args.m_exec_only or args.all:
        results["m_exec"] = test_m_exec_connectivity(args.executor_api, args.executor_model)

    # 汇总
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL/SKIP"
        print(f"  {name:30s}: {status}")

    n_pass = sum(1 for v in results.values() if v)
    print(f"\n{n_pass}/{len(results)} tests passed")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
