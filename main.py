"""
文档问答 Agent 主循环(复用 week3 agent_loop:压缩 / goal 判断器 / trace 全保留)
改动:去掉 MCP 与工具缓存;系统提示词改为"检索 → 带引用回答"
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

from llm import llm_chat
from tool_system import tool_registry, execute_tool
from memory import create_memory_session, MemoryManager
from context_compactor import ContextCompactor
from goal_evaluator import GoalEvaluator
from trace.tracer import Tracer, set_tracer, clear_tracer

MAX_ITER = int(os.getenv("MAX_ITER", 8))
GOAL_BLOCK_CAP = 3  # goal loop:连续未达成的最大轮数,防死循环

TOOL_INFO = tool_registry.get_all_schemas()

DEFAULT_PROMPT = """你是文档问答 Agent,基于已加载的语料回答业务问题。
工作流程:
1. 先用 retrieve 检索相关段落(一个问法没结果就换关键词再检索);
2. 需要精确数字/表格时用 query_table 读 Excel;
3. 回答必须基于检索到的内容,并用 cite_source 附上出处(id 用逗号分隔);
4. 检索不到就说"语料中没有相关信息",绝不编造。"""


def agent_loop(user_query: str, memory_session: MemoryManager,
               completion_condition: str = "", system_prompt: str | None = None) -> str:
    # ===== 初始化 Tracer =====
    tracer = Tracer(session_id=memory_session.sid, user_input=user_query, max_iterations=MAX_ITER)
    set_tracer(tracer)
    tracer.add_event("user_input", {"query": user_query})

    system_prompt = system_prompt or DEFAULT_PROMPT
    compactor = ContextCompactor(llm_chat, token_threshold=1500, keep_recent_rounds=8)
    compactor.set_system_prompt(system_prompt)

    messages = memory_session.get_short_messages()
    messages = [{"role": "system", "content": system_prompt}] + messages
    messages.append({"role": "user", "content": user_query})

    iter_count = 0
    # Goal Loop:有完成条件才启用独立判断器(无条件时零开销)
    goal_eval = GoalEvaluator(llm_chat) if completion_condition else None
    consecutive_blocks = 0
    print("===== Agent 执行开始 =====")

    while iter_count < MAX_ITER:
        # 上下文压缩
        if compactor.is_need_compact(messages):
            print("\n========== 触发上下文自动压缩 ==========")
            messages, before_tok, after_tok = compactor.compact_messages(messages)
            print(f"压缩前Token:{before_tok} → 压缩后:{after_tok}")
            print("========================================\n")

        iter_count += 1
        tracer.increment_iteration()

        # ===== LLM 调用 =====
        llm_resp = llm_chat(messages, tools=TOOL_INFO)
        choice = llm_resp["choices"][0]
        msg = choice["message"]
        tracer.add_event(
            "llm_decision",
            {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
                "finish_reason": choice.get("finish_reason", ""),
            },
            token_count=llm_resp.get("usage", {}).get("total_tokens", 0),
        )
        memory_session.add_short_msg(msg["role"], msg["content"])

        # ===== 无工具调用 → 结束 或 goal 判断 =====
        if not msg.get("tool_calls"):
            if goal_eval is None:
                final_output = msg["content"]
                tracer.set_final_output(final_output, success=True)
                tracer.save()
                clear_tracer()
                print(f"模型最终回答:\n{final_output}")
                return final_output

            messages.append(msg)
            try:
                decision = goal_eval.evaluate(completion_condition, messages)
            except Exception as e:
                # 判断器挂了 → fail-open:不把好结果变坏
                final_output = msg["content"]
                tracer.set_final_output(final_output, success=True)
                tracer.save()
                clear_tracer()
                print(f"⚠️ 完成条件判断器异常,按模型回答返回:{e}")
                return final_output
            tracer.add_event("goal_status", decision)

            if decision["ok"]:
                final_output = msg["content"]
                tracer.set_final_output(final_output, success=True)
                tracer.save()
                clear_tracer()
                print(f"✅ 目标达成:{decision['reason']}")
                return final_output
            if decision["impossible"]:
                final_output = f"【目标无法完成】{decision['reason']}\n{msg.get('content') or ''}"
                tracer.set_final_output(final_output, success=False)
                tracer.save()
                clear_tracer()
                print(f"❌ 目标无法完成:{decision['reason']}")
                return final_output

            consecutive_blocks += 1
            if consecutive_blocks >= GOAL_BLOCK_CAP:
                final_output = f"【目标未达成,连续判断{GOAL_BLOCK_CAP}轮】{decision['reason']}\n{msg.get('content') or ''}"
                tracer.set_final_output(final_output, success=False)
                tracer.save()
                clear_tracer()
                print(f"❌ 连续{GOAL_BLOCK_CAP}轮未达成,停止:{decision['reason']}")
                return final_output

            # 未达成 → 注入判断器反馈,继续下一轮
            print(f"⏳ 目标未达成:{decision['reason']} → 继续")
            messages.append({
                "role": "user",
                "content": (
                    "[目标仍未完成]\n"
                    f"条件: {completion_condition}\n"
                    f"判断器: {decision['reason']}\n"
                    "请继续工作并给出缺失的证据。"
                ),
            })
            continue

        # ===== 有工具调用 → 执行 =====
        tool_obs_list = []
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            print(f"调用工具:{name} 参数:{args}")
            observation = execute_tool(name, args, memory_session)
            memory_session.add_tool_call(name)
            print(f"观测:{observation[:300]}")
            tracer.add_event(
                "tool_call",
                {"tool_name": name, "arguments": args, "observation": observation[:500]},
            )
            tool_obs_list.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": observation,
            })

        messages.append(msg)
        messages.extend(tool_obs_list)
        consecutive_blocks = 0  # 模型干了活 → 重置「连续未达成」计数

    # ===== 强制终止 =====
    print(f"\n===== 强制终止:到达最大迭代 {MAX_ITER} 次 =====")
    final_resp = llm_chat(messages)["choices"][0]["message"]["content"]
    final_output = f"【达到最大迭代次数 {MAX_ITER},强制终止】\n{final_resp}"
    tracer.set_final_output(final_output, success=False)
    tracer.save()
    clear_tracer()
    memory_session.add_short_msg("assistant", final_output)
    return final_output


if __name__ == "__main__":
    session = create_memory_session("session_demo")
    question = "介绍一下当前语料的主要内容"
    print(agent_loop(question, memory_session=session))
