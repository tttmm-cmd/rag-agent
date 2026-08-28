"""
文档问答 Agent 主循环(复用 week3 agent_loop:压缩 / goal 判断器 / trace 全保留)
改动:去掉 MCP 与工具缓存;系统提示词改为"检索 → 带引用回答"
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

from llm import llm_chat
from tool_system import tool_registry, execute_tool
from memory import create_memory_session, MemoryManager
from context_compactor import ContextCompactor
from goal_evaluator import GoalEvaluator
from trace.tracer import Tracer, set_tracer, clear_tracer
from rag.store import rag_store
from rag.direct import direct_answer

MAX_ITER = int(os.getenv("MAX_ITER", 8))
GOAL_BLOCK_CAP = 3  # goal loop:连续未达成的最大轮数,防死循环

TOOL_INFO = tool_registry.get_all_schemas()


def _stalling(content: str) -> bool:
    """回答质量闸门:要么带出处(引用/出处/引文/块id),要么诚实说"语料中没有",
    否则视为空话(模型常输出"以上是…汇总""如需了解可进一步查询"这类指走内容的假回答)"""
    if not content or len(content.strip()) < 60:
        return True
    cited = any(t in content for t in ("出处", "引用", "引文", "id=", "cite_source")) \
        or re.search(r"#\d+", content) is not None
    honest = any(t in content for t in ("语料中", "不在语料", "未找到", "不存在", "无法从", "无法获取"))
    return not (cited or honest)

DEFAULT_PROMPT = """你是文档问答 Agent,基于已加载的政府采购语料回答业务问题。

检索要点:
- query 只放内容关键词(「资格要求」「供应商资格」「资格审查」「投标截止时间」「采购内容」「资质」等),绝不要把项目全名塞进 query——项目名只命中封面/邀请页,信息量低,还会把真正条文挤到后面;
- 检索结果里 source 字段自带文件路径(含项目文件夹名),回答时用 cite_source 引用它即可定位到正确项目——结果里看不到项目名是正常的,source 路径属于目标项目就对;
- 一次检索没命中就换同义词(资格/资质/审查/条件/条款),或把 top_k 调到 8~10 看更多;
- 回答「某项目有什么资格要求/条件」类问题:先 retrieve 纯内容关键词(如「供应商资格要求」,别带项目名),在结果里找到 source 含目标项目名的块确认文件名;再用 parse_document(该文件名, keyword=「资格要求」) 读该文件资格条文原文(带前后文),不要反复 retrieve;
- 找章节内容用 retrieve,不要用 parse_document 翻整篇(它只回前 2000 字);只有想核对某段原文时才用 parse_document 并传 keyword。

回答要点:
- 证据够了就立刻用纯文字回答,不要反复检索、不要调用记忆工具;
- 回答必须用 cite_source 引用检索到的 id 附出处;
- 检索不到就说「语料中没有相关信息」,绝不编造。"""


def agent_loop(user_query: str, memory_session: MemoryManager,
               completion_condition: str = "", system_prompt: str | None = None,
               corpus: str = "zhizheng") -> str:
    # ===== 确保语料已加载(命令行/测试入口常漏掉,工具侧 retrieve 会因此报错) =====
    if rag_store.corpus != corpus:
        rag_store.load(corpus)

    # ===== 初始化 Tracer =====
    tracer = Tracer(session_id=memory_session.sid, user_input=user_query, max_iterations=MAX_ITER)
    set_tracer(tracer)
    tracer.add_event("user_input", {"query": user_query})

    system_prompt = system_prompt or DEFAULT_PROMPT
    compactor = ContextCompactor(llm_chat, token_threshold=3000, keep_recent_rounds=8)
    compactor.set_system_prompt(system_prompt)

    messages = memory_session.get_short_messages()
    messages = [{"role": "system", "content": system_prompt}] + messages
    messages.append({"role": "user", "content": user_query})
    # 用户问题也要进短期记忆——否则下一轮 get_short_messages() 只有模型回复,
    # "这个项目"之类的指代无从解析(多轮记忆的 State 子系统缺陷)
    memory_session.add_short_msg("user", user_query)

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
                if _stalling(final_output):  # 空话/复述 → 走确定性兜底
                    print(f"⚠️ 模型回答为空话({final_output[:40]}…),改走确定性兜底")
                    final_output = direct_answer(user_query)
                    memory_session.add_short_msg("assistant", final_output)
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
    # 弱模型常在此空转 → 走确定性兜底(direct_answer 内部自带 LLM 一次生成)
    try:
        final_resp = direct_answer(user_query)
    except Exception as e:
        # 兜底也挂了 → 让模型硬答(不传 tools,避免回退成 XML 工具调用)
        final_msgs = messages + [
            {"role": "system", "content": "现在你必须直接给出最终文字答案。禁止调用任何工具,禁止输出 <tool_calls> 或任何 XML 标签,只输出纯文本。证据不足就如实说明。"},
            {"role": "user", "content": "根据已检索到的内容,直接回答用户的原始问题。"},
        ]
        try:
            final_resp = llm_chat(final_msgs, tools=[])["choices"][0]["message"].get("content") or "(模型未输出文字)"
        except Exception as e2:
            final_resp = f"(强制终止时总结失败:{e} / {e2})"
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
