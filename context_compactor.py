import tiktoken
from typing import List, Dict, Tuple

# 全局编码适配Deepseek系列模型
TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_single_text(text: str) -> int:
    """统计单段文本token数量"""
    return len(TOKEN_ENCODER.encode(text))

def count_message_total(messages: List[Dict[str, str]]) -> int:
    """批量统计整条消息列表总token（包含role固定占位）"""
    total_tok = 0
    for msg in messages:
        role_tok = count_single_text(msg["role"])
        content_tok = count_single_text(msg["content"])
        total_tok += role_tok + content_tok + 4
    return total_tok

class ContextCompactor:
    def __init__(self, llm_api_func, token_threshold: int = 1800, keep_recent_rounds: int = 8):
        """
        上下文压缩器
        :param llm_api_func: 外部传入llm_chat请求函数
        :param token_threshold: 触发压缩的token上限
        :param keep_recent_rounds: 保留末尾多少轮原始交互不压缩
        """
        self.llm_chat = llm_api_func
        self.threshold = token_threshold
        self.keep_recent = keep_recent_rounds
        self.system_rule = ""

    def set_system_prompt(self, sys_text: str):
        """绑定全局系统提示，压缩时永久置顶不丢失"""
        self.system_rule = sys_text

    def is_need_compact(self, msg_list: List[Dict[str, str]]) -> bool:
        """判断当前上下文是否超出阈值需要压缩"""
        total = count_message_total(msg_list)
        return total > self.threshold

    def compact_messages(self, msg_list: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], int, int]:
        """
        执行上下文摘要压缩
        返回：新消息列表、压缩前token、压缩后token
        """
        pre_total = count_message_total(msg_list)
        sys_msg = None
        non_sys_msgs = []

        # 1. 分离 system 消息
        for m in msg_list:
            if m["role"] == "system":
                sys_msg = m
            else:
                non_sys_msgs.append(m)

        # 如果没有 system 消息，创建一个空的
        if sys_msg is None:
            sys_msg = {"role": "system", "content": ""}

        # 2. 从后往前找最近 keep_recent_rounds 个 user 消息
        user_count = 0
        recent_start_index = len(non_sys_msgs)  # 默认保留全部
        for i in range(len(non_sys_msgs) - 1, -1, -1):
            if non_sys_msgs[i]["role"] == "user":
                user_count += 1
                if user_count >= self.keep_recent:
                    recent_start_index = i
                    break

        # 3. 切分
        old_history_msgs = non_sys_msgs[:recent_start_index]
        recent_msgs = non_sys_msgs[recent_start_index:]

        # 4. 无老旧历史，无需压缩
        if len(old_history_msgs) == 0:
            return msg_list, pre_total, pre_total

        # 5. 请求 LLM 生成摘要
        concat_history = "\n".join([f"{m['role']}: {m['content']}" for m in old_history_msgs])
        summary_req = [
            {
                "role": "system",
                "content": "你是对话压缩助手，精简下面多轮历史，必须保留：读取过的文件名、KV存储键值、Todo任务进度，删除重复工具重试记录，摘要控制300token以内"
            },
            {
                "role": "user",
                "content": f"历史对话内容：\n{concat_history}"
            }
        ]
        summary_resp = self.llm_chat(summary_req)
        summary_content = summary_resp["choices"][0]["message"]["content"]

        # 6. 重组新消息列表
        new_msg_list = []
        # 原有 system（如果内容非空）
        if sys_msg["content"].strip():
            new_msg_list.append(sys_msg)
        # 摘要作为单独的 system 消息
        new_msg_list.append({
            "role": "system",
            "content": f"【久远对话精简摘要】{summary_content}"
        })
        new_msg_list.extend(recent_msgs)

        post_total = count_message_total(new_msg_list)
        return new_msg_list, pre_total, post_total