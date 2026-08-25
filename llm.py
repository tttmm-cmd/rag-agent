"""
LLM 请求封装:OpenAI 兼容 /chat/completions

从 main.py 抽出来独立成模块,让 tool_system(summarize 工具)也能复用,避免循环导入。
**kwargs 兼容 week3 goal_evaluator 传 include_tools=False(本设计 tools 默认 None 即不带工具)。
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")

if not LLM_API_KEY or not LLM_BASE_URL:
    raise RuntimeError("请检查 .env: 缺失 LLM_API_KEY / LLM_BASE_URL 配置!")


def llm_chat(messages: list, tools: list | None = None, **kwargs) -> dict:
    """调 LLM。tools 传本地工具 schema 时自动走 function calling。"""
    payload = {"model": LLM_MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"LLM 接口请求失败,状态码:{resp.status_code},响应:{resp.text[:300]}")
    resp_json = resp.json()
    if "choices" not in resp_json:
        raise RuntimeError(f"接口返回无 choices 字段:{resp_json}")
    return resp_json
