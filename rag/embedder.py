"""向量化:DashScope qwen3.7-text-embedding(OpenAI 兼容端点),密钥/模型可换"""
import os
from functools import lru_cache

import requests
from dotenv import load_dotenv

load_dotenv()  # index.py 进程也要读 .env,必须在本模块加载

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3.7-text-embedding")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BATCH = 10  # DashScope 兼容端点批量上限 10


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not EMBED_API_KEY:
        raise RuntimeError("缺少向量化密钥:请在 .env 配置 DASHSCOPE_API_KEY(或 EMBED_API_KEY)")
    all_emb = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = requests.post(
            EMBED_BASE_URL,
            headers={"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": batch},
            timeout=30,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        all_emb.extend(d["embedding"] for d in data)
    return all_emb


@lru_cache(maxsize=128)
def _embed_query_cached(text: str) -> tuple:
    """query 向量缓存:agent 多轮里反复检索同一 query 时,省 DashScope 往返。
    tuple 不可变,防调用方误改污染缓存;embed_query 返回时再转 list。
    建索引走的 embed_texts(批量)不经过这里,不受缓存影响。"""
    return tuple(embed_texts([text])[0])


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text))
