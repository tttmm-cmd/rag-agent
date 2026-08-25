"""向量化:DashScope text-embedding-v3(OpenAI 兼容端点),密钥/模型可换"""
import os
import requests

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
BATCH = 32


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


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
