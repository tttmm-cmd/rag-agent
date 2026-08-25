"""检索:查询向量化 → FAISS top-k → 轻量 Re-ranking(中文 bigram 重合加权)

面试点(海能达 JD"检索优化 + Re-ranking"):
向量召回保证语义,bigram 重合保证关键词精确命中,加权后排序。
后续可升级为 cross-encoder 精排,接口不变。
"""
import numpy as np
from rag.embedder import embed_query


def retrieve(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    qvec = np.array([embed_query(query)], dtype="float32")
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, min(top_k, index.ntotal))
    hits = []
    for score, cid in zip(scores[0], ids[0]):
        if cid < 0:
            continue
        c = chunks[int(cid)]
        hits.append({"score": float(score), **c})
    return hits


def rerank_lexical(hits: list[dict], query: str, alpha: float = 0.3) -> list[dict]:
    """向量分数 + 中文 bigram 重合度加权排序"""
    q_terms = set(_bigrams(query))
    for h in hits:
        overlap = sum(1 for t in _bigrams(h["text"]) if t in q_terms)
        lexical = min(overlap / max(len(q_terms), 1), 1)
        h["final_score"] = alpha * h["score"] + (1 - alpha) * lexical
    hits.sort(key=lambda h: h["final_score"], reverse=True)
    return hits


def _bigrams(text: str) -> list[str]:
    t = "".join(ch for ch in text if ch.strip() and ch not in " \t\n\r")
    return [t[i:i + 2] for i in range(len(t) - 1)]
