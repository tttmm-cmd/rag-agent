"""评测执行:检索召回率 / 答案准确率 / 引用正确率 + 时延

两类指标对应两类验收:
- retrieval_recall: 只测 RAG 管道(不调 LLM),期望关键词是否在召回块里
- answer_accuracy:  跑完整 agent,答案关键词命中 + 是否带引用(归因闭环)
"""
import time

from rag.store import rag_store


def retrieval_recall(cases, corpus, top_k=5) -> float:
    """期望关键词出现在召回块中 → 召回率"""
    total = hit = 0
    for c in cases:
        if c[0] != corpus:
            continue
        total += 1
        _q, kws = c[1], c[2]
        joined = "".join(h["text"] for h in rag_store.search(_q, top_k=top_k))
        if not kws or all(k in joined for k in kws):
            hit += 1
    return hit / total if total else 1.0


def answer_accuracy(cases, corpus, run_agent) -> list[dict]:
    """跑完整 agent,记录每条的准确/引用/时延"""
    rows = []
    for c in cases:
        if c[0] != corpus:
            continue
        _q, kws, kind = c[1], c[2], c[3]
        t0 = time.time()
        answer = run_agent(_q)
        latency = time.time() - t0
        ok = not kws or all(k in answer for k in kws)
        cited = any(tag in answer for tag in ("出处", "引用", "引文"))
        rows.append({
            "question": _q, "kind": kind,
            "accuracy": ok, "cited": cited,
            "latency_s": round(latency, 2), "answer": answer[:120],
        })
    return rows
