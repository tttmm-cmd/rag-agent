"""评测执行:检索召回率 / 答案准确率 / 引用正确率 + 时延

两类指标对应两类验收:
- retrieval_recall: 只测 RAG 管道(不调 LLM),期望关键词是否在召回块里
- answer_accuracy:  跑完整 agent,答案关键词命中 + 是否带引用(归因闭环)
"""
import re
import time

from rag.store import rag_store


def retrieval_recall(cases, corpus, top_k=10, query_fn=None) -> float:
    """期望关键词出现在召回块中 → 召回率

    query_fn: 可选的 query 预处理(如剥项目名)。系统实际检索的就是预处理后的
    内容关键词,指标应测同一查询;query_fn 返回 (query, 项目) 时还会按项目过滤,
    与 direct_answer 的"定位项目内检索"行为一致。
    """
    from rag.direct import _in_project
    total = hit = 0
    for c in cases:
        if c[0] != corpus:
            continue
        total += 1
        _q, kws = c[1], c[2]
        proj = None
        q = _q
        if query_fn:
            r = query_fn(_q)
            if isinstance(r, tuple):
                q, proj = r[0], r[1]
            else:
                q = r
        # 双 query:向量吃整句含项目名(消歧),BM25 吃清洗关键词(防封面绑架),与 direct_answer 一致
        # project_filter:已知项目时直接在项目子集内融合,防别项目共词块挤掉目标块
        pf = (lambda s: _in_project(s, proj)) if proj else None
        hits = rag_store.search(_q, top_k=top_k, bm25_query=q, project_filter=pf)
        joined = "".join(h["text"] for h in hits)
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
        # 引用判定与 main._stalling 同一口径:agent 实际输出 [id=文件#块号],
        # 只看"出处/引用/引文"会把真实引用误判为缺失 → 引用正确率假 0%
        cited = bool(re.search(r"#\d+", answer)) or any(
            t in answer for t in ("出处", "引用", "引文", "id=", "cite_source"))
        rows.append({
            "question": _q, "kind": kind,
            "accuracy": ok, "cited": cited,
            "latency_s": round(latency, 2), "answer": answer[:120],
        })
    return rows
