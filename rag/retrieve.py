"""检索:向量 top-k + BM25 top-k → RRF 融合(向量+关键词混合检索)

向量召回保证语义,BM25 保证精确词命中(资格要求/TOPS/工期25天这类)。
RRF 只按排名融合——两路分数量纲不同(cosine≈0~1 vs BM25 无上限),
直接加权不可比,用 1/(k+rank) 求和,两路都命中的文档自然排前。
封面页(每文档第1块)降权。

面试点(海能达 JD"检索优化 + Re-ranking"):召回 → 混合 → 精排,
后续可升级为 cross-encoder 精排,接口不变。
"""
import numpy as np
import faiss
from rag.embedder import embed_query
from rag.keyword import bm25_for
from rag.text_norm import normalize_text

RRF_K = 60  # RRF 平滑常数(原论文标准值)


def retrieve(query: str, index, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """向量通道:FAISS 余弦相似度 top-k(query 先 NFKC 归一化,与入库文本一致)"""
    qvec = np.array([embed_query(normalize_text(query))], dtype="float32")
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, min(top_k, index.ntotal))
    hits = []
    for score, cid in zip(scores[0], ids[0]):
        if cid < 0:
            continue
        c = chunks[int(cid)]
        hits.append({"score": float(score), **c})
    return hits


def retrieve_hybrid(raw_query: str, index, chunks: list[dict], top_k: int = 5,
                    keyword_query: str | None = None,
                    project_filter=None) -> list[dict]:
    """双路召回(向量 + BM25)→ RRF 融合 → 封面(#0)降权 → 返回 top_k

    raw_query 走向量(embedding 吃整句,不清洗更准);
    keyword_query 走 BM25(精确词匹配,需先清洗;默认与 raw_query 相同)。

    project_filter: callable(source)->bool,给了则在项目子集内融合——
    先放大两路候选池,过滤到目标项目,再在项目内 RRF。修掉「中小企业/特定资格
    要求」这类跨项目共词时,别项目块全局命中更强、把目标项目块挤出 top-k 的问题。
    """
    kq = normalize_text(keyword_query or raw_query)
    if project_filter is None:
        vec_hits = retrieve(raw_query, index, chunks, top_k=top_k * 3)
    else:
        # 项目内融合:向量通道全局取大候选池(项目块不至于被别项目挤出),再过滤到项目内
        cand = min(index.ntotal, max(top_k * 20, 300))
        vec_hits = [h for h in retrieve(raw_query, index, chunks, top_k=cand)
                    if project_filter(h.get("source", ""))]
    vec_hits = vec_hits[:top_k * 3]

    scores = bm25_for(chunks).scores(kq)
    kw_hits = [{"score": scores[i], **chunks[i]} for i in range(len(scores)) if scores[i] > 0]
    if project_filter is not None:
        kw_hits = [h for h in kw_hits if project_filter(h.get("source", ""))]
    kw_hits.sort(key=lambda h: h["score"], reverse=True)
    kw_hits = kw_hits[:top_k * 3]

    # RRF:按名次给分,两路求和
    merged: dict = {}
    for hits in (vec_hits, kw_hits):
        for rank, h in enumerate(hits, start=1):
            item = merged.setdefault(h["id"], {"hit": h, "rrf": 0.0})
            item["rrf"] += 1.0 / (RRF_K + rank)
    ordered = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)

    out = []
    for item in ordered:
        h = item["hit"]
        final = item["rrf"]
        if str(h.get("id", "")).endswith("#0"):
            final *= 0.5
        h["final_score"] = final
        out.append(h)
    return out[:top_k]
