"""关键词检索:jieba 分词 + 手写 BM25。与 FAISS 向量组成双路召回,由 retrieve_hybrid 做 RRF 融合。

BM25(Okapi):
  score(d,q) = Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
  IDF(t) = ln((N − df + 0.5) / (df + 0.5) + 1)
  k1=1.5, b=0.75 为标准默认值

为什么手写而不是 rank_bm25 库:几十行,面试能讲清每个符号;停用词可控。
"""
import math

import jieba

jieba.setLogLevel(20)  # 静默首次加载的 "Building prefix dict" 日志

K1 = 1.5
B = 0.75
STOP = set("的了是在和与及等对给为有个中里于而之其或从被把将并")


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip() and w not in STOP]


class BM25:
    """一份语料的 BM25 索引(语料切换时重建,只在第一次检索时构建)"""

    def __init__(self, texts: list[str]):
        self.doc_tokens = [_tokenize(t) for t in texts]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.n = len(self.doc_tokens)
        self.avgdl = sum(self.doc_lens) / max(self.n, 1)
        self.df: dict[str, int] = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def scores(self, query: str) -> list[float]:
        """每个 chunk 的 BM25 分数(未命中任何词 = 0)"""
        q_tokens = [t for t in _tokenize(query) if t in self.df]
        if not q_tokens or self.n == 0:
            return [0.0] * self.n
        out = [0.0] * self.n
        for t in q_tokens:
            df = self.df[t]
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(t)
                if not tf:
                    continue
                norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.doc_lens[i] / self.avgdl))
                out[i] += idf * norm
        return out


# 语料级缓存:换语料后 chunks 是新的 list → 引用比较重建;强引用防 id 复用误命中
_bm25 = None
_bm25_chunks = None


def bm25_for(chunks: list[dict]) -> BM25:
    global _bm25, _bm25_chunks
    if _bm25 is None or _bm25_chunks is not chunks:
        _bm25 = BM25([c["text"] for c in chunks])
        _bm25_chunks = chunks
    return _bm25
