"""语料加载单例:按 corpus 名加载索引,工具与 FastAPI 共用

换语料 = 换名字:index.py --corpus techdocs 重建,rag_store.load("techdocs") 切换。
"""
import os
from rag.vector_store import load_index
from rag.retrieve import retrieve, rerank_lexical


class RAGStore:
    def __init__(self):
        self.corpus = None
        self.index = None
        self.chunks: list[dict] = []

    def load(self, corpus: str, root: str = "."):
        prefix = os.path.join(root, f"faiss_{corpus}")
        if not os.path.exists(prefix + ".index"):
            raise FileNotFoundError(f"语料 [{corpus}] 未建索引,先跑: python index.py --corpus {corpus}")
        self.index, self.chunks = load_index(prefix)
        self.corpus = corpus
        print(f"✅ 语料 [{corpus}] 已加载: {self.index.ntotal} 条索引")

    def search(self, query: str, top_k: int = 5, rerank: bool = True) -> list[dict]:
        hits = retrieve(query, self.index, self.chunks, top_k=top_k)
        return rerank_lexical(hits, query) if rerank else hits


# 全局单例:index.py / app.py / 工具都读写它
rag_store = RAGStore()
