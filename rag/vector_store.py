"""向量库:FAISS 余弦(归一化后内积)+ 侧边元数据 JSON,按语料分文件保存"""
import json
import numpy as np
import faiss


def build_index(embeddings: list[list[float]]):
    mat = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(mat)  # 余弦相似度 → 内积
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


def save_index(index, chunks: list[dict], prefix: str):
    """prefix 形如 faiss_zhizheng → 生成 faiss_zhizheng.index + .json"""
    faiss.write_index(index, prefix + ".index")
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)


def load_index(prefix: str):
    index = faiss.read_index(prefix + ".index")
    with open(prefix + ".json", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
