"""
建索引 CLI: python index.py --corpus zhizheng [--root .]

同一套解析/分块/向量化,换语料目录重跑 → 生成 faiss_<corpus>.index + .json。
面试点(帆软"数据接入标准化"):管道与语料解耦,换数据源只改 --corpus。
"""
import argparse
import glob
import os

from rag.parse_document import parse_document
from rag.chunker import split_into_chunks
from rag.embedder import embed_texts
from rag.vector_store import build_index, save_index


def build(corpus: str, root: str = "."):
    data_dir = os.path.join(root, "data", corpus)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"❌ 语料目录不存在: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".pdf", ".docx", ".xlsx", ".txt"))]
    print(f"📂 语料 [{corpus}]: 发现 {len(files)} 个文档")

    all_chunks = []
    for f in files:
        blocks = parse_document(f)
        chunks = split_into_chunks(blocks)
        for i, c in enumerate(chunks):
            c["id"] = f"{os.path.basename(f)}#{i}"
            c["source"] = os.path.relpath(f, os.path.join(root, "data"))  # 引用显示相对路径
        all_chunks.extend(chunks)
        print(f"  {os.path.basename(f)}: {len(blocks)} 段 → {len(chunks)} 块")

    if not all_chunks:
        raise SystemExit("❌ 没有可解析的文档,先把文件放进 data/ 目录")

    print(f"共 {len(all_chunks)} 块,开始向量化...")
    embs = embed_texts([c["text"] for c in all_chunks])
    index = build_index(embs)
    prefix = os.path.join(root, f"faiss_{corpus}")
    save_index(index, all_chunks, prefix)
    print(f"✅ 索引已保存: {prefix}.index + .json ({index.ntotal} 条)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="按语料构建向量索引")
    ap.add_argument("--corpus", required=True, help="语料名,对应 data/<corpus>/ 目录")
    ap.add_argument("--root", default=".", help="项目根目录,默认当前目录")
    args = ap.parse_args()
    build(args.corpus, args.root)
