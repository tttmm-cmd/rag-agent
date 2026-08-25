"""冒烟测试:文档解析 + 分块(纯逻辑,不依赖 LLM/向量化)"""
import os

from rag.parse_document import ParsedBlock, parse_document
from rag.chunker import split_into_chunks, _slide


def _write_tmp_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text(
        "第一条:供应商需具备ISO9001质量管理体系认证。\n"
        "第二条:投标截止时间为2026年9月30日17时。\n",
        encoding="utf-8",
    )
    return str(p)


def test_parse_txt(tmp_path):
    blocks = parse_document(_write_tmp_txt(tmp_path))
    assert len(blocks) == 1
    assert blocks[0].section == "全文"
    assert "ISO9001" in blocks[0].text


def test_chunker_small_text_single_chunk(tmp_path):
    blocks = parse_document(_write_tmp_txt(tmp_path))
    chunks = split_into_chunks(blocks)
    assert len(chunks) == 1  # 小文本 → 单块
    assert chunks[0]["source"] == blocks[0].source


def test_chunker_long_text_splits_and_keeps_section():
    blocks = [ParsedBlock("x.txt", "第1页", "段落内容。" * 300)]
    chunks = split_into_chunks(blocks)
    assert len(chunks) > 1
    assert all(c["section"] == "第1页" for c in chunks)


def test_slide_overlap():
    pieces = _slide("abcdefghij", size=4, overlap=1)
    assert pieces == ["abcd", "defg", "ghij"]


def test_chunker_long_paragraph_slide():
    # 单个超长段落(无换行)应走滑动窗口,不丢内容
    text = "词" * 1200
    blocks = [ParsedBlock("y.txt", "正文", text)]
    chunks = split_into_chunks(blocks)
    assert len(chunks) > 1
    assert sum(len(c["text"]) for c in chunks) >= len(text)
