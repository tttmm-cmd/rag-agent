"""文本分块:段落优先合并,超长段落滑动窗口硬切,带 overlap 防切碎语义"""
from rag.parse_document import ParsedBlock
from rag.text_norm import normalize_text

CHUNK_SIZE = 500  # 块目标字符数(中文约 250-300 token)
OVERLAP = 50      # 相邻块重叠字符


def split_into_chunks(blocks: list[ParsedBlock], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    chunks = []
    for b in blocks:
        # 入库前 NFKC 归一化:PDF 提取的康熙部首变体(机器⼈)统一成标准字符(机器人)
        for text in _split_one(normalize_text(b.text), size, overlap):
            chunks.append({"source": b.source, "section": b.section, "text": text})
    return chunks


def _split_one(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text]
    pieces, current = [], ""
    for p in paragraphs:
        if len(p) > size:                     # 超长段落:滑动窗口硬切
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_slide(p, size, overlap))
        elif len(current) + len(p) + 1 <= size:  # 还能塞进当前块
            current += p + "\n"
        else:                                   # 当前块满了,另起一块
            pieces.append(current.strip())
            current = p + "\n"
    if current:
        pieces.append(current.strip())
    return [p for p in pieces if p]


def _slide(text: str, size: int, overlap: int) -> list[str]:
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return out
