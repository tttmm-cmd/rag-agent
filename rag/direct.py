"""确定性兜底回答:agent 循环答不出来时,直接「定位项目 → 检索/读原文 → 一次生成答案」。

不依赖模型多步工具调用的稳定性:弱模型(如 deepseek)常在 8 步内迷失关键词,
而资格清单等条文常埋在向量检索 top-k 之外,只能靠 parse_document(keyword) 读原文。
demo 题必须能答,所以把这条可靠路径做成 agent 循环的 safety-net。
"""
import os
import re

from rag.store import rag_store
from rag.parse_document import parse_document
from llm import llm_chat

DATA_DIR = "./data"

# 问题里的"内容焦点" → 读原文时的目标关键词
FOCUS_KEYWORD = {
    "资格": "资格要求",
    "资质": "资格要求",
    "审查": "资格审查",
    "截止": "投标截止时间",
    "开标": "开标时间",
    "采购内容": "采购内容",
    "报价": "报价",
    "付款": "付款方式",
    "质保": "质保",
}


def _clean_project_name(name: str) -> str:
    """把文件夹名/文件名洗干净 → 项目候选词(去序号/「采购文件-」前缀/「（SDGP…）」/「-文件集」)"""
    name = re.sub(r"^\d+[\.\-、]?\s*", "", name)
    while True:  # 循环剥「采购文件-」「磋商文件-」等前缀
        m = re.match(r"^(采购文件|磋商文件|招标文件|需求文件|公开招标文件|竞争性磋商文件)[\-—]?", name)
        if not m:
            break
        name = name[m.end():]
    name = name.split("（")[0]
    name = re.sub(r"[\-—]?文件集$", "", name)
    name = re.sub(r"[\-—]?SDGP.*$", "", name)
    return name.strip()


def _project_candidates() -> list:
    """语料里所有项目候选词:来自项目文件夹名 + 直接躺在 data/ 下的文件名
    (如「采购文件-山东电子…基地-….pdf」无文件夹,得从文件名提炼)"""
    key = rag_store.corpus
    _cache = getattr(_project_candidates, "_cache", None)
    if _cache and _cache[0] == key:
        return _cache[1]
    cands = set()
    for c in rag_store.chunks:
        parts = c.get("source", "").replace("\\", "/").split("/")
        name = parts[1] if len(parts) >= 3 else os.path.splitext(parts[-1])[0]
        name = _clean_project_name(name)
        if len(name) >= 6:
            cands.add(name)
    _project_candidates._cache = (key, sorted(cands, key=len, reverse=True))
    return _project_candidates._cache[1]


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


def match_project(question: str) -> str | None:
    """问题提到哪个项目(最长候选优先;子串命中或公共前缀≥6都算)
    例:「…机器人项目」vs「…机器人产教融合基地」靠公共前缀命中。"""
    for cand in _project_candidates():
        if cand in question:
            return cand
        if len(_common_prefix(cand, question)) >= 6:
            return cand
    return None


def _strip_project(question: str, proj: str | None) -> str:
    if not proj:
        return question
    q = question.replace(proj, "")
    if q != question:
        return q
    # 候选词常比问题里的表述长(公共前缀命中,如「…机器人产教融合基地」vs「…机器人项目」),剥公共前缀
    cp = _common_prefix(proj, question)
    if len(cp) >= 6:
        return question.replace(cp, "", 1)
    return question


def _clean_fillers(q: str) -> str:
    """剥衬词,留内容实词(给 BM25 用)。长词在前:先"是什么时候"再"什么",否则留"时候"残渣"""
    for w in ("请给出", "请告诉我", "请", "具体条文是什么", "具体条文", "和出处",
              "是什么时候", "什么时候", "是不是", "是什么", "有什么", "多少",
              "哪些", "什么", "怎么样", "如何", "呢", "吗", "了", "吧",
              "采购项目", "项目", "对供应商", "供应商需要", "供应商", "需要",
              "采购了", "采购内容", "?", "？", "。", "，", ",", "的", "是"):
        q = q.replace(w, "")
    return " ".join(q.split())


def _content_query(question: str) -> tuple[str, str | None]:
    """去掉项目名和问题衬词,留内容关键词(如「供应商资格要求」)"""
    proj = match_project(question)
    return _clean_fillers(_strip_project(question, proj)), proj


def _in_project(source: str, proj: str) -> bool:
    """source 路径是否属于该项目(候选词从真实路径提炼,子串即可命中)"""
    src = source.replace("\\", "/")
    if proj in src:
        return True
    # 兜底:路径段以 proj 开头(如文件夹名加了编号前缀)
    return any(seg.startswith(proj) for seg in src.split("/"))


def _format_hits(hits) -> str:
    lines = []
    for h in hits:
        lines.append(
            f"[id={h['id']}] [{h['source']} | {h['section']}]\n{h['text']}"
        )
    return "\n\n".join(lines[:6])


def _read_qualification_section(proj: str) -> list:
    """在目标项目内找 招标/磋商文件,读 keyword=「资格要求」的段落(带前后文)"""
    for c in rag_store.chunks:
        src = c.get("source", "")
        if _in_project(src, proj):
            for cand in ("资格性内容",):
                if cand in os.path.basename(src):
                    return []
    # 找该项目的主招标文件
    seen = set()
    for c in rag_store.chunks:
        src = c.get("source", "")
        if not _in_project(src, proj):
            continue
        fn = os.path.basename(src.replace("\\", "/"))  # Windows 索引里是反斜杠,Linux 上必须归一化
        if fn in seen:
            continue
        seen.add(fn)
        if ("招标文件" in fn or "磋商文件" in fn or "需求文件" in fn) and fn.endswith((".pdf", ".docx")):
            blocks = parse_document(os.path.join(DATA_DIR, src.replace("\\", "/")))
            idxs = [i for i, b in enumerate(blocks) if "资格要求" in b.text]
            if idxs:
                ctx = set()
                for i in idxs:
                    for j in range(max(0, i - 2), min(len(blocks), i + 3)):
                        ctx.add(j)
                return [f"[{blocks[i].section}] {blocks[i].text}" for i in sorted(ctx)]
    return []


def direct_answer(question: str, top_k: int = 6) -> str:
    """兜底回答:一次检索 + 必要时读原文,一次生成答案。返回纯文本。"""
    # 双 query:向量吃整句(含项目名,消歧定位),BM25 吃清洗后的内容关键词(防封面绑架)
    proj = match_project(question)
    clean, _ = _content_query(question)
    if len(clean) < 2:
        clean = question
    # 已知项目:直接在项目子集内融合(全局召回候选 → 项目过滤 → 项目内 RRF),
    # 避免「中小企业/特定资格要求」等跨项目共词把别项目块顶进 top-k、挤掉本项目块
    pf = (lambda s: _in_project(s, proj)) if proj else None
    hits = rag_store.search(question, top_k=top_k, bm25_query=clean, project_filter=pf)

    extra = ""
    if any(w in question for w in ("资格", "资质", "审查")) and proj:
        sec = _read_qualification_section(proj)
        if sec:
            extra = "\n\n=== 目标文件资格条文原文 ===\n" + "\n".join(sec[:6000])

    context = _format_hits(hits) + extra
    msgs = [
        {"role": "system", "content": "你是文档问答助手,基于语料片段回答,必须引用片段 id 附出处;片段里没有就说「语料中没有相关信息」,绝不编造。输出纯文本。"},
        {"role": "user", "content": f"用户问题:{question}\n\n检索到的语料片段:\n{context}\n\n请回答问题并附上引用的 id。"},
    ]
    try:
        resp = llm_chat(msgs)
        return resp["choices"][0]["message"].get("content") or "(兜底回答无输出)"
    except Exception as e:
        return f"(兜底回答失败:{e})"
