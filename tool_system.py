"""
工具系统:注册表核心 + 4 个记忆工具(week3 原样)+ 5 个文档问答工具

架构:ToolRegistry 查表分发,与 agent_loop 解耦。
MCP 扩展点:未来接外部工具,只需在 execute_tool 入口加一个外部工具源判断。
"""
import os

from rag.parse_document import parse_document
from rag.store import rag_store

MAX_OUTPUT_LEN = 2000
DATA_DIR = "./data"


def truncate_text(text: str, max_len: int = MAX_OUTPUT_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n...【已截断,原文共{len(text)}字,仅保留前{max_len}字】"


# ====================== 1. 注册表核心 ======================
class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, name, description, parameters, func, category="common"):
        self._tools[name] = {
            "schema": {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            },
            "func": func,
            "category": category,
        }

    def get_all_schemas(self) -> list:
        return [t["schema"] for t in self._tools.values()]

    def get_tool_func(self, name):
        return self._tools[name]["func"] if name in self._tools else None


tool_registry = ToolRegistry()


# ====================== 2. 记忆工具(week3 原样) ======================
def tool_memory_kv_write(memory_session, key: str, value: str, desc: str = "") -> str:
    memory_session.add_kv(key, value, desc)
    return f"【KV记忆写入成功】key={key}, value={value}"


def tool_memory_kv_get(memory_session, target_key: str) -> str:
    val = memory_session.get_kv(target_key)
    if val is None:
        return f"【KV查询结果】不存在key={target_key}"
    return f"【KV查询结果】key={target_key},value={val}"


def tool_memory_sem_add(memory_session, content: str, desc: str = "") -> str:
    memory_session.add_semantic(content, desc)
    return "【语义记忆保存成功】"


def tool_memory_sem_search(memory_session, keyword: str) -> str:
    return memory_session.search_semantic(keyword)


MEMORY_TOOLS = {"memory_kv_write", "memory_kv_get", "memory_sem_add", "memory_sem_search"}


# ====================== 3. 文档问答工具 ======================
def _corpus_error() -> str | None:
    return None if rag_store.index is not None else "❌ 语料未加载,请通过接口指定 corpus(如 ?corpus=zhizheng)"


def tool_parse_document(file_name: str) -> str:
    """解析 data/ 目录下文档(路径安全:强制限制在 data 内)"""
    path = os.path.abspath(os.path.join(DATA_DIR, file_name))
    if not path.startswith(os.path.abspath(DATA_DIR)):
        return f"❌ 路径越界拦截:{file_name} 超出 data/ 目录"
    if not os.path.exists(path):
        return f"❌ 文件不存在:{file_name}(相对于 data/ 目录)"
    blocks = parse_document(path)
    if not blocks:
        return "❌ 解析失败或无文字内容(扫描件需 OCR)"
    text = "\n".join(f"[{b.section}] {b.text}" for b in blocks)
    return f"【解析成功】{file_name} 共{len(blocks)}段\n{truncate_text(text)}"


def tool_retrieve(query: str, top_k: int = 5) -> str:
    err = _corpus_error()
    if err:
        return err
    hits = rag_store.search(query, top_k=top_k)
    if not hits:
        return "❌ 未检索到相关内容,建议换关键词再试"
    lines = [
        f"[id={h['id']}] [{h['source']} | {h['section']} | score={h.get('final_score', h['score']):.3f}]\n{h['text']}"
        for h in hits
    ]
    return "【检索结果 top-k,回答时用 cite_source 引用 id】\n\n" + truncate_text("\n\n".join(lines))


def tool_query_table(file_name: str, keyword: str = "") -> str:
    """从 data/ 下 Excel 抽取数据行,可选按关键词过滤,返回结构化文本供模型推理"""
    if not file_name.lower().endswith(".xlsx"):
        return "❌ 仅支持 data/ 下的 .xlsx 文件"
    path = os.path.abspath(os.path.join(DATA_DIR, file_name))
    if not os.path.exists(path):
        return f"❌ 文件不存在:{file_name}(相对于 data/ 目录)"
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    lines = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells and (not keyword or keyword.lower() in " | ".join(cells).lower()):
                lines.append(" | ".join(cells))
    wb.close()
    if not lines:
        return f"❌ 未找到包含「{keyword}」的行"
    return f"【表格数据 {file_name} 共匹配 {len(lines)} 行】\n" + truncate_text("\n".join(lines))


def tool_summarize(text: str) -> str:
    """对一段文本生成摘要(调 LLM),压缩上下文后返回"""
    from llm import llm_chat
    msgs = [
        {"role": "system", "content": "你是摘要助手,用不超过150字概括核心内容,输出纯文本。"},
        {"role": "user", "content": text[:4000]},
    ]
    try:
        resp = llm_chat(msgs)
        return "【摘要】" + resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ 摘要失败:{e}"


def tool_cite_source(chunk_ids: str) -> str:
    """按检索结果的 id 生成规范引用(文件+位置+引文),保证答案可溯源"""
    err = _corpus_error()
    if err:
        return err
    ids = [i.strip() for i in chunk_ids.replace("，", ",").split(",") if i.strip()]
    if not ids:
        return "❌ 请传入检索结果里的 id,用逗号分隔"
    by_id = {c.get("id"): c for c in rag_store.chunks}
    lines = []
    for cid in ids:
        c = by_id.get(cid)
        lines.append(f"[{cid}] 出处:{c['source']} {c['section']}\n引文:{c['text'][:200]}" if c else f"[{cid}] ❌ 未找到该块")
    return "【引用来源】\n\n" + "\n\n".join(lines)


# ====================== 4. 注册所有工具 ======================
tool_registry.register(
    "memory_kv_write", "写入长期KV键值记忆,存储文件名/参数等确定信息",
    {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}, "desc": {"type": "string"}},
    tool_memory_kv_write, "memory",
)
tool_registry.register(
    "memory_kv_get", "根据key精确查询长期KV记忆",
    {"type": "object", "properties": {"target_key": {"type": "string"}}},
    tool_memory_kv_get, "memory",
)
tool_registry.register(
    "memory_sem_add", "存入长期语义记忆,保存长文本/任务过程",
    {"type": "object", "properties": {"content": {"type": "string"}, "desc": {"type": "string"}}},
    tool_memory_sem_add, "memory",
)
tool_registry.register(
    "memory_sem_search", "模糊检索语义记忆,按关键词匹配历史文本",
    {"type": "object", "properties": {"keyword": {"type": "string"}}},
    tool_memory_sem_search, "memory",
)
tool_registry.register(
    "parse_document", "解析 data/ 目录下的 PDF/Word/Excel 文档,返回带来源定位的文本(页码/工作表)",
    {"type": "object", "properties": {"file_name": {"type": "string", "description": "data/ 下的相对路径,如 zhizheng/xx.pdf"}}},
    tool_parse_document, "doc",
)
tool_registry.register(
    "retrieve", "在当前语料中向量检索相关内容,返回带 id 的 top-k 结果(含来源和分数)",
    {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}},
    tool_retrieve, "rag",
)
tool_registry.register(
    "query_table", "从 data/ 下的 Excel 抽取数据行,可选按关键词过滤,供数据智能分析",
    {"type": "object", "properties": {"file_name": {"type": "string"}, "keyword": {"type": "string", "description": "可选,过滤包含该关键词的行"}}},
    tool_query_table, "table",
)
tool_registry.register(
    "summarize", "对一段文本生成摘要,压缩上下文",
    {"type": "object", "properties": {"text": {"type": "string"}}},
    tool_summarize, "common",
)
tool_registry.register(
    "cite_source", "按检索结果的 id 生成规范引用(文件+位置+引文),回答必须用它附出处",
    {"type": "object", "properties": {"chunk_ids": {"type": "string", "description": "id 列表,逗号分隔"}}},
    tool_cite_source, "rag",
)


def execute_tool(name: str, args: dict, memory_session=None) -> str:
    """统一工具分发(MCP 扩展点:未来接外部工具,在此加外部工具源判断)"""
    func = tool_registry.get_tool_func(name)
    if func is None:
        return f"❌ 未知工具: {name}"
    if name in MEMORY_TOOLS:
        return func(memory_session, **args)
    return func(**args)
