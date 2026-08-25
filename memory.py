import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# ===================== 基础配置 =====================
# 记忆持久化存储目录
MEMORY_SAVE_DIR = "./agent_memory"
# 会话文件命名模板
SESSION_FILE_TPL = os.path.join(MEMORY_SAVE_DIR, "session_{sid}.json")
# 初始化存储文件夹
os.makedirs(MEMORY_SAVE_DIR, exist_ok=True)

# ===================== 数据结构定义 =====================
@dataclass
class KVRecord:
    """KV长期键值记忆：存储确定静态信息（姓名、配置、固定参数）"""
    key: str
    value: str
    desc: str  # 这条记忆的来源说明，用于溯源

@dataclass
class SemanticRecord:
    """语义长期记忆：存储长文本、过程记录，支持模糊检索"""
    content: str
    desc: str
    embed_cache: Optional[list[float]] = None  # 向量缓存，本demo暂不实现向量计算

@dataclass
class SessionMemory:
    """单会话完整内存容器"""
    session_id: str
    # 1. 短期记忆：当前对话messages（传给LLM上下文）
    short_messages: List[Dict[str, str]]
    # 2. 长期KV记忆：确定性键值对
    kv_memory: List[KVRecord]
    # 3. 长期语义记忆：过程、文档、长文本记录
    semantic_memory: List[SemanticRecord]

# ===================== 会话内存管理器 =====================
class MemoryManager:
    def __init__(self, session_id: str):
        self.sid = session_id
        self.session_path = SESSION_FILE_TPL.format(sid=session_id)
        self.memory: SessionMemory = self._load_session()
        # 新增：记录本次会话调用的工具名称列表
        self.tools_called = []

    def add_tool_call(self, tool_name: str):
        """记录一次工具调用"""
        self.tools_called.append(tool_name)

    def get_tools_called(self) -> List[str]:
        """获取本次会话已调用的工具列表"""
        return self.tools_called

    def clear_tools_called(self):
        """清空工具调用记录（用于会话重置）"""
        self.tools_called = []

    def _load_session(self) -> SessionMemory:
        """从本地json加载会话，文件不存在返回空内存"""
        if not os.path.exists(self.session_path):
            return SessionMemory(
                session_id=self.sid,
                short_messages=[],
                kv_memory=[],
                semantic_memory=[]
            )
        with open(self.session_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 反序列化
        kv_list = [KVRecord(**item) for item in raw["kv_memory"]]
        sem_list = [SemanticRecord(**item) for item in raw["semantic_memory"]]
        return SessionMemory(
            session_id=raw["session_id"],
            short_messages=raw["short_messages"],
            kv_memory=kv_list,
            semantic_memory=sem_list
        )

    def save(self):
        """持久化保存当前会话到本地文件"""
        kv_dump = [asdict(item) for item in self.memory.kv_memory]
        sem_dump = [asdict(item) for item in self.memory.semantic_memory]
        dump_data = {
            "session_id": self.sid,
            "short_messages": self.memory.short_messages,
            "kv_memory": kv_dump,
            "semantic_memory": sem_dump
        }
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, ensure_ascii=False, indent=2)

    # ---------------- 短期对话内存操作 ----------------
    def add_short_msg(self, role: str, content: str):
        """追加一条短期对话消息"""
        self.memory.short_messages.append({"role": role, "content": content})
        self.save()

    def get_short_messages(self) -> List[Dict[str, str]]:
        """获取完整短期上下文（直接传给LLM）"""
        return self.memory.short_messages

    def clear_short(self):
        """清空短期对话，保留长期记忆"""
        self.memory.short_messages = []
        self.save()

    # ---------------- KV长期键值记忆操作 ----------------
    def add_kv(self, key: str, value: str, desc: str = "用户对话提取"):
        """新增一条KV静态记忆"""
        new_rec = KVRecord(key=key, value=value, desc=desc)
        self.memory.kv_memory.append(new_rec)
        self.save()

    def get_kv(self, target_key: str) -> Optional[str]:
        """根据key精确查询KV"""
        for item in self.memory.kv_memory:
            if item.key == target_key:
                return item.value
        return None

    def list_all_kv(self) -> str:
        """批量查看全部KV记忆，给模型读取"""
        if len(self.memory.kv_memory) == 0:
            return "暂无长期KV记忆"
        out = "【长期KV记忆列表】\n"
        for item in self.memory.kv_memory:
            out += f"key:{item.key} | value:{item.value} | 来源:{item.desc}\n"
        return out

    # ---------------- 语义长期记忆操作 ----------------
    def add_semantic(self, content: str, desc: str = "任务过程记录"):
        """新增一段语义记忆"""
        new_rec = SemanticRecord(content=content, desc=desc)
        self.memory.semantic_memory.append(new_rec)
        self.save()

    def search_semantic(self, keyword: str) -> str:
        """简单文本模糊检索语义记忆（向量后续拓展）"""
        res = []
        for item in self.memory.semantic_memory:
            if keyword.lower() in item.content.lower():
                res.append(f"来源:{item.desc}\n内容:{item.content}")
        if not res:
            return f"语义记忆中未检索到包含「{keyword}」的内容"
        return "【语义检索结果】\n" + "\n=====\n".join(res)

    def list_all_semantic(self) -> str:
        """查看全部语义记忆"""
        if len(self.memory.semantic_memory) == 0:
            return "暂无长期语义记忆"
        out = "【长期语义记忆列表】\n"
        for idx, item in enumerate(self.memory.semantic_memory, 1):
            out += f"{idx}. 描述:{item.desc}\n片段:{item.content[:200]}...\n"
        return out

# ===================== 全局工具：创建会话实例 =====================
def create_memory_session(session_id: str) -> MemoryManager:
    """对外暴露入口，新建/加载会话内存"""
    return MemoryManager(session_id)