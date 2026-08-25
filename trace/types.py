from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from datetime import datetime
import json

@dataclass
class TraceEvent:
    """Trace 中的单个事件"""
    step: int                    # 步骤序号
    type: str                    # "user_input" | "llm_decision" | "tool_call" | "tool_result" | "final_output"
    timestamp: str               # 时间戳
    data: Dict[str, Any]         # 事件数据
    duration_ms: float = 0.0     # 该步骤耗时（毫秒）
    token_count: int = 0         # 该步骤消耗的 token

    def to_dict(self):
        return {
            "step": self.step,
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count
        }

@dataclass
class Trace:
    """完整执行轨迹"""
    trace_id: str                # 唯一标识
    session_id: str              # 会话ID
    start_time: str              # 开始时间
    end_time: str                # 结束时间
    user_input: str              # 用户输入
    max_iterations: int          # 最大迭代次数
    actual_iterations: int       # 实际迭代次数
    total_tokens: int = 0        # 总 Token 消耗
    total_duration_ms: float = 0.0  # 总耗时
    events: List[TraceEvent] = field(default_factory=list)  # 事件列表
    final_output: str = ""       # 最终输出
    success: bool = False        # 是否成功
    error: Optional[str] = None  # 错误信息

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "user_input": self.user_input,
            "max_iterations": self.max_iterations,
            "actual_iterations": self.actual_iterations,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "events": [e.to_dict() for e in self.events],
            "final_output": self.final_output,
            "success": self.success,
            "error": self.error
        }

    def to_json(self) -> str:
        """输出为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_jsonl(self) -> str:
        """输出为 JSONL 格式（每行一个事件）"""
        lines = []
        for event in self.events:
            line = {
                "trace_id": self.trace_id,
                "session_id": self.session_id,
                "step": event.step,
                "type": event.type,
                "timestamp": event.timestamp,
                "data": event.data,
                "duration_ms": event.duration_ms,
                "token_count": event.token_count
            }
            lines.append(json.dumps(line, ensure_ascii=False))
        return "\n".join(lines)