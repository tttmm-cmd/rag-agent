import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from .types import Trace, TraceEvent

class Tracer:
    """Trace 收集器 - 记录 Agent 执行的每一步"""

    def __init__(self, session_id: str, user_input: str, max_iterations: int = 8):
        self.trace = Trace(
            trace_id=f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            end_time="",
            user_input=user_input,
            max_iterations=max_iterations,
            actual_iterations=0,
            total_tokens=0,
            total_duration_ms=0.0,
            events=[],
            final_output="",
            success=False,
            error=None
        )
        self._step = 0
        self._start_time = time.time()

    def add_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        duration_ms: float = 0.0,
        token_count: int = 0
    ):
        """添加一个事件到 Trace"""
        self._step += 1
        event = TraceEvent(
            step=self._step,
            type=event_type,
            timestamp=datetime.now().isoformat(),
            data=data,
            duration_ms=duration_ms,
            token_count=token_count
        )
        self.trace.events.append(event)
        self.trace.total_tokens += token_count

    def set_final_output(self, output: str, success: bool = True):
        """设置最终输出"""
        self.trace.final_output = output
        self.trace.success = success
        self.trace.end_time = datetime.now().isoformat()
        self.trace.total_duration_ms = (time.time() - self._start_time) * 1000

    def set_error(self, error: str):
        """设置错误信息"""
        self.trace.error = error
        self.trace.success = False
        self.trace.end_time = datetime.now().isoformat()
        self.trace.total_duration_ms = (time.time() - self._start_time) * 1000

    def increment_iteration(self):
        """增加迭代计数"""
        self.trace.actual_iterations += 1

    def get_trace(self) -> Trace:
        """获取完整的 Trace"""
        return self.trace

    def save(self, directory: str = "traces"):
        """保存 Trace 到文件"""
        import os
        os.makedirs(directory, exist_ok=True)

        # 保存完整 Trace（JSON）
        json_path = os.path.join(directory, f"{self.trace.trace_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.trace.to_json())

        # 保存 JSONL 格式（便于流式读取）
        jsonl_path = os.path.join(directory, f"{self.trace.trace_id}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(self.trace.to_jsonl())

        return json_path


# 全局 Tracer 实例
_current_tracer: Optional[Tracer] = None

def get_tracer() -> Optional[Tracer]:
    """获取当前 Tracer"""
    return _current_tracer

def set_tracer(tracer: Tracer):
    """设置当前 Tracer"""
    global _current_tracer
    _current_tracer = tracer

def clear_tracer():
    """清除当前 Tracer"""
    global _current_tracer
    _current_tracer = None