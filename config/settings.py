"""
配置管理 - 支持模型切换、版本管理
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    api_key: str
    base_url: str
    max_iterations: int = 8
    temperature: float = 0.7


@dataclass
class AgentSettings:
    """Agent全局配置"""
    version: str = "1.0.1"  # Agent版本
    default_model: str = "deepseek-v4-flash"  # 默认模型
    models: dict = field(default_factory=dict)
    max_iterations: int = 8
    enable_mcp: bool = True
    enable_cache: bool = True
    enable_trace: bool = True
    alert_threshold_failure_rate: float = 20.0  # 失败率告警阈值 (%)
    alert_threshold_cost: float = 100.0  # 单次成本告警阈值 ($)

    def __post_init__(self):
        # 从环境变量加载模型配置
        self.models = {
            "deepseek-v4-flash": ModelConfig(
                name="deepseek-v4-flash",
                api_key=os.getenv("LLM_API_KEY", ""),
                base_url=os.getenv("LLM_BASE_URL", ""),
                max_iterations=self.max_iterations
            ),
            # 可添加更多模型
            # "gpt-4": ModelConfig(...),
            # "claude-3": ModelConfig(...),
        }

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models.get(name)

    def get_current_model(self) -> ModelConfig:
        """获取当前模型"""
        return self.get_model(self.default_model) or list(self.models.values())[0]


# 全局配置单例
_settings = None


def get_settings() -> AgentSettings:
    global _settings
    if _settings is None:
        _settings = AgentSettings()
    return _settings


def reload_settings():
    """重新加载配置"""
    global _settings
    _settings = AgentSettings()


def save_version_history(self):
    """记录版本变更历史"""
    history_file = "version_history.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)

    history.append({
        "version": self.version,
        "timestamp": datetime.now().isoformat(),
        "model": self.default_model
    })

    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)