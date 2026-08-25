#!/usr/bin/env bash
# verify.sh — 自动验证:单元测试(不依赖 LLM/服务,真门禁)
#
# 用法:
#   bash verify.sh   # 快检:pytest tests/(mock/纯逻辑,不调模型)
#   bash verify.sh --all  # 预留:Day3 加评测阶段(bash eval/run_eval.sh)
set -e
export PYTHONIOENCODING=utf-8

# 优先使用 langchain-demo 环境;找不到回退 PATH 中的 python
PYTHON="D:/Anaconda/envs/langchain-demo/python.exe"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(command -v python || true)"
fi

echo "========== [阶段 1] 单元测试(pytest) =========="
if ! "$PYTHON" -m pytest tests/ -q; then
    echo ""
    echo "❌ 单元测试存在失败,停止。"
    exit 1
fi
echo ""
echo "✅ 单测全绿。"

if [ "$1" = "--all" ]; then
    echo "⚠️  评测阶段尚未接入(Day3 排期中)。"
fi
exit 0
