"""评测报告:汇总指标,输出 markdown 文本(供 eval/report.md)"""
from datetime import datetime


def build_report(recall: float, rows: list[dict]) -> str:
    total = len(rows)
    acc = sum(r["accuracy"] for r in rows) / total if total else 0.0
    cit = sum(r["cited"] for r in rows) / total if total else 0.0
    avg_lat = sum(r["latency_s"] for r in rows) / total if total else 0.0

    lines = [
        "# 评测报告",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"用例数: {total} | 检索召回率: {recall:.0%} | 答案准确率: {acc:.0%} | 引用正确率: {cit:.0%} | 平均时延: {avg_lat:.1f}s",
        "",
        "| 问题 | 类型 | 准确 | 引用 | 时延(s) | 答案预览 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['question']} | {r['kind']} | {'✅' if r['accuracy'] else '❌'} "
            f"| {'✅' if r['cited'] else '❌'} | {r['latency_s']} | {r['answer']} |"
        )
    return "\n".join(lines)
