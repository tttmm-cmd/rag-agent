"""跑评测: python eval/run_eval.py --corpus zhizheng [--recall-only]

--recall-only: 只测检索召回(不调 LLM,快);完整评测要调 LLM 跑 agent。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.store import rag_store
from rag.direct import _content_query
from memory import create_memory_session
from main import agent_loop
from eval.evaluator import retrieval_recall, answer_accuracy
from eval.report import build_report
from eval.test_cases import CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="zhizheng")
    ap.add_argument("--recall-only", action="store_true")
    args = ap.parse_args()

    rag_store.load(args.corpus)
    # 与系统实际行为一致:剥掉项目名 + 定位项目内检索(_content_query 返回 (query, 项目))
    recall = retrieval_recall(CASES, args.corpus, query_fn=_content_query)
    print(f"检索召回率: {recall:.0%}")

    if args.recall_only:
        return
    rows = answer_accuracy(CASES, args.corpus,
                           lambda q: agent_loop(q, create_memory_session("eval")))
    report = build_report(recall, rows)
    print(report)
    with open("eval/report.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
