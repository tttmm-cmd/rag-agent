"""评测用例:按语料分,三类(正常/边界/对抗)。Day3 补到 30-50 条。

格式: (corpus, question, 期望答案包含关键词[], 类型)
- normal       正常业务问题
- boundary     边界/精确值问题(时间、数字)
- adversarial  语料中没有的虚构内容(应诚实说"没有")
"""
CASES = [
    # ---------- zhizheng 政务语料 ----------
    ("zhizheng", "供应商需要具备什么质量管理认证?", ["ISO9001"], "normal"),
    ("zhizheng", "投标截止时间是什么时候?", ["9月30日"], "boundary"),
    ("zhizheng", "语料中不存在的虚构内容是什么?", [], "adversarial"),

    # ---------- techdocs 技术手册语料 ----------
    # ("techdocs", "这款对讲机的防水等级是多少?", ["IP68"], "boundary"),
]
