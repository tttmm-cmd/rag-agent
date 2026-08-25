# gov-doc-qa — 文档智能问答 Agent

对标 5 家 agent 岗 JD 的独立项目。**week3 原样保留,本仓只复用其已验证组件**
(memory / context_compactor / goal_evaluator / trace / eval 框架)。

## 定位
文档问答 + 表格智能分析,**数据源可换**:同一套解析/分块/向量化管道,
换语料目录重索引即可(面试讲"数据接入标准化")。

## 结构
```
rag/                RAG 管线(新写):解析 → 分块 → 向量化 → FAISS → 检索(+Re-ranking)
  parse_document.py   PDF/Word/Excel/txt 解析,带来源定位(页码/工作表)
  chunker.py          段落优先 + 滑动窗口,带 overlap
  embedder.py         DashScope text-embedding-v3(可换)
  vector_store.py     FAISS 余弦索引 + 侧边元数据
  retrieve.py         向量检索 + 中文 bigram 轻量 Re-ranking
  store.py            RAGStore 单例,按 corpus 名切换
main.py              agent_loop(复用 week3:压缩/goal 判断器/trace)
tool_system.py       9 工具 = 4 记忆(week3)+ 5 文档问答;统一分发入口(可扩展 MCP)
llm.py               LLM 请求封装(独立模块,避免循环导入)
index.py             建索引 CLI: python index.py --corpus <名>
app.py               FastAPI: POST /ask?corpus=zhizheng
eval/                评测按语料分(eval/zhizheng/, eval/techdocs/)
config/  trace/      week3 原样复制
```

## 命令
| 命令 | 作用 |
|---|---|
| `python index.py --corpus zhizheng` | 建/重建政务语料索引 |
| `python app.py` | 起 FastAPI(端口 8000) |
| `bash verify.sh` | 快检:pytest 单测(不依赖 LLM/服务) |
| `python -m pytest tests/ -q` | 直接跑单测 |

## 5 个文档工具
parse_document / retrieve / query_table / summarize / cite_source
(回答必须用 cite_source 附出处 → 评测归因闭环)

## 规则
- 密钥只在 `.env`(已 gitignore),改 `.env.example` 加注释
- 工具函数保持纯函数式,需要语料走 `rag.store.rag_store` 单例
- 新工具:在 tool_system.py 写函数 + register 一行,不改 agent_loop
