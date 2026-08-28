# 政务采购文档智能问答 Agent

> 独立项目(对标 5 家 agent 岗 JD)。文档问答 + 表格智能分析,**数据源可换**:
> 同一套解析/分块/向量化管道,换语料目录重索引即可(面试讲"数据接入标准化")。
> 手写 agent_loop,不依赖 RAG/Agent 框架;Python 全栈,含前端与 Docker 部署。

- 语料:政府采购招标文件(青岛、滨海、电子职院等 7 个项目,113 份文档,1926 块)
- 评测:30 条用例(正常 16 / 边界 10 / 对抗 4),已实测 10 条:召回率 **100%**、答案准确 **10/10**、引用正确 **10/10**、平均时延 3.3s

---

## 快速开始

```bash
conda activate langchain-demo    # 项目依赖装在此环境(Python 3.11)
pip install -r requirements.txt
python index.py --corpus zhizheng   # 建索引(只需一次;换语料换名字重跑)
python app.py                       # 起 FastAPI: http://localhost:8000
```

浏览器打开 **http://localhost:8000** 即单页对话界面。接口:`POST /api/ask`
body `{"question": "...", "session_id": "..."}`,返回 `{answer, corpus, sources}`。

```bash
bash verify.sh                          # 快检:pytest 单测(不依赖 LLM/服务)
python eval/run_eval.py --corpus zhizheng   # 全量评测
```

---

## 架构

```
浏览器(单页) ──/api/ask──> FastAPI(app.py)
                            │  _sessions 会话表 + ?corpus= 切语料
                            ▼
                      agent_loop(main.py)  手写多轮循环,MAX_ITER 兜底
                            │  goal_evaluator 完成条件判断(可选)
                            ▼
                       工具系统(tool_system.py)  注册表分发,9 个工具
   ┌──────────┬──────────────┬─────────────┬─────────────┐
   │ retrieve │ parse_document│ query_table │ 4×memory    │ cite_source
   ▼          ▼               ▼             ▼             ▼
RAG 管线    原文精读(±2段)   Excel 行抽取   三层记忆     引用生成
(rag/ 模块)  (data/ 文件)    (openpyxl)    (memory.py)   (归因闭环)
```

- LLM(回答生成)与 embedding(向量化)走**云端 API**,密钥只在 `.env`;本机只做
  解析/分块/检索(FAISS+BM25)/编排。
- 本地开发直接 `python app.py`;生产用 Docker + nginx(见部署),nginx 反代
  80 → 8000,`proxy_read_timeout 300s` 保证长问答不被掐断。

---

## 核心技术

### 1. RAG 检索策略(向量 + BM25 + RRF 混合检索)

| 环节 | 做法 | 解决什么问题 |
|---|---|---|
| 双路召回 | 向量(FAISS 余弦)+ BM25(jieba 中文 bigram)并集 | 向量管语义同义,关键词管精确命中(TOPS/工期/文号),互补 |
| 融合 | RRF(k=60)只按名次求和,不按分数 | 两路分数量纲不同(cosine≈0~1 vs BM25 无上限),分数不可比 |
| query 分工 | 向量吃**整句含项目名**(消歧);BM25 吃**剥项目名的内容词** | 项目名是高频衬词,不剥会把封面/邀请页顶到最前(坑②) |
| 项目过滤 | `project_filter` = metadata filtering:全局召回 → 按 source 过滤 → **项目子集内 RRF** | 别项目共词块(如"专门面向中小企业")全局命中更强,会挤掉本项目真块 |
| 文本归一化 | NFKC(库两侧都归一) | pypdf 提取的康熙部首变体『机器⼈』导致 BM25/向量双失配 |
| 封面降权 | `#0` 块 score × 0.5 | 每文档封面块语义最近,不降权会霸榜 |
| 原文兜底 | 资格/资质类:`parse_document(keyword=资格要求)` 读原文(±2 段) | 资格条文常埋在向量 top-k 之外,检索读不到,原文一读就有 |

### 2. Harness 五子系统落地(对标 learn-harness-engineering)

| 子系统 | 干什么 | 本项目落点 | 实战坑(面试讲) |
|---|---|---|---|
| **Instructions** | 指令定义模型怎么干活 | `main.py DEFAULT_PROMPT`(检索只放内容词/必须引用/诚实边界)+ 9 工具 schema | 系统提示词是"怎么干活"的契约 |
| **State** | 完整状态(输入+输出+记忆+trace) | `memory.py` 三层记忆 + `trace/` 事件落盘 | **只记模型输出、漏用户输入 → 多轮"这个项目"指代断裂**(已修:用户问题也入短期记忆) |
| **Verification** | 判断答得好不好 | `_stalling` 质量闸门(带出处或诚实拒绝才放行)+ `goal_evaluator` 完成条件判断器 | 弱模型"以上是…全部要求"复述废话 → 拦下走兜底 |
| **Scope** | 能做什么/不能做什么 | 工具白名单注册表(`❌ 未知工具` 拒绝)+ `project_filter` 项目内检索限定 + "语料没有就说没有"诚实边界 | 对抗用例(问语料外内容)→ 诚实拒绝不编造 |
| **Lifecycle** | 生命周期与失败恢复 | MAX_ITER 强制终止 → `direct_answer` 确定性兜底 → 兜底也挂则硬答(tools=[]) | 弱模型 8 步内空转 → 生命周期把控制权交给更稳的确定性路径 |

### 3. 多层记忆管理(`memory.py`)

| 层 | 存什么 | 存取方式 | 工具 |
|---|---|---|---|
| 短期对话 | 当前对话 user/assistant 消息 | 直接拼进 LLM 上下文;每轮落盘 `agent_memory/session_*.json` | — |
| 长期 KV | 确定性键值(文件/参数) | `get_kv(key)` 精确查 | `memory_kv_write/get` |
| 长期语义 | 长文本/任务过程 | `search_semantic(keyword)` 模糊检索 | `memory_sem_add/search` |

坑(State 子系统,面试必讲):短期记忆只存模型输出时,第二轮"这个项目…?"的
"这个"无从解析 → 答"未指明项目"。修复:**用户问题 + 模型回复 + 兜底答案都入
短期记忆**,第二轮正确指代回第一轮项目。

### 4. 工具系统(9 个,注册表分发)

4 个记忆工具 + 5 个文档问答工具:`retrieve`(混合检索,带 id/来源/分数)、
`parse_document`(PDF/Word/Excel 解析,`keyword` 精读)、`query_table`(Excel 行抽取,
数据智能分析)、`summarize`(摘要压缩)、`cite_source`(按 id 生成规范引用 →
**回答必须用它附出处 → 评测归因闭环**)。注册表查表分发,新增工具只加一个
`register` 调用,不动 agent_loop。

### 5. 确定性兜底(`rag/direct.py`)

agent 空转/空话/超步数时接管:识别项目 → 项目内双路检索 → 资格类读原文 →
一次生成(仍必须带引用)。**路线全写死,只有措辞交给模型**,产出比模型自由发挥更稳。

### 6. 评测体系(`eval/`)

- 三类用例:**normal**(业务问答)/ **boundary**(精确值:TOPS/工期 25 天/无特定资格)/
  **adversarial**(诚实拒绝:语料外内容)。
- 指标:检索召回率 / 答案准确率 / 引用正确率 / 时延。
- **口径一致**:评测器与 `_stalling` 用同一套引用判定(`[id=…]`/cite_source/出处),
  否则评测数字会撒谎(曾把真实 10/10 引用误判成 0%)。
- 评测集 30 条(正常 16 / 边界 10 / 对抗 4);已实测 10 条:召回 100%,准确 10/10,
  引用 10/10,平均 3.3s(其余用例随语料扩展逐步纳入实测)。

---

## 目录结构

```
rag/            RAG 管线:parse_document → chunker → embedder → vector_store → retrieve
  retrieve.py     向量+BM25 双路 → RRF 融合;project_filter 项目内融合
  direct.py       确定性兜底(直接答/读原文)
  store.py        RAGStore 单例,按 corpus 名切换
main.py         agent_loop(多轮循环 + _stalling 闸门 + 强制终止 + 兜底)
tool_system.py  9 工具注册表 + execute_tool 统一分发
memory.py       三层记忆 + 持久化
llm.py          LLM 请求封装(OpenAI 兼容 /chat/completions)
index.py        建索引 CLI:python index.py --corpus <名>
app.py          FastAPI:静态前端 + /api/ask(+ /api 别名) + 引用提取 + 500 兜底
static/         前端单页(index.html/style.css/app.js)
eval/           评测:test_cases.py / run_eval.py / evaluator.py / report.md / logs/
tests/          解析/分块单测
Dockerfile*     部署:api 容器 + nginx 容器
docker-compose.yml / nginx.conf / .dockerignore / DEPLOY.md
faiss_zhizheng.index + .json   向量索引(按语料分文件)
data/zhizheng/   语料(政府采购招标文件)
```

---

## 部署

- **本地开发**:`python app.py` → `http://localhost:8000`
- **Docker + nginx**(云服务器):`docker compose up --build -d` → `http://<服务器IP>`
  双容器(api + nginx),密钥经 `env_file: .env` 注入不进镜像,`agent_memory/ traces/`
  挂 volume 持久化。详见 **DEPLOY.md**。

部署关键坑(均已修):
- Windows 建索引的 `source` 反斜杠路径,在 Linux 上 `os.path.basename`/`os.path.join`
  不识别 → `direct.py` 归一化为正斜杠。
- 单 worker 是故意的:进程内会话表 + 索引单例,多 worker 会串台。
- Windows 下 stdout 重定向回退 GBK,打印 emoji 崩 `UnicodeEncodeError` →
  `app.py` 入口 `reconfigure(utf-8)`。

---

## 面试亮点 / 技术决策

1. **手写 agent_loop + goal 判断器**,不依赖框架(能讲 loop 每一步)。
2. **评测体系 + 归因闭环**(5 家 JD 里 4 家点名 Evaluation/Observability)。
3. **数据源可换**:管道与语料解耦,`index.py --corpus <名>` 重索引即换语料。
4. **确定性兜底**:模型能力天花板用工程绕行(Lifecycle 失败恢复的落点)。
5. **harness 五子系统每个都有代码落点 + 实战坑**(对标 learn-harness-engineering)。
6. 决策:**用 Python 不用 Pi**——5 家 JD 全 Python 生态,3 天可交付;Pi 无内置权限
   系统,换栈全推倒。

## 已知边界 / 后续方向

- **无鉴权、单 worker 串行**:演示够用;公网暴露前加访问口令 + 换 Redis 会话。
- 扫描件/图片页需 OCR 或多模态(当前 pypdf 纯文本解析,纯图片页跳过)。
- Re-ranking 可升级为 cross-encoder 精排(检索接口已预留 `rerank` 开关)。
- 同义多 query 扩充、章节级分块(见 PROGRESS.md 候选清单)。

---

*详细踩坑记录见 `PROGRESS.md`(Day1~Day3 + 前端部署)与 `RAG问答Agent四大坑总结.txt`。*
