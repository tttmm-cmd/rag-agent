# 进度状态(与计划文档 3 天排期对应)

## Day1 数据+入库
- [x] 收集政务文档 30-50 份 → `data/zhizheng/`(用户下载 113 个文件,7 个项目)
- [x] `python index.py --corpus zhizheng` 建索引(1926 chunks,现用 qwen3.7-text-embedding)
- [x] 解析工具单测通过(5 tests)
- [x] 冒烟:retrieve 能召回(资格/截止时间/采购内容实测)

## Day2 问答链路
- [x] agent_loop + 5 工具多轮跑通(带引用 + 确定性兜底 `rag/direct.py`)
- [x] FastAPI `?corpus=` 接口可调(实测 200 / ~20s / 语料切换正常)
- [x] 多轮记忆(session 追问)——已修 bug 并验证通过
- [x] eval CASES 更新为真实语料(10 条,recall 100%)
- [x] harness 五子系统逐项验证通过(State 完整落地 + Verification/Scope/Lifecycle/Instructions,2026-08-26)

## Day3 评测+交付
- [ ] 30-50 政务用例评测(准确率/召回率/引用正确率)
  (重索引演示/招聘语料任务已取消;演示脚本/README/Dockerfile 并入前端+部署阶段)

## 已修复的关键坑(Day2 踩坑记录)
1. **上下文压缩器 bug**:不足 keep_recent 轮 user 消息时把整个对话(含原始问题)摘要掉 → 模型漂移。
   已改为"不足 N 轮就不压缩",单轮问答不再触发。
2. **检索被项目名 bigram 绑架**:query 带"青岛大学附属医院塑料制品采购项目"永远命中封面/报价页。
   已让系统剥项目名(rag/direct.py `_content_query`)用内容关键词检索。
3. **parse_document 截断**:整篇文档前 2000 字看不到深处章节。
   已支持 keyword 过滤 + 命中段带前后文(±2 段),资格清单横跨多页也能一次读到。
4. **弱模型(deepseek)多步导航不可靠**:8 步内迷失关键词/答空话。
   已加 `rag/direct.py direct_answer` 确定性兜底,agent 空转或空话时自动接管。
   (① ② ③ 是工程问题,④ 是模型能力天花板,只能绕行)
5. **多轮记忆 State 子系统缺陷**:agent_loop 只把模型回复存进短期记忆,
   用户问题没存(main.py:71 只追加局部 messages,没 add_short_msg)。
   第二轮"这个项目…?"时,上下文里没有第一轮问题,"这个项目"无从解析
   → 答"您未指明具体是哪个项目"。修复:用户问题进短期记忆 + 兜底答案
   也存回。修复后第二轮正确指代青岛项目,答出采购包1-4。
6. **评测器引用判定假报警(eval/evaluator.py)**:cited 只看("出处","引用","引文")
   三个中文词,而 agent 实际输出 [id=文件#块号] 格式 → 10 条全带引用却判 0% 引用正确率。
   修复:与 main._stalling 同一口径(id=/cite_source/#\d+/出处/引用/引文),离线验证 6/6。
   面试点:评测器和质量闸门用同一判定口径,否则评测数字会撒谎。

## Day3 全量评测(2026-08-27,10 条 CASES)
- 检索召回率 100% | 答案准确率 10/10 | 引用正确率 10/10(修复评测器后)| 平均时延 9.3s
- 三类用例全过:normal 业务问答、boundary 精确值(21TOPS/25天/无特定资格)、
  adversarial 诚实拒绝(大模型训练/区块链,均答"语料没有"并说明)
- 亮点:截止时间题答"详见采购公告"(正文确实未写明,诚实不编造)

## 召回率 30% → 70% 做了什么
1. **query 改写**:剥项目名+衬词只留内容词(`_content_query`, rag/direct.py:93-99)
2. **项目识别重写**:候选来自文件夹名+直接文件 basename,子串/公共前缀≥6 命中
3. **评测按项目过滤命中**(`_in_project`):别项目飘来的块不计分(指标变公平)
4. **top_k 5 → 10**

## 召回率 70% → 80% → 100% 做了什么(本轮)
1. **换 embedding 模型**:text-embedding-v3 免费额度耗尽(400 FreeTierOnly) →
   换 `qwen3.7-text-embedding`(免费额度剩 980K token,1024 维不变)。
   embedder.py 默认值已改,`.env` 无覆盖则全局生效。
2. **NFKC 归一化修复字符变体**(`rag/text_norm.py`):pypdf 提取的康熙部首变体
   『机器⼈』(U+2F08)/『算⼒』(U+2F0A)导致 BM25 词不匹配、向量语义漂移。
   NFKC 统一成标准字符,入库(chunker)+ 查询(retrieve)两侧都归一化。
   → 设备(机器人)/TOPS 两条历史 MISS 翻成 HIT。
3. **项目内融合检索**(`retrieve_hybrid` 增加 `project_filter`):已知项目时,
   「全局召回候选 → 项目过滤 → **项目子集内** BM25+RRF 融合」。
   修掉「中小企业/特定资格要求」这类跨项目共词时,别项目块全局命中更强、
   把目标项目块挤出 top-k 的问题(青岛案例:滨海磋商文件"专门面向中小企业
   采购 100%"把青岛真正的 SME 块挤出)。evaluator 与 direct_answer 同步改。

## 召回率再提高的候选方案(Day3 可选,已落地项标注 ✅)
1. ✅ 混合检索(向量 + BM25/jieba 关键词并集)
2. ✅ 项目已知 → 强制项目内召回(已推广到常规检索,project_filter)
3. 同义多 query 扩充(设备/算力/资质)
4. DashScope embedding 带 query_type 指令
5. 章节级分块(section-aware)
6. "应诚实拒绝"类用例单列指标口径(采购公告不在语料等)

## 前端 + Docker 部署(2026-08-27)
- [x] 前端单页 `static/`(index.html + style.css + app.js):对话界面、提问、
      answer 里 [id=文件#块号] 渲染成底部来源标签、多轮追问(本地 sessionId)、新会话
- [x] app.py 升级:挂 /static、/ 返回单页、`/api/ask` 别名路由(本地直连与
      nginx 反代同路径,前端不用改地址)、extract_sources 从答案提取引用、
      try/except 把 agent 异常包装成 500 JSON(不裸崩)
- [x] direct.py 修复:Windows 反斜杠 source 在 Linux 上 os.path.basename / os.path.join
      不识别 → 归一化为正斜杠(Docker 部署必踩)
- [x] Dockerfile(python:3.11-slim, tiktoken cl100k_base 预热,单 worker)/
      Dockerfile.nginx + nginx.conf(proxy_read_timeout 300s, gzip)/
      docker-compose.yml(双服务, env_file: .env, agent_memory/traces 挂 volume)/
      .dockerignore(排除 .env)/ DEPLOY.md
- [x] 本地验证:health / 首页 / 静态资源 200;真实问答 HTTP 200,12.0s,引用正确提取
- [ ] Docker 部署实测(本机未装 Docker,待云服务器)
- 坑 #7(前端阶段):Windows 下 stdout 重定向到文件/管道时 Python 退回 GBK 编码,
  agent_loop 打印 ⚠️✅❌(U+26A0 等)触发 UnicodeEncodeError → 整个请求 500。
  前台终端跑没事,后台/重定向必现。修复:app.py 入口 sys.stdout/stderr.reconfigure(utf-8)。
  面试点:服务端代码的终端输出要显式指定编码,不能依赖运行环境。

## README 收尾(2026-08-27)
- [x] README.md:功能特性 / 快速开始 / 架构图 / RAG 检索策略 / Harness 五子系统 /
      多层记忆 / 工具系统 / 确定性兜底 / 评测体系 / 目录结构 / 部署 / 面试亮点 / 边界方向
- [x] query embedding LRU 缓存(rag/embedder.py,128 条):agent 多轮重复检索省 DashScope 往返
