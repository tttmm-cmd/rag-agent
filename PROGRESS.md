# 进度状态(与计划文档 3 天排期对应)

## Day1 数据+入库
- [ ] 收集政务文档 30-50 份 → `data/zhizheng/`
- [ ] `python index.py --corpus zhizheng` 建索引
- [ ] 解析工具单测通过
- [ ] 冒烟:retrieve 能召回

## Day2 问答链路
- [ ] agent_loop + 5 工具多轮跑通(带引用)
- [ ] FastAPI `?corpus=` 接口可调
- [ ] 多轮记忆(session 追问)

## Day3 评测+交付
- [ ] 30-50 政务用例评测(准确率/召回率/引用正确率)
- [ ] 技术手册重索引演示(`data/techdocs/` 10-20 份,答 3 问)
- [ ] 演示脚本 + README + (可选)Dockerfile
