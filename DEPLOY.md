# 部署说明(Docker + nginx)

本地开发(不上 Docker):
```bash
conda activate langchain-demo       # 项目依赖装在这个环境(Python 3.11)
pip install -r requirements.txt
python index.py --corpus zhizheng   # 只需建一次索引
python app.py                       # 浏览器打开 http://localhost:8000
```

Docker 部署(云服务器):
```bash
# 前提:装了 Docker + Docker Compose,项目根目录有 .env(含 LLM_API_KEY 等)
docker compose up --build -d
# 浏览器打开 http://<服务器IP>   (nginx 80 端口 → FastAPI 8000)
# 看日志: docker compose logs -f api
```

## 架构
```
浏览器 ──80──> nginx(反代/超时300s/gzip) ──> FastAPI(静态页 + /api/ask)
                                              │
                              faiss_zhizheng.index 语料索引(data/ 进镜像)
                              agent_memory/ traces/(挂 volume 持久化)
```

## 注意
- **单 worker 是故意的**:进程内会话表 + rag_store 单例,多 worker 各持一份索引和
  会话,多轮记忆会串台。要扩并发得把会话和索引外置(Redis + 独立检索服务)。
- 密钥只经 `env_file: .env` 注入,不进镜像;`.dockerignore` 已排除 `.env`。
- 索引是 Windows 上建的,source 路径含反斜杠;`rag/direct.py` 已对 Linux 归一化。
- 换语料:放文件进 `data/<corpus>/` → `docker compose exec api python index.py --corpus <corpus>`
  → 请求带 `?corpus=<corpus>`。
