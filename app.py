"""
FastAPI 接口:?corpus= 切换语料;多轮对话用 session_id 关联
本地启动: python app.py (等价 uvicorn app:app --port 8000)
前端单页: GET / → static/index.html;问答请求走 POST /api/ask
(nginx 反代时 /api/ask 原样转发到本服务,本地直连也通,前端不用改地址)
"""
import os
import re
import sys

# Windows 下 stdout 重定向到文件/管道时 Python 退回 GBK,打印 ⚠️✅❌ 会 UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.store import rag_store
from memory import create_memory_session
from main import agent_loop

app = FastAPI(title="文档智能问答 Agent", version="0.1.0")

# 进程内会话表(演示够用;生产换 Redis/数据库)
_sessions = {}

# 答案里 [id=文件名#块号] → 前端可点来源
SOURCE_RE = re.compile(r"id=([^\s,，;；]+?)#(\d+)")


class AskRequest(BaseModel):
    question: str
    session_id: str = "s0"
    completion_condition: str = ""


def extract_sources(answer: str) -> list[dict]:
    """从答案提取引用来源,按文件去重 → [{file, block}]"""
    seen: dict[str, int] = {}
    for m in SOURCE_RE.finditer(answer):
        fn, blk = m.group(1), m.group(2)
        if fn not in seen:
            seen[fn] = int(blk)
    return [{"file": f, "block": b} for f, b in seen.items()]


@app.get("/health")
def health():
    return {"status": "ok", "corpus": rag_store.corpus}


@app.post("/ask")
@app.post("/api/ask")  # 本地直连与 nginx 反代共用一条路径
def ask(req: AskRequest, corpus: str = Query("zhizheng")):
    try:
        if rag_store.corpus != corpus:
            rag_store.load(corpus)
        if req.session_id not in _sessions:
            _sessions[req.session_id] = create_memory_session(req.session_id)
        answer = agent_loop(
            req.question,
            _sessions[req.session_id],
            completion_condition=req.completion_condition,
        )
        return {"answer": answer, "corpus": corpus, "sources": extract_sources(answer)}
    except Exception as e:
        # agent_loop 内部异常(LLM 欠费/接口超时/索引缺失)→ 返回 JSON,不裸崩
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# 静态前端
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
