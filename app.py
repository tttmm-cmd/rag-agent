"""
FastAPI 接口:?corpus= 切换语料;多轮对话用 session_id 关联
启动: python app.py (等价 uvicorn app:app --port 8000)
"""
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query
from pydantic import BaseModel

from rag.store import rag_store
from memory import create_memory_session
from main import agent_loop

app = FastAPI(title="文档智能问答 Agent", version="0.1.0")

# 进程内会话表(演示够用;生产换 Redis/数据库)
_sessions = {}


class AskRequest(BaseModel):
    question: str
    session_id: str = "s0"
    completion_condition: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "corpus": rag_store.corpus}


@app.post("/ask")
def ask(req: AskRequest, corpus: str = Query("zhizheng")):
    if rag_store.corpus != corpus:
        rag_store.load(corpus)
    if req.session_id not in _sessions:
        _sessions[req.session_id] = create_memory_session(req.session_id)
    answer = agent_loop(
        req.question,
        _sessions[req.session_id],
        completion_condition=req.completion_condition,
    )
    return {"answer": answer, "corpus": corpus}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
