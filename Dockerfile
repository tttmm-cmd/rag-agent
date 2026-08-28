FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # 预热 tiktoken cl100k_base 词表:运行时首次调用会联网下载,容器里容易失败
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# 全量内容:data/(语料) + faiss_*.index/.json(索引) + static/(前端) 都会进镜像
COPY . .

EXPOSE 8000

# 单 worker:进程内会话表 + rag_store 单例,多 worker 会各持一份索引/会话,多轮记忆会串台
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
