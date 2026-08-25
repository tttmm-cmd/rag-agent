"""环境自检 + 依赖安装 + .env 校验 + 建索引 + 启动服务"""
import glob
import os
import re
import subprocess
import sys


def main():
    print(f"使用解释器: {sys.executable}")
    print(f"Python: {sys.version}")
    print("=" * 50)

    print("========== [1/4] 安装依赖 ==========")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("========== [2/4] 校验 .env ==========")
    if not os.path.exists(".env"):
        print("❌ 缺少 .env,请: cp .env.example .env 并填入密钥")
        sys.exit(1)
    with open(".env", encoding="utf-8") as f:
        content = f.read()
    for k in ("LLM_API_KEY", "LLM_BASE_URL", "DASHSCOPE_API_KEY"):
        if not re.search(rf"^{k}=.+", content, re.MULTILINE):
            print(f"❌ .env 缺少必填项 {k}")
            sys.exit(1)
    print("✅ .env 校验通过")

    print("========== [3/4] 检查并建索引 ==========")
    for corpus in os.listdir("data"):
        if not os.path.isdir(os.path.join("data", corpus)):
            continue
        if os.path.exists(f"faiss_{corpus}.index"):
            continue
        docs = glob.glob(os.path.join("data", corpus, "**", "*.*"), recursive=True)
        if any(d.lower().endswith((".pdf", ".docx", ".xlsx", ".txt")) for d in docs):
            print(f"📂 为语料 [{corpus}] 建索引...")
            subprocess.run([sys.executable, "index.py", "--corpus", corpus])

    print("========== [4/4] 🚀 启动服务 ==========")
    os.execv(sys.executable, [sys.executable, "app.py"])


if __name__ == "__main__":
    main()
