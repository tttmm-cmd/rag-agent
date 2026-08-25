import os
import sys

# 把项目根目录放进 sys.path,让 pytest 能 import rag/ / main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
