"""文本归一化:NFKC 把兼容字符(康熙部首等)映射回标准 CJK 字符。

招投标 PDF 提取常见污染:『机器⼈』(U+2F08)应为『机器人』(U+4EBA),
『算⼒』(U+2F0A)应为『算力』。标准库 unicodedata,确定性、零依赖、幂等。
"""

import unicodedata


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
