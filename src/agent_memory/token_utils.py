import re
from typing import Any


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese = re.sub(r"[\u4e00-\u9fff]", "", text)
    
    words = re.findall(r"\b\w+\b", non_chinese)
    punctuation = len(re.findall(r"[.,!?;:'\"()\[\]{}]", non_chinese))
    
    chinese_tokens = chinese_chars
    word_tokens = len(words)
    punct_tokens = punctuation
    
    total = chinese_tokens + word_tokens + punct_tokens
    return max(1, total)


def estimate_message_tokens(message: Any) -> int:
    role_tokens = len(message.role.value)
    content_tokens = estimate_tokens(message.content)
    return role_tokens + content_tokens + 3


def estimate_list_tokens(items: list[Any]) -> int:
    total = 0
    for item in items:
        if hasattr(item, "content"):
            total += estimate_tokens(item.content)
        elif isinstance(item, str):
            total += estimate_tokens(item)
        elif isinstance(item, dict):
            for v in item.values():
                if isinstance(v, str):
                    total += estimate_tokens(v)
    return total
