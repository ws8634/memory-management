from datetime import datetime
from typing import Any, Optional

from agent_memory.config import ConversationConfig
from agent_memory.models import Message, MemoryItem, MemoryType, MessageRole
from agent_memory.token_utils import estimate_tokens


class ConversationMemory:
    def __init__(self, config: Optional[ConversationConfig] = None):
        self.config = config or ConversationConfig()
        self.summaries: list[str] = []
        self.key_facts: list[dict[str, Any]] = []
        self._summary_tokens: int = 0
        self._facts_tokens: int = 0

    def add_summary(self, summary: str) -> None:
        if not self.config.enable_summarization:
            return
        
        summary_tokens = estimate_tokens(summary)
        
        while self._summary_tokens + summary_tokens > self.config.max_summary_tokens:
            if self.summaries:
                removed = self.summaries.pop(0)
                self._summary_tokens -= estimate_tokens(removed)
        
        self.summaries.append(summary)
        self._summary_tokens += summary_tokens

    def add_key_fact(
        self,
        fact: str,
        source: Optional[str] = None,
        importance: float = 1.0,
    ) -> None:
        existing_idx = self._find_similar_fact(fact)
        
        fact_entry = {
            "fact": fact,
            "source": source,
            "importance": importance,
            "timestamp": datetime.now(),
            "is_active": True,
        }
        
        if existing_idx is not None:
            self.key_facts[existing_idx] = fact_entry
        else:
            self.key_facts.append(fact_entry)
        
        self._facts_tokens += estimate_tokens(fact)

    def invalidate_fact(self, fact_pattern: str) -> int:
        invalidated_count = 0
        for fact in self.key_facts:
            if fact_pattern.lower() in fact["fact"].lower() and fact["is_active"]:
                fact["is_active"] = False
                fact["importance"] *= 0.3
                invalidated_count += 1
        return invalidated_count

    def get_active_facts(self, max_tokens: Optional[int] = None) -> list[dict[str, Any]]:
        active = [f for f in self.key_facts if f["is_active"]]
        active.sort(key=lambda x: x["importance"], reverse=True)
        
        if max_tokens is None:
            return active
        
        result: list[dict[str, Any]] = []
        current_tokens = 0
        
        for fact in active:
            fact_tokens = estimate_tokens(fact["fact"])
            if current_tokens + fact_tokens <= max_tokens:
                result.append(fact)
                current_tokens += fact_tokens
            else:
                break
        
        return result

    def get_summaries(self, max_tokens: Optional[int] = None) -> list[str]:
        if max_tokens is None:
            return self.summaries.copy()
        
        result: list[str] = []
        current_tokens = 0
        
        for summary in reversed(self.summaries):
            summary_tokens = estimate_tokens(summary)
            if current_tokens + summary_tokens <= max_tokens:
                result.insert(0, summary)
                current_tokens += summary_tokens
            else:
                break
        
        return result

    def summarize_messages(self, messages: list[Message]) -> str:
        if not messages:
            return ""
        
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
        
        key_points: list[str] = []
        
        if user_messages:
            first_user = user_messages[0].content[:100]
            key_points.append(f"用户询问: {first_user}")
        
        if assistant_messages:
            last_resp = assistant_messages[-1].content[:150]
            key_points.append(f"助手回复: {last_resp}")
        
        key_topics = self._extract_key_topics(messages)
        if key_topics:
            key_points.append(f"讨论主题: {', '.join(key_topics)}")
        
        summary = " | ".join(key_points)
        return summary

    def format_for_context(
        self,
        max_summary_tokens: Optional[int] = None,
        max_fact_tokens: Optional[int] = None,
        separator: str = "\n",
    ) -> str:
        parts: list[str] = []
        
        summaries = self.get_summaries(max_summary_tokens)
        if summaries:
            parts.append("### 对话摘要")
            parts.extend(f"- {s}" for s in summaries)
        
        facts = self.get_active_facts(max_fact_tokens)
        if facts:
            parts.append("### 关键事实")
            parts.extend(f"- {f['fact']}" for f in facts)
        
        return separator.join(parts) if parts else ""

    def get_token_count(self) -> int:
        return self._summary_tokens + self._facts_tokens

    def get_summary_count(self) -> int:
        return len(self.summaries)

    def get_fact_count(self, include_inactive: bool = False) -> int:
        if include_inactive:
            return len(self.key_facts)
        return len([f for f in self.key_facts if f["is_active"]])

    def clear(self) -> None:
        self.summaries.clear()
        self.key_facts.clear()
        self._summary_tokens = 0
        self._facts_tokens = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summaries": self.summaries,
            "key_facts": [
                {
                    **f,
                    "timestamp": f["timestamp"].isoformat() if isinstance(f["timestamp"], datetime) else f["timestamp"],
                }
                for f in self.key_facts
            ],
            "_summary_tokens": self._summary_tokens,
            "_facts_tokens": self._facts_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Optional[ConversationConfig] = None) -> "ConversationMemory":
        cm = cls(config)
        cm.summaries = data.get("summaries", [])
        cm.key_facts = [
            {
                **f,
                "timestamp": datetime.fromisoformat(f["timestamp"]) if isinstance(f.get("timestamp"), str) else datetime.now(),
            }
            for f in data.get("key_facts", [])
        ]
        cm._summary_tokens = data.get("_summary_tokens", 0)
        cm._facts_tokens = data.get("_facts_tokens", 0)
        return cm

    def _extract_key_topics(self, messages: list[Message]) -> list[str]:
        all_content = " ".join(m.content for m in messages)
        keywords: list[str] = []
        
        common_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那"}
        
        words = [w for w in all_content.split() if len(w) >= 2 and w not in common_words]
        
        word_counts: dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [w for w, c in sorted_words[:3] if c >= 2]
        
        return keywords

    def _find_similar_fact(self, new_fact: str) -> Optional[int]:
        new_fact_lower = new_fact.lower()
        
        for idx, existing in enumerate(self.key_facts):
            existing_lower = existing["fact"].lower()
            
            if new_fact_lower in existing_lower or existing_lower in new_fact_lower:
                return idx
            
            new_words = set(new_fact_lower.split())
            existing_words = set(existing_lower.split())
            if new_words and existing_words:
                intersection = new_words & existing_words
                union = new_words | existing_words
                jaccard = len(intersection) / len(union)
                if jaccard > self.config.compression_threshold:
                    return idx
        
        return None
