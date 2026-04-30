from datetime import datetime
from typing import Any, Optional

from agent_memory.config import ShortTermConfig
from agent_memory.models import Message, MemoryItem, MemoryType
from agent_memory.token_utils import estimate_message_tokens, estimate_tokens


class ShortTermMemory:
    def __init__(self, config: Optional[ShortTermConfig] = None):
        self.config = config or ShortTermConfig()
        self.messages: list[Message] = []
        self._tokens: int = 0

    def add_message(self, message: Message) -> tuple[bool, list[Message]]:
        message_tokens = estimate_message_tokens(message)
        self.messages.append(message)
        self._tokens += message_tokens
        
        evicted: list[Message] = []
        
        while self._should_evict():
            if self.messages:
                evicted_msg = self.messages.pop(0)
                self._tokens -= estimate_message_tokens(evicted_msg)
                evicted.append(evicted_msg)
        
        return len(evicted) > 0, evicted

    def get_messages(self, max_tokens: Optional[int] = None) -> list[Message]:
        if max_tokens is None:
            return self.messages.copy()
        
        result: list[Message] = []
        current_tokens = 0
        
        for msg in reversed(self.messages):
            msg_tokens = estimate_message_tokens(msg)
            if current_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result

    def get_token_count(self) -> int:
        return self._tokens

    def get_message_count(self) -> int:
        return len(self.messages)

    def _should_evict(self) -> bool:
        token_ratio = self._tokens / self.config.max_tokens
        message_ratio = len(self.messages) / self.config.max_messages
        threshold = self.config.eviction_threshold
        
        return token_ratio > threshold or message_ratio > threshold

    def format_for_context(
        self, 
        messages: Optional[list[Message]] = None,
        separator: str = "\n"
    ) -> str:
        msgs = messages or self.messages
        return separator.join(
            f"[{msg.role.value}] {msg.content}"
            for msg in msgs
        )

    def clear(self) -> None:
        self.messages.clear()
        self._tokens = 0

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
                "metadata": msg.metadata,
                "timestamp": msg.timestamp.isoformat(),
                "id": msg.id,
            }
            for msg in self.messages
        ]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]], config: Optional[ShortTermConfig] = None) -> "ShortTermMemory":
        from agent_memory.models import MessageRole
        
        stm = cls(config)
        for item in data:
            msg = Message(
                role=MessageRole(item["role"]),
                content=item["content"],
                metadata=item.get("metadata", {}),
                timestamp=datetime.fromisoformat(item["timestamp"]) if "timestamp" in item else datetime.now(),
                id=item.get("id", ""),
            )
            stm.messages.append(msg)
            stm._tokens += estimate_message_tokens(msg)
        return stm
