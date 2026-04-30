from datetime import datetime
from typing import Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"


class Message(BaseModel):
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    id: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = f"{self.role.value}_{self.timestamp.timestamp():.6f}"


class MemoryItem(BaseModel):
    id: str
    memory_type: MemoryType
    content: str
    source: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = 1.0
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionType(str, Enum):
    ADD_STM = "ADD_STM"
    EVICT_STM = "EVICT_STM"
    SUMMARIZE = "SUMMARIZE"
    UPSERT_LTM = "UPSERT_LTM"
    RETRIEVE_LTM = "RETRIEVE_LTM"
    INVALIDATE = "INVALIDATE"
    DEDUPLICATE = "DEDUPLICATE"
    BUILD_CONTEXT = "BUILD_CONTEXT"


class ActionLog(BaseModel):
    action: ActionType
    description: str
    input_summary: str
    output_summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class MemoryDecisionLog(BaseModel):
    conversation_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    actions: list[ActionLog] = Field(default_factory=list)
    token_budget_initial: Optional[int] = None
    token_budget_remaining: Optional[int] = None
    short_term_count: int = 0
    conversation_count: int = 0
    long_term_count: int = 0

    def add_action(
        self,
        action: ActionType,
        description: str,
        input_summary: str,
        output_summary: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.actions.append(
            ActionLog(
                action=action,
                description=description,
                input_summary=input_summary,
                output_summary=output_summary,
                details=details or {},
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp.isoformat(),
            "actions": [
                {
                    "action": a.action.value,
                    "description": a.description,
                    "input_summary": a.input_summary,
                    "output_summary": a.output_summary,
                    "details": a.details,
                }
                for a in self.actions
            ],
            "token_budget_initial": self.token_budget_initial,
            "token_budget_remaining": self.token_budget_remaining,
            "short_term_count": self.short_term_count,
            "conversation_count": self.conversation_count,
            "long_term_count": self.long_term_count,
        }
