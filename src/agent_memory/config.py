from pydantic import BaseModel, Field


class ShortTermConfig(BaseModel):
    max_messages: int = Field(default=20, ge=1)
    max_tokens: int = Field(default=2000, ge=10)
    eviction_threshold: float = Field(default=0.8, ge=0.3, le=1.0)


class ConversationConfig(BaseModel):
    compression_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    max_summary_tokens: int = Field(default=500, ge=100)
    summary_chunk_size: int = Field(default=5, ge=2)
    enable_summarization: bool = True


class LongTermConfig(BaseModel):
    embedding_type: str = "tfidf"
    top_k: int = Field(default=3, ge=1)
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    deduplication_threshold: float = Field(default=0.7, ge=0.3, le=1.0)
    max_documents: int = Field(default=1000, ge=10)
    decay_factor: float = Field(default=0.95, ge=0.8, le=1.0)


class MemoryConfig(BaseModel):
    short_term: ShortTermConfig = Field(default_factory=ShortTermConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    long_term: LongTermConfig = Field(default_factory=LongTermConfig)
    default_token_budget: int = Field(default=4000, ge=500)
    context_format_template: str = """## 相关知识
{long_term_context}

## 对话摘要
{conversation_context}

## 当前对话
{short_term_context}"""

    @classmethod
    def default(cls) -> "MemoryConfig":
        return cls()
