from datetime import datetime
from typing import Any, Optional
import json
import os
import uuid

from agent_memory.config import MemoryConfig
from agent_memory.conversation import ConversationMemory
from agent_memory.long_term import LongTermVectorMemory
from agent_memory.models import (
    ActionType,
    Message,
    MessageRole,
    MemoryDecisionLog,
    MemoryType,
)
from agent_memory.short_term import ShortTermMemory
from agent_memory.token_utils import estimate_tokens, estimate_message_tokens


class MemoryManager:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        conversation_id: Optional[str] = None,
        persistence_dir: Optional[str] = None,
    ):
        self.config = config or MemoryConfig.default()
        self.conversation_id = conversation_id or self._generate_conversation_id()
        self.persistence_dir = persistence_dir
        
        self.short_term = ShortTermMemory(self.config.short_term)
        self.conversation = ConversationMemory(self.config.conversation)
        self.long_term = LongTermVectorMemory(self.config.long_term)
        
        self._last_log: Optional[MemoryDecisionLog] = None

    def add_message(
        self,
        role: str | MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryDecisionLog:
        role_enum = role if isinstance(role, MessageRole) else MessageRole(role)
        message = Message(
            role=role_enum,
            content=content,
            metadata=metadata or {},
        )
        
        log = MemoryDecisionLog(conversation_id=self.conversation_id)
        log.short_term_count = self.short_term.get_message_count()
        log.conversation_count = self.conversation.get_fact_count()
        log.long_term_count = self.long_term.get_document_count()
        
        log.add_action(
            action=ActionType.ADD_STM,
            description="添加消息到短时记忆",
            input_summary=f"[{role_enum.value}] {content[:50]}...",
            output_summary="消息已添加到短时记忆队列",
            details={
                "role": role_enum.value,
                "content_length": len(content),
                "estimated_tokens": estimate_message_tokens(message),
            },
        )
        
        did_evict, evicted_messages = self.short_term.add_message(message)
        
        if did_evict and evicted_messages:
            log.add_action(
                action=ActionType.EVICT_STM,
                description="短时记忆超出阈值，淘汰旧消息",
                input_summary=f"淘汰了 {len(evicted_messages)} 条消息",
                output_summary="消息已从短时记忆移除，将进行摘要处理",
                details={
                    "evicted_count": len(evicted_messages),
                    "evicted_roles": [m.role.value for m in evicted_messages],
                },
            )
            
            if self.config.conversation.enable_summarization:
                summary = self.conversation.summarize_messages(evicted_messages)
                if summary:
                    self.conversation.add_summary(summary)
                    log.add_action(
                        action=ActionType.SUMMARIZE,
                        description="将淘汰的消息压缩为摘要",
                        input_summary=f"{len(evicted_messages)} 条消息内容",
                        output_summary=f"生成摘要: {summary[:100]}...",
                        details={
                            "summary_length": len(summary),
                            "summary_tokens": estimate_tokens(summary),
                        },
                    )
            
            for msg in evicted_messages:
                if msg.role == MessageRole.USER and len(msg.content) > 20:
                    fact = f"用户提及: {msg.content}"
                    self.conversation.add_key_fact(
                        fact=fact,
                        source=f"conversation:{self.conversation_id}",
                        importance=0.8,
                    )
        
        self._last_log = log
        self._persist_if_needed()
        
        return log

    def build_context(
        self,
        query: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> tuple[str, MemoryDecisionLog]:
        budget = token_budget or self.config.default_token_budget
        
        log = MemoryDecisionLog(conversation_id=self.conversation_id)
        log.token_budget_initial = budget
        log.short_term_count = self.short_term.get_message_count()
        log.conversation_count = self.conversation.get_fact_count()
        log.long_term_count = self.long_term.get_document_count()
        
        remaining_budget = budget
        context_parts: dict[str, str] = {
            "long_term_context": "",
            "conversation_context": "",
            "short_term_context": "",
        }
        
        if query:
            retrieval_results = self.long_term.retrieve(
                query=query,
                top_k=self.config.long_term.top_k,
            )
            
            if retrieval_results:
                formatted_ltm = self.long_term.format_for_context(
                    retrieval_results,
                    include_score=True,
                )
                ltm_tokens = estimate_tokens(formatted_ltm)
                
                if ltm_tokens <= remaining_budget * 0.3:
                    context_parts["long_term_context"] = formatted_ltm
                    remaining_budget -= ltm_tokens
                    
                    log.add_action(
                        action=ActionType.RETRIEVE_LTM,
                        description="从长时记忆检索相关知识",
                        input_summary=f"查询: {query[:50]}...",
                        output_summary=f"检索到 {len(retrieval_results)} 条相关结果",
                        details={
                            "query": query,
                            "results_count": len(retrieval_results),
                            "results": [
                                {"content": r[0].content[:100], "score": r[1]}
                                for r in retrieval_results
                            ],
                        },
                    )
        
        conversation_context = self.conversation.format_for_context(
            max_summary_tokens=int(remaining_budget * 0.2),
            max_fact_tokens=int(remaining_budget * 0.2),
        )
        
        if conversation_context:
            conv_tokens = estimate_tokens(conversation_context)
            if conv_tokens <= remaining_budget * 0.4:
                context_parts["conversation_context"] = conversation_context
                remaining_budget -= conv_tokens
        
        short_term_messages = self.short_term.get_messages(
            max_tokens=int(remaining_budget * 0.9)
        )
        
        if short_term_messages:
            stm_context = self.short_term.format_for_context(short_term_messages)
            context_parts["short_term_context"] = stm_context
            remaining_budget -= estimate_tokens(stm_context)
        
        template = self.config.context_format_template
        final_context = template.format(**context_parts)
        
        actual_tokens = estimate_tokens(final_context)
        log.token_budget_remaining = budget - actual_tokens
        
        log.add_action(
            action=ActionType.BUILD_CONTEXT,
            description="构建最终prompt上下文",
            input_summary=f"预算: {budget} tokens",
            output_summary=f"生成上下文: {actual_tokens} tokens, 剩余: {budget - actual_tokens} tokens",
            details={
                "initial_budget": budget,
                "actual_tokens": actual_tokens,
                "remaining_budget": budget - actual_tokens,
                "context_parts_lengths": {
                    "long_term": len(context_parts["long_term_context"]),
                    "conversation": len(context_parts["conversation_context"]),
                    "short_term": len(context_parts["short_term_context"]),
                },
            },
        )
        
        self._last_log = log
        self._persist_if_needed()
        
        return final_context, log

    def add_knowledge(
        self,
        content: str,
        source: Optional[str] = None,
        tags: Optional[list[str]] = None,
        importance: float = 1.0,
    ) -> MemoryDecisionLog:
        log = MemoryDecisionLog(conversation_id=self.conversation_id)
        
        doc_id, is_update = self.long_term.upsert(
            content=content,
            source=source,
            tags=tags,
            importance=importance,
        )
        
        dedup_count = self.long_term.deduplicate()
        
        log.add_action(
            action=ActionType.UPSERT_LTM,
            description="插入/更新长时记忆知识",
            input_summary=f"内容: {content[:100]}...",
            output_summary=f"文档ID: {doc_id}, 更新: {is_update}",
            details={
                "doc_id": doc_id,
                "is_update": is_update,
                "source": source,
                "tags": tags,
                "importance": importance,
            },
        )
        
        if dedup_count > 0:
            log.add_action(
                action=ActionType.DEDUPLICATE,
                description="检测并合并重复文档",
                input_summary=f"当前文档数: {self.long_term.get_document_count()}",
                output_summary=f"合并了 {dedup_count} 组重复文档",
                details={"merged_count": dedup_count},
            )
        
        self._last_log = log
        self._persist_if_needed()
        
        return log

    def correct_fact(
        self,
        incorrect_pattern: str,
        correction: Optional[str] = None,
    ) -> MemoryDecisionLog:
        log = MemoryDecisionLog(conversation_id=self.conversation_id)
        
        ltm_invalidated = self.long_term.invalidate(
            pattern=incorrect_pattern,
            mark_inactive=True,
            reduce_importance_to=0.1,
        )
        
        conv_invalidated = self.conversation.invalidate_fact(incorrect_pattern)
        
        log.add_action(
            action=ActionType.INVALIDATE,
            description="标记错误事实为过期",
            input_summary=f"匹配模式: {incorrect_pattern}",
            output_summary=f"长时记忆失效: {ltm_invalidated}, 对话事实失效: {conv_invalidated}",
            details={
                "pattern": incorrect_pattern,
                "ltm_invalidated": ltm_invalidated,
                "conv_invalidated": conv_invalidated,
            },
        )
        
        if correction:
            self.add_knowledge(
                content=correction,
                source=f"correction:{self.conversation_id}",
                importance=1.0,
            )
        
        self._last_log = log
        self._persist_if_needed()
        
        return log

    def reset_conversation(self, conversation_id: Optional[str] = None) -> None:
        self.conversation_id = conversation_id or self._generate_conversation_id()
        self.short_term.clear()
        self.conversation.clear()
        self._last_log = None

    def get_last_log(self) -> Optional[MemoryDecisionLog]:
        return self._last_log

    def save_state(self, file_path: str) -> None:
        state = {
            "conversation_id": self.conversation_id,
            "config": {
                "short_term": self.config.short_term.model_dump(),
                "conversation": self.config.conversation.model_dump(),
                "long_term": self.config.long_term.model_dump(),
            },
            "short_term": self.short_term.to_list(),
            "conversation": self.conversation.to_dict(),
            "long_term": self.long_term.to_dict(),
            "saved_at": datetime.now().isoformat(),
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, file_path: str) -> "MemoryManager":
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        config = MemoryConfig(
            short_term=state["config"]["short_term"],
            conversation=state["config"]["conversation"],
            long_term=state["config"]["long_term"],
        )
        
        manager = cls(
            config=config,
            conversation_id=state["conversation_id"],
        )
        
        from agent_memory.config import ShortTermConfig, ConversationConfig, LongTermConfig
        
        manager.short_term = ShortTermMemory.from_list(
            state["short_term"],
            config.short_term,
        )
        manager.conversation = ConversationMemory.from_dict(
            state["conversation"],
            config.conversation,
        )
        manager.long_term = LongTermVectorMemory.from_dict(
            state["long_term"],
            config.long_term,
        )
        
        return manager

    def _generate_conversation_id(self) -> str:
        return f"conv_{uuid.uuid4().hex[:12]}"

    def _persist_if_needed(self) -> None:
        if self.persistence_dir and self._last_log:
            runs_dir = os.path.join(self.persistence_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            
            log_file = os.path.join(runs_dir, f"{self.conversation_id}.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._last_log.to_dict(), ensure_ascii=False) + "\n")
