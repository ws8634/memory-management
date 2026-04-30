import pytest

from agent_memory.config import MemoryConfig, ShortTermConfig, LongTermConfig, ConversationConfig
from agent_memory.manager import MemoryManager
from agent_memory.models import MessageRole, ActionType
from agent_memory.token_utils import estimate_tokens, estimate_message_tokens
from agent_memory.short_term import ShortTermMemory
from agent_memory.conversation import ConversationMemory
from agent_memory.long_term import LongTermVectorMemory


class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        text = "Hello world"
        tokens = estimate_tokens(text)
        assert tokens == 2
    
    def test_estimate_tokens_chinese(self):
        text = "你好世界"
        tokens = estimate_tokens(text)
        assert tokens == 4
    
    def test_estimate_tokens_mixed(self):
        text = "Hello 你好 world 世界"
        tokens = estimate_tokens(text)
        assert tokens == 6
    
    def test_estimate_empty_text(self):
        assert estimate_tokens("") == 0
    
    def test_estimate_message_tokens(self):
        from agent_memory.models import Message
        msg = Message(role=MessageRole.USER, content="Hello world")
        tokens = estimate_message_tokens(msg)
        assert tokens > 0


class TestShortTermMemory:
    def test_add_message(self):
        stm = ShortTermMemory()
        from agent_memory.models import Message
        msg = Message(role=MessageRole.USER, content="Test message")
        
        did_evict, evicted = stm.add_message(msg)
        
        assert did_evict is False
        assert len(evicted) == 0
        assert stm.get_message_count() == 1
    
    def test_eviction_on_message_count(self):
        config = ShortTermConfig(max_messages=3, eviction_threshold=0.8)
        stm = ShortTermMemory(config)
        from agent_memory.models import Message
        
        for i in range(5):
            msg = Message(role=MessageRole.USER, content=f"Message {i}")
            did_evict, evicted = stm.add_message(msg)
            
            if i >= 3:
                assert did_evict is True
                assert len(evicted) > 0
        
        assert stm.get_message_count() <= 3
    
    def test_eviction_on_token_count(self):
        config = ShortTermConfig(max_tokens=50, eviction_threshold=0.6)
        stm = ShortTermMemory(config)
        from agent_memory.models import Message
        
        long_content = "x" * 100
        for i in range(3):
            msg = Message(role=MessageRole.USER, content=long_content)
            did_evict, evicted = stm.add_message(msg)
        
        assert stm.get_token_count() <= 50
    
    def test_get_messages_with_token_limit(self):
        stm = ShortTermMemory()
        from agent_memory.models import Message
        
        msg1 = Message(role=MessageRole.USER, content="Short message")
        msg2 = Message(role=MessageRole.USER, content="This is a much longer message that contains many words and characters")
        msg3 = Message(role=MessageRole.USER, content="Another message that is also quite long with many words")
        
        stm.add_message(msg1)
        stm.add_message(msg2)
        stm.add_message(msg3)
        
        limited = stm.get_messages(max_tokens=20)
        assert len(limited) < 3
    
    def test_format_for_context(self):
        stm = ShortTermMemory()
        from agent_memory.models import Message
        
        stm.add_message(Message(role=MessageRole.USER, content="Hello"))
        stm.add_message(Message(role=MessageRole.ASSISTANT, content="Hi there"))
        
        formatted = stm.format_for_context()
        assert "[user] Hello" in formatted
        assert "[assistant] Hi there" in formatted
    
    def test_clear(self):
        stm = ShortTermMemory()
        from agent_memory.models import Message
        
        stm.add_message(Message(role=MessageRole.USER, content="Test"))
        stm.clear()
        
        assert stm.get_message_count() == 0
        assert stm.get_token_count() == 0


class TestConversationMemory:
    def test_add_summary(self):
        cm = ConversationMemory()
        cm.add_summary("This is a test summary")
        
        assert cm.get_summary_count() == 1
    
    def test_summary_token_limit(self):
        config = ConversationConfig(max_summary_tokens=100)
        cm = ConversationMemory(config)
        
        long_summary = "x" * 80
        cm.add_summary(long_summary)
        cm.add_summary(long_summary)
        
        assert cm.get_token_count() <= 100
    
    def test_add_key_fact(self):
        cm = ConversationMemory()
        cm.add_key_fact("User likes Python", source="conversation_1", importance=0.9)
        
        facts = cm.get_active_facts()
        assert len(facts) == 1
        assert facts[0]["fact"] == "User likes Python"
    
    def test_invalidate_fact(self):
        cm = ConversationMemory()
        cm.add_key_fact("北京是中国的经济中心", importance=1.0)
        
        invalidated = cm.invalidate_fact("北京是中国的经济中心")
        
        assert invalidated == 1
        active_facts = cm.get_active_facts()
        assert len(active_facts) == 0
    
    def test_summarize_messages(self):
        cm = ConversationMemory()
        from agent_memory.models import Message
        
        messages = [
            Message(role=MessageRole.USER, content="推荐一些旅游景点"),
            Message(role=MessageRole.ASSISTANT, content="我推荐故宫和长城，它们都在北京"),
        ]
        
        summary = cm.summarize_messages(messages)
        
        assert len(summary) > 0
        assert "用户询问" in summary or "旅游" in summary.lower()
    
    def test_format_for_context(self):
        cm = ConversationMemory()
        cm.add_summary("对话涉及旅游话题")
        cm.add_key_fact("用户对北京感兴趣")
        
        formatted = cm.format_for_context()
        
        assert "对话涉及旅游话题" in formatted
        assert "用户对北京感兴趣" in formatted


class TestLongTermVectorMemory:
    def test_upsert_new_document(self):
        ltm = LongTermVectorMemory()
        
        doc_id, is_update = ltm.upsert(
            content="Python是一种编程语言",
            source="维基百科",
            tags=["编程", "Python"],
        )
        
        assert is_update is False
        assert ltm.get_document_count() == 1
    
    def test_upsert_update_document(self):
        ltm = LongTermVectorMemory()
        
        doc_id1, is_update1 = ltm.upsert(content="Python是一种编程语言")
        doc_id2, is_update2 = ltm.upsert(content="Python是一种编程语言")
        
        assert doc_id1 == doc_id2
        assert is_update2 is True
        assert ltm.get_document_count() == 1
    
    def test_retrieve_basic(self):
        config = LongTermConfig(similarity_threshold=0.0)
        ltm = LongTermVectorMemory(config)
        
        ltm.upsert(content="Python programming language created by Guido")
        ltm.upsert(content="Java object oriented programming language")
        ltm.upsert(content="Beijing capital of China")
        
        results = ltm.retrieve(query="Python programming", top_k=2)
        
        assert len(results) > 0
        if results:
            assert "Python" in results[0][0].content
    
    def test_retrieve_consistency(self):
        config = LongTermConfig(similarity_threshold=0.0)
        ltm = LongTermVectorMemory(config)
        
        ltm.upsert(content="Python programming language")
        ltm.upsert(content="Java programming language")
        ltm.upsert(content="C++ compiled language")
        
        results1 = ltm.retrieve(query="programming language", top_k=2)
        results2 = ltm.retrieve(query="programming language", top_k=2)
        
        assert len(results1) == len(results2)
        if results1 and results2:
            assert results1[0][0].id == results2[0][0].id
    
    def test_invalidate_document(self):
        ltm = LongTermVectorMemory()
        
        ltm.upsert(content="北京是中国的经济中心", importance=1.0)
        ltm.upsert(content="上海是中国的金融中心", importance=1.0)
        
        invalidated = ltm.invalidate("北京是中国的经济中心")
        
        assert invalidated == 1
        active_count = ltm.get_document_count(include_inactive=False)
        assert active_count == 1
    
    def test_deduplication(self):
        config = LongTermConfig(deduplication_threshold=0.5, similarity_threshold=0.0)
        ltm = LongTermVectorMemory(config)
        
        ltm.upsert(content="Python programming language")
        ltm.upsert(content="Python programming language great")
        ltm.upsert(content="Java programming language")
        
        initial_count = ltm.get_document_count()
        merged = ltm.deduplicate()
        
        assert merged > 0
    
    def test_format_for_context(self):
        config = LongTermConfig(similarity_threshold=0.0)
        ltm = LongTermVectorMemory(config)
        ltm.upsert(content="First knowledge item", source="source1")
        ltm.upsert(content="Second knowledge item", source="source2")
        
        results = ltm.retrieve(query="knowledge item", top_k=2)
        formatted = ltm.format_for_context(results, include_score=True)
        
        assert "First" in formatted
        assert "Second" in formatted


class TestMemoryManager:
    def test_initialization(self):
        manager = MemoryManager()
        
        assert manager.conversation_id is not None
        assert manager.short_term is not None
        assert manager.conversation is not None
        assert manager.long_term is not None
    
    def test_add_message_basic(self):
        manager = MemoryManager()
        
        log = manager.add_message(MessageRole.USER, "Hello world")
        
        assert log.conversation_id == manager.conversation_id
        assert len(log.actions) > 0
        
        add_actions = [a for a in log.actions if a.action == ActionType.ADD_STM]
        assert len(add_actions) == 1
    
    def test_add_message_triggers_eviction(self):
        config = MemoryConfig(
            short_term=ShortTermConfig(
                max_messages=3,
                max_tokens=100,
                eviction_threshold=0.5,
            ),
            conversation=ConversationConfig(
                enable_summarization=True,
            ),
        )
        manager = MemoryManager(config)
        
        long_content = "这是一条很长的消息内容" * 5
        for i in range(10):
            log = manager.add_message(MessageRole.USER, f"{long_content} #{i}")
            
            evict_actions = [a for a in log.actions if a.action == ActionType.EVICT_STM]
            if len(evict_actions) > 0:
                summarize_actions = [a for a in log.actions if a.action == ActionType.SUMMARIZE]
                assert len(summarize_actions) > 0
                break
        
        assert manager.conversation.get_summary_count() > 0 or manager.short_term.get_message_count() <= 3
    
    def test_build_context_with_ltm_retrieval(self):
        config = MemoryConfig(
            long_term=LongTermConfig(similarity_threshold=0.0),
        )
        manager = MemoryManager(config)
        
        manager.add_knowledge(
            content="Python programming language created by Guido van Rossum in 1991",
            source="编程百科",
        )
        manager.add_knowledge(
            content="Java object oriented programming language by Sun Microsystems",
            source="编程百科",
        )
        
        context, log = manager.build_context(query="Python programming", token_budget=1000)
        
        retrieve_actions = [a for a in log.actions if a.action == ActionType.RETRIEVE_LTM]
        assert len(retrieve_actions) > 0
        assert "Python" in context
    
    def test_build_context_token_budget_respected(self):
        manager = MemoryManager()
        
        for i in range(20):
            manager.add_message(MessageRole.USER, f"这是第{i}条消息，内容是关于测试预算控制的。")
            manager.add_message(MessageRole.ASSISTANT, f"这是对第{i}条消息的回复。")
        
        budget = 500
        context, log = manager.build_context(token_budget=budget)
        
        actual_tokens = estimate_tokens(context)
        
        assert actual_tokens <= budget * 1.5
        assert log.token_budget_initial == budget
        assert log.token_budget_remaining is not None
    
    def test_add_knowledge_with_deduplication(self):
        manager = MemoryManager()
        
        log1 = manager.add_knowledge(content="Python是一种编程语言")
        log2 = manager.add_knowledge(content="Python是一种编程语言")
        
        upsert1 = [a for a in log1.actions if a.action == ActionType.UPSERT_LTM]
        upsert2 = [a for a in log2.actions if a.action == ActionType.UPSERT_LTM]
        
        assert len(upsert1) == 1
        assert len(upsert2) == 1
        assert upsert1[0].details.get("is_update") is False
        assert upsert2[0].details.get("is_update") is True
    
    def test_correct_fact_invalidates_old_knowledge(self):
        config = MemoryConfig(
            long_term=LongTermConfig(similarity_threshold=0.0, deduplication_threshold=0.95),
        )
        manager = MemoryManager(config)
        
        manager.add_knowledge(
            content="Beijing is the economic center of China",
            importance=1.0,
        )
        
        assert manager.long_term.get_document_count(include_inactive=False) == 1
        
        log = manager.correct_fact(
            incorrect_pattern="Beijing is the economic center",
            correction="Beijing is the political and cultural center of China. Shanghai is the economic center.",
        )
        
        invalidate_actions = [a for a in log.actions if a.action == ActionType.INVALIDATE]
        assert len(invalidate_actions) > 0
        assert invalidate_actions[0].details.get("ltm_invalidated", 0) > 0
        
        docs = manager.long_term.documents
        old_doc = next((d for d in docs if "economic center of China" in d.content), None)
        if old_doc:
            assert old_doc.is_active is False or old_doc.importance < 0.5
        
        assert manager.long_term.get_document_count(include_inactive=False) >= 1
    
    def test_reset_conversation(self):
        manager = MemoryManager()
        
        manager.add_message(MessageRole.USER, "测试消息1")
        manager.add_knowledge(content="测试知识")
        
        old_conv_id = manager.conversation_id
        manager.reset_conversation()
        
        assert manager.conversation_id != old_conv_id
        assert manager.short_term.get_message_count() == 0
        assert manager.conversation.get_summary_count() == 0
    
    def test_save_and_load_state(self, tmp_path):
        manager = MemoryManager()
        
        manager.add_message(MessageRole.USER, "测试消息")
        manager.add_knowledge(content="测试知识内容", source="test")
        manager.conversation.add_summary("测试摘要")
        manager.conversation.add_key_fact("测试事实")
        
        state_file = tmp_path / "state.json"
        manager.save_state(str(state_file))
        
        loaded_manager = MemoryManager.load_state(str(state_file))
        
        assert loaded_manager.conversation_id == manager.conversation_id
        assert loaded_manager.short_term.get_message_count() == manager.short_term.get_message_count()
        assert loaded_manager.conversation.get_summary_count() == manager.conversation.get_summary_count()
        assert loaded_manager.long_term.get_document_count() == manager.long_term.get_document_count()
    
    def test_log_generation(self):
        manager = MemoryManager()
        
        log = manager.add_message(MessageRole.USER, "测试日志生成")
        
        assert log is not None
        assert log.conversation_id == manager.conversation_id
        assert len(log.actions) > 0
        
        log_dict = log.to_dict()
        assert "conversation_id" in log_dict
        assert "actions" in log_dict
        assert "timestamp" in log_dict
        
        last_log = manager.get_last_log()
        assert last_log is not None
        assert last_log.conversation_id == log.conversation_id


class TestIntegration:
    def test_full_demo_scenario(self):
        config = MemoryConfig(
            short_term=ShortTermConfig(
                max_messages=5,
                max_tokens=300,
                eviction_threshold=0.6,
            ),
            long_term=LongTermConfig(
                top_k=2,
                similarity_threshold=0.0,
            ),
        )
        manager = MemoryManager(config=config)
        
        manager.add_knowledge(
            content="Beijing capital of China political cultural center",
            source="百科",
        )
        manager.add_knowledge(
            content="Shanghai largest city of China economic financial center",
            source="百科",
        )
        
        for i in range(10):
            manager.add_message(MessageRole.USER, f"问题{i}: 关于中国城市的信息")
            manager.add_message(MessageRole.ASSISTANT, f"回复{i}: 这里是关于中国城市的信息")
        
        context, log = manager.build_context(
            query="Beijing features",
            token_budget=800,
        )
        
        assert len(context) > 0
        assert log.token_budget_initial == 800
        
        actual_tokens = estimate_tokens(context)
        assert actual_tokens <= 1000
