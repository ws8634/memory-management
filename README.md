# Agent 混合记忆管理系统

一个"可运行、可测试、可演示"的Agent混合记忆管理最小实现，同时支持短时记忆、对话记忆和长时向量记忆。

## 功能特性

- **短时记忆 (Short-term Memory)**: 上下文窗口内的消息滚动队列，支持基于token和消息数的双重阈值淘汰
- **对话记忆 (Conversation Memory)**: 支持消息摘要压缩和关键事实提取，淘汰的短时记忆会自动摘要存入
- **长时向量记忆 (Long-term Vector Memory)**: 基于TF-IDF的向量检索，支持文档插入、更新、去重、检索和权重衰减
- **统一接口**: `MemoryManager` 提供一致的API来管理所有记忆类型
- **可解释日志**: 每次操作生成结构化JSON日志，追踪所有记忆决策
- **持久化**: 支持记忆状态的保存和加载，日志自动写入JSONL文件

## 安装

### 方法一：使用pip安装（推荐）

```bash
pip install -e .
```

### 方法二：安装依赖并以模块运行

```bash
pip install -r requirements.txt
```

如果没有requirements.txt，可以手动安装依赖：

```bash
pip install pydantic scikit-learn numpy click pytest
```

## 快速开始

### 运行演示场景

运行演示将自动展示所有核心功能：

```bash
agent-memory
```

或者使用模块方式：

```bash
python -m agent_memory
```

演示内容包括：
1. ✅ 短时记忆滚动与淘汰触发
2. ✅ 淘汰消息自动摘要生成
3. ✅ 长时记忆知识检索
4. ✅ 事实纠错与权重调整
5. ✅ 上下文预算控制
6. ✅ 持久化保存与日志记录

### 交互模式

进入交互式对话模式：

```bash
agent-memory interactive
```

可用命令：
- `/add <知识内容>` - 添加知识到长时记忆
- `/correct <错误内容>` - 纠错并标记事实
- `/context <查询>` - 构建并查看上下文
- `/status` - 查看当前记忆状态
- `/reset` - 重置对话
- `/save <文件>` - 保存状态
- `/load <文件>` - 加载状态
- `/help` - 显示帮助
- `/quit` - 退出

### 运行测试

```bash
pytest -v
```

带覆盖率报告：

```bash
pytest -v --cov=agent_memory
```

## 演示示例输出

运行 `agent-memory` 后，你将看到类似以下的输出：

```
============================================================
  Agent混合记忆管理系统 - 演示模式
============================================================

📚 预加载长时记忆知识...
✅ 已加载3条知识到长时记忆
------------------------------------------------------------

📝 阶段1: 短时记忆滚动演示

用户 [1]: 你好，我想了解一下中国的城市。
助手: 这是关于您问题的回复。当前短时记忆有 2 条消息。
   📊 状态: 短时记忆=2条, 对话摘要=0条

...

用户 [6]: 谢谢你的推荐，我还想了解一下Python。
⚠️  [触发] 短时记忆已满，淘汰旧消息并生成摘要
📝  [摘要] 生成摘要: 用户询问: 你好，我想了解一下中国的城市。 | 助手回复: 这是关于您问题的回复...
   📊 状态: 短时记忆=2条, 对话摘要=1条

------------------------------------------------------------

🔍 阶段2: 长时记忆检索演示

查询: 北京作为首都有什么特点？
✅ [检索] 找到 1 条相关知识
   📄 北京是中华人民共和国的首都，位于中国北部，是政治、文化中心。 (相关度: 1.00)

查询: Python是什么时候创建的？
✅ [检索] 找到 1 条相关知识
   📄 Python是一种高级编程语言，由Guido van Rossum于1991年创建。 (相关度: 1.00)

------------------------------------------------------------

🔧 阶段3: 事实纠错与权重调整演示

⚠️  已添加错误知识: '北京是中国的经济中心'

查询纠错前: 北京是经济中心吗？
   检索到: 北京是中国的经济中心。... (score: 1.00)
   检索到: 北京是中华人民共和国的首都... (score: 0.70)

🔄 执行纠错: 标记'北京是中国的经济中心'为错误...

📋 记忆决策日志:
  [INVALIDATE] 标记错误事实为过期
      输入: 匹配模式: 北京是中国的经济中心
      输出: 长时记忆失效: 1, 对话事实失效: 0

查询纠错后: 北京是经济中心吗？
   检索到: 北京是中国的政治和文化中心，上海是中国的经济中心。 (score: 0.90)
   检索到: 上海是中国最大的城市... (score: 0.65)

------------------------------------------------------------

📄 阶段4: 构建最终Prompt上下文

生成的上下文:
## 相关知识
1. 北京是中华人民共和国的首都... (相关度: 0.90)
2. 上海是中国最大的城市... (相关度: 0.65)

## 对话摘要
- 用户询问: 你好，我想了解一下中国的城市。 | 助手回复: ...
- 关键事实: 用户提及: 谢谢你的推荐，我还想了解一下Python。

## 当前对话
[user] 谢谢你的推荐，我还想了解一下Python。
[assistant] 这是关于您问题的回复...

📊 上下文统计:
   初始预算: 1500 tokens
   实际使用: 352 tokens
   剩余预算: 1148 tokens

------------------------------------------------------------

💾 阶段5: 持久化保存

✅ 记忆状态已保存到: memory_state.json
📋 决策日志已保存到: runs/demo_session_001.jsonl

📊 日志预览 (共15条记录):
   对话ID: demo_session_001
   动作数: 2
   短时记忆数: 2
   对话记忆数: 1
   长时记忆数: 5
------------------------------------------------------------

🎉 演示完成！

📁 生成的文件:
   - runs/demo_session_001.jsonl (决策日志)
   - memory_state.json (记忆状态)

💡 你可以:
   - 运行 'pytest' 执行单元测试
   - 运行 'agent-memory interactive' 进行交互式对话
```

## 三类记忆的区别与混合策略

### 记忆类型对比

| 特性 | 短时记忆 (STM) | 对话记忆 (CM) | 长时记忆 (LTM) |
|------|----------------|---------------|-----------------|
| **存储位置** | 内存队列 | 内存摘要/事实 | 向量索引 |
| **生命周期** | 对话内滚动 | 整个对话周期 | 跨对话持久化 |
| **容量限制** | Token/消息数阈值 | 摘要token限制 | 文档数限制 |
| **检索方式** | 顺序访问 | 全部加载 | 相似度检索 |
| **更新策略** | FIFO淘汰 | 摘要合并 | UPSERT + 去重 |

### 模块对应关系

```
MemoryManager (统一接口)
│
├── ShortTermMemory (src/agent_memory/short_term.py)
│   ├── 消息队列: list[Message]
│   ├── 淘汰策略: 超过 eviction_threshold 时触发
│   └── 配置项: max_messages, max_tokens, eviction_threshold
│
├── ConversationMemory (src/agent_memory/conversation.py)
│   ├── summaries[]: 压缩的对话摘要
│   ├── key_facts[]: 提取的关键事实
│   ├── 压缩方法: summarize_messages() 提取关键点
│   └── 配置项: max_summary_tokens, compression_threshold
│
└── LongTermVectorMemory (src/agent_memory/long_term.py)
    ├── 向量化: TfidfVectorizer (可替换为真实embedding)
    ├── 相似度: cosine_similarity
    ├── 去重: 基于Jaccard相似度阈值合并
    ├── 权重: importance 字段影响检索排序
    └── 配置项: top_k, similarity_threshold, deduplication_threshold
```

### 混合策略工作流

```
用户输入 add_message(role, content)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 1: 写入短时记忆 (ShortTermMemory)             │
│  - 估算消息token数                                    │
│  - 检查是否超过阈值 (消息数或token数)                 │
└─────────────────────────────────────────────────────┘
         │ 超过阈值?
    是 ──┴── 否
    │         │
    ▼         │
┌─────────────────────────────────────────────────────┐
│  Step 2: 触发淘汰与压缩                              │
│  - 从短时记忆队头移除消息 (EVICT_STM)               │
│  - 调用 summarize_messages() 生成摘要               │
│  - 摘要存入 ConversationMemory.summaries[]         │
│  - 关键事实提取存入 ConversationMemory.key_facts[] │
└─────────────────────────────────────────────────────┘
         │
         ▼
用户调用 build_context(query, token_budget)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: 长时记忆检索 (LongTermVectorMemory)        │
│  - 使用 query 进行 TF-IDF 向量化                     │
│  - 余弦相似度计算，取 top_k 结果                      │
│  - 结果按 importance 权重调整排序                     │
│  - 消耗预算约 30%                                     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: 对话记忆加载 (ConversationMemory)          │
│  - 加载活跃的对话摘要 (按importance排序)             │
│  - 加载活跃的关键事实                                 │
│  - 消耗预算约 40%                                     │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: 短时记忆加载 (ShortTermMemory)             │
│  - 从队尾取最新消息 (保持对话连贯性)                  │
│  - 直至剩余预算耗尽                                   │
│  - 消耗剩余预算约 90%                                 │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Step 6: 生成最终上下文                              │
│  - 按模板拼接: 长时记忆 → 对话记忆 → 短时记忆         │
│  - 返回 context 字符串和决策日志                      │
└─────────────────────────────────────────────────────┘
```

### 纠错与去重策略

**去重策略 (LongTermMemory.deduplicate()):**
- 文档内容采用TF-IDF向量化
- 计算余弦相似度矩阵
- 相似度 > deduplication_threshold (默认0.85) 视为重复
- 合并时保留较新文档，合并tags和metadata

**纠错策略 (MemoryManager.correct_fact()):**
1. 标记匹配内容为 `is_active = False`
2. 降低 importance 至原来的 30%
3. 可选地插入正确的修正内容
4. 检索时非活跃文档不会被返回，低权重文档排在后面

## 项目结构

```
13-agent-memory-management/
├── src/
│   └── agent_memory/
│       ├── __init__.py       # 包导出
│       ├── models.py         # 数据模型 (Message, MemoryItem, ActionLog)
│       ├── config.py         # 配置类 (MemoryConfig, ShortTermConfig等)
│       ├── token_utils.py    # Token估算工具
│       ├── short_term.py     # 短时记忆实现
│       ├── conversation.py   # 对话记忆实现
│       ├── long_term.py      # 长时向量记忆实现
│       ├── manager.py        # MemoryManager统一接口
│       └── cli.py            # 命令行界面
├── tests/
│   └── test_memory_manager.py  # 单元测试
├── runs/                     # 运行时日志 (自动生成)
│   └── <conversation_id>.jsonl
├── pyproject.toml            # 项目配置
├── README.md                 # 本文档
└── memory_state.json         # 持久化状态 (演示后生成)
```

## API 参考

### MemoryManager 核心方法

```python
from agent_memory import MemoryManager, MessageRole

manager = MemoryManager(
    conversation_id="my_conv_001",
    persistence_dir="./data",
)

# 添加消息
log = manager.add_message(
    role=MessageRole.USER,  # 或 "user", "assistant", "system"
    content="你好，我想了解Python",
    metadata={"source": "user_input"},
)

# 添加长时知识
log = manager.add_knowledge(
    content="Python是一种高级编程语言",
    source="维基百科",
    tags=["编程", "Python"],
    importance=0.9,
)

# 构建上下文
context, log = manager.build_context(
    query="Python是什么",
    token_budget=4000,
)

# 纠错
log = manager.correct_fact(
    incorrect_pattern="错误的事实",
    correction="正确的事实",
)

# 重置对话
manager.reset_conversation(new_conversation_id)

# 保存/加载状态
manager.save_state("state.json")
manager = MemoryManager.load_state("state.json")
```

### 日志结构

每次操作返回的 `MemoryDecisionLog` 可转换为JSON：

```json
{
  "conversation_id": "demo_session_001",
  "timestamp": "2026-04-30T10:30:00.123456",
  "actions": [
    {
      "action": "ADD_STM",
      "description": "添加消息到短时记忆",
      "input_summary": "[user] 你好...",
      "output_summary": "消息已添加",
      "details": {
        "role": "user",
        "content_length": 100,
        "estimated_tokens": 50
      }
    }
  ],
  "token_budget_initial": 4000,
  "token_budget_remaining": 3500,
  "short_term_count": 5,
  "conversation_count": 2,
  "long_term_count": 10
}
```

## 扩展说明

### 替换向量后端

当前实现使用 `sklearn.TfidfVectorizer` 作为默认向量化方案。要替换为真实的embedding模型（如OpenAI Embeddings、Sentence-Transformers），只需修改 `LongTermVectorMemory` 中的以下方法：

- `_fit_vectorizer()`: 初始化embedding模型
- `upsert()`: 文档插入时的向量化
- `retrieve()`: 查询时的向量化

接口设计已预留扩展空间，只需保持 `MemoryItem` 数据结构和 `retrieve()` 返回格式不变。

### 调整配置参数

```python
from agent_memory.config import MemoryConfig, ShortTermConfig

config = MemoryConfig(
    short_term=ShortTermConfig(
        max_messages=10,        # 最大消息数
        max_tokens=2000,        # 最大token数
        eviction_threshold=0.7, # 淘汰阈值比例
    ),
    default_token_budget=8000,
)

manager = MemoryManager(config=config)
```

## 许可证

MIT License
