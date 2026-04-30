import json
import os
import sys
from typing import Optional

import click

from agent_memory.config import MemoryConfig, ShortTermConfig
from agent_memory.manager import MemoryManager
from agent_memory.models import MessageRole


def print_header(title: str) -> None:
    click.echo("=" * 60)
    click.echo(f"  {title}")
    click.echo("=" * 60)


def print_separator() -> None:
    click.echo("-" * 60)


def print_log_actions(log) -> None:
    if log and log.actions:
        click.echo("\n📋 记忆决策日志:")
        for action in log.actions:
            click.echo(f"  [{action.action.value}] {action.description}")
            click.echo(f"      输入: {action.input_summary[:80]}")
            click.echo(f"      输出: {action.output_summary[:80]}")


def demo_scenario() -> None:
    print_header("Agent混合记忆管理系统 - 演示模式")
    
    work_dir = os.getcwd()
    runs_dir = os.path.join(work_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    config = MemoryConfig(
        short_term=ShortTermConfig(
            max_messages=5,
            max_tokens=500,
            eviction_threshold=0.6,
        ),
    )
    
    manager = MemoryManager(
        config=config,
        conversation_id="demo_session_001",
        persistence_dir=work_dir,
    )
    
    click.echo("\n📚 预加载长时记忆知识...")
    manager.add_knowledge(
        content="北京是中华人民共和国的首都，位于中国北部，是政治、文化中心。",
        source="百科知识",
        tags=["中国", "城市", "首都"],
        importance=0.9,
    )
    manager.add_knowledge(
        content="上海是中国最大的城市，位于长江入海口，是经济、金融中心。",
        source="百科知识",
        tags=["中国", "城市", "经济"],
        importance=0.9,
    )
    manager.add_knowledge(
        content="Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
        source="编程知识",
        tags=["编程", "语言", "Python"],
        importance=0.8,
    )
    click.echo("✅ 已加载3条知识到长时记忆")
    
    print_separator()
    click.echo("\n📝 阶段1: 短时记忆滚动演示\n")
    
    user_messages = [
        "你好，我想了解一下中国的城市。",
        "北京和上海哪个更适合旅游？",
        "北京有哪些著名的景点？",
        "故宫和长城哪个更值得去？",
        "上海有什么特色美食？",
        "谢谢你的推荐，我还想了解一下Python。",
    ]
    
    for i, msg in enumerate(user_messages, 1):
        click.echo(f"用户 [{i}]: {msg}")
        log = manager.add_message(MessageRole.USER, msg)
        
        if log.actions:
            for action in log.actions:
                if action.action.value == "EVICT_STM":
                    click.echo(f"⚠️  [触发] 短时记忆已满，淘汰旧消息并生成摘要")
                elif action.action.value == "SUMMARIZE":
                    click.echo(f"📝  [摘要] {action.output_summary[:100]}...")
        
        assistant_resp = f"这是关于您问题的回复。当前短时记忆有 {manager.short_term.get_message_count()} 条消息。"
        manager.add_message(MessageRole.ASSISTANT, assistant_resp)
        
        stm_count = manager.short_term.get_message_count()
        summary_count = manager.conversation.get_summary_count()
        click.echo(f"助手: {assistant_resp}")
        click.echo(f"   📊 状态: 短时记忆={stm_count}条, 对话摘要={summary_count}条\n")
    
    print_separator()
    click.echo("\n🔍 阶段2: 长时记忆检索演示\n")
    
    queries = [
        "北京作为首都有什么特点？",
        "Python是什么时候创建的？",
        "上海的经济地位如何？",
    ]
    
    for query in queries:
        click.echo(f"查询: {query}")
        context, log = manager.build_context(query=query, token_budget=2000)
        
        retrieved = False
        for action in log.actions:
            if action.action.value == "RETRIEVE_LTM" and action.details.get("results_count", 0) > 0:
                click.echo(f"✅ [检索] 找到 {action.details['results_count']} 条相关知识")
                for result in action.details.get("results", []):
                    click.echo(f"   📄 {result['content'][:100]}... (相关度: {result['score']:.2f})")
                retrieved = True
        
        if not retrieved:
            click.echo("❌ [检索] 未找到相关知识")
        click.echo()
    
    print_separator()
    click.echo("\n🔧 阶段3: 事实纠错与权重调整演示\n")
    
    manager.add_knowledge(
        content="北京是中国的经济中心。",
        source="错误信息",
        tags=["错误"],
        importance=0.7,
    )
    click.echo("⚠️  已添加错误知识: '北京是中国的经济中心'")
    
    query = "北京是经济中心吗？"
    click.echo(f"\n查询纠错前: {query}")
    _, log = manager.build_context(query=query)
    for action in log.actions:
        if action.action.value == "RETRIEVE_LTM":
            for result in action.details.get("results", []):
                click.echo(f"   检索到: {result['content'][:80]}... (score: {result['score']:.2f})")
    
    click.echo("\n🔄 执行纠错: 标记'北京是中国的经济中心'为错误...")
    log = manager.correct_fact(
        incorrect_pattern="北京是中国的经济中心",
        correction="北京是中国的政治和文化中心，上海是中国的经济中心。",
    )
    print_log_actions(log)
    
    click.echo(f"\n查询纠错后: {query}")
    _, log = manager.build_context(query=query)
    for action in log.actions:
        if action.action.value == "RETRIEVE_LTM":
            for result in action.details.get("results", []):
                click.echo(f"   检索到: {result['content'][:80]}... (score: {result['score']:.2f})")
    
    print_separator()
    click.echo("\n📄 阶段4: 构建最终Prompt上下文\n")
    
    final_context, log = manager.build_context(
        query="帮我规划一次中国城市之旅",
        token_budget=1500,
    )
    
    click.echo("生成的上下文:")
    click.echo(final_context)
    click.echo()
    click.echo(f"📊 上下文统计:")
    for action in log.actions:
        if action.action.value == "BUILD_CONTEXT":
            details = action.details
            click.echo(f"   初始预算: {details['initial_budget']} tokens")
            click.echo(f"   实际使用: {details['actual_tokens']} tokens")
            click.echo(f"   剩余预算: {details['remaining_budget']} tokens")
            part_lens = details["context_parts_lengths"]
            click.echo(f"   各部分长度: 长时={part_lens['long_term']}, 对话={part_lens['conversation']}, 短时={part_lens['short_term']}")
    
    print_separator()
    click.echo("\n💾 阶段5: 持久化保存\n")
    
    state_file = os.path.join(work_dir, "memory_state.json")
    manager.save_state(state_file)
    click.echo(f"✅ 记忆状态已保存到: {state_file}")
    
    log_file = os.path.join(runs_dir, "demo_session_001.jsonl")
    click.echo(f"📋 决策日志已保存到: {log_file}")
    
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            click.echo(f"\n📊 日志预览 (共{len(lines)}条记录):")
            if lines:
                last_log = json.loads(lines[-1])
                click.echo(f"   对话ID: {last_log['conversation_id']}")
                click.echo(f"   动作数: {len(last_log['actions'])}")
                click.echo(f"   短时记忆数: {last_log['short_term_count']}")
                click.echo(f"   对话记忆数: {last_log['conversation_count']}")
                click.echo(f"   长时记忆数: {last_log['long_term_count']}")
    
    print_separator()
    click.echo("\n🎉 演示完成！")
    click.echo("\n📁 生成的文件:")
    click.echo(f"   - runs/demo_session_001.jsonl (决策日志)")
    click.echo(f"   - memory_state.json (记忆状态)")
    click.echo("\n💡 你可以:")
    click.echo("   - 运行 'pytest' 执行单元测试")
    click.echo("   - 运行 'agent-memory interactive' 进行交互式对话")


def interactive_mode() -> None:
    print_header("Agent混合记忆管理系统 - 交互模式")
    
    work_dir = os.getcwd()
    manager = MemoryManager(
        persistence_dir=work_dir,
    )
    
    click.echo(f"\n对话ID: {manager.conversation_id}")
    click.echo("输入命令进行操作:")
    click.echo("  /add <知识内容>  - 添加知识到长时记忆")
    click.echo("  /correct <错误内容>  - 纠错并标记事实")
    click.echo("  /context <查询>  - 构建并查看上下文")
    click.echo("  /status           - 查看当前记忆状态")
    click.echo("  /reset            - 重置对话")
    click.echo("  /save <文件>      - 保存状态")
    click.echo("  /load <文件>      - 加载状态")
    click.echo("  /help             - 显示帮助")
    click.echo("  /quit             - 退出")
    click.echo("直接输入文本则作为用户消息添加到对话")
    print_separator()
    
    while True:
        try:
            user_input = click.prompt("\n你", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\n再见！")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == "quit" or cmd == "exit":
                click.echo("再见！")
                break
            
            elif cmd == "help":
                click.echo("可用命令:")
                click.echo("  /add <知识内容>  - 添加知识到长时记忆")
                click.echo("  /correct <错误内容>  - 纠错并标记事实")
                click.echo("  /context <查询>  - 构建并查看上下文")
                click.echo("  /status           - 查看当前记忆状态")
                click.echo("  /reset            - 重置对话")
                click.echo("  /save <文件>      - 保存状态")
                click.echo("  /load <文件>      - 加载状态")
                click.echo("  /quit             - 退出")
            
            elif cmd == "add":
                if arg:
                    log = manager.add_knowledge(content=arg)
                    click.echo("✅ 已添加知识")
                    print_log_actions(log)
                else:
                    click.echo("❌ 请提供知识内容")
            
            elif cmd == "correct":
                if arg:
                    log = manager.correct_fact(incorrect_pattern=arg)
                    click.echo("🔄 已执行纠错")
                    print_log_actions(log)
                else:
                    click.echo("❌ 请提供错误内容模式")
            
            elif cmd == "context":
                query = arg if arg else None
                context, log = manager.build_context(query=query)
                click.echo("\n生成的上下文:")
                click.echo(context)
                click.echo()
                print_log_actions(log)
            
            elif cmd == "status":
                click.echo("\n📊 记忆状态:")
                click.echo(f"   短时记忆: {manager.short_term.get_message_count()} 条消息, {manager.short_term.get_token_count()} tokens")
                click.echo(f"   对话记忆: {manager.conversation.get_summary_count()} 条摘要, {manager.conversation.get_fact_count()} 个事实")
                click.echo(f"   长时记忆: {manager.long_term.get_document_count()} 条知识 (活跃: {manager.long_term.get_document_count(include_inactive=False)})")
            
            elif cmd == "reset":
                manager.reset_conversation()
                click.echo(f"✅ 对话已重置，新对话ID: {manager.conversation_id}")
            
            elif cmd == "save":
                if arg:
                    manager.save_state(arg)
                    click.echo(f"✅ 状态已保存到: {arg}")
                else:
                    click.echo("❌ 请提供文件路径")
            
            elif cmd == "load":
                if arg and os.path.exists(arg):
                    manager = MemoryManager.load_state(arg)
                    manager.persistence_dir = work_dir
                    click.echo(f"✅ 状态已从 {arg} 加载")
                    click.echo(f"   对话ID: {manager.conversation_id}")
                else:
                    click.echo("❌ 文件不存在")
            
            else:
                click.echo(f"❌ 未知命令: {cmd}，输入 /help 查看帮助")
        
        else:
            log = manager.add_message(MessageRole.USER, user_input)
            click.echo("✅ 消息已添加")
            print_log_actions(log)
            
            mock_response = f"这是对'{user_input[:30]}...'的回复"
            manager.add_message(MessageRole.ASSISTANT, mock_response)
            click.echo(f"助手: {mock_response}")


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        demo_scenario()


@main.command()
def demo():
    """运行演示场景"""
    demo_scenario()


@main.command()
def interactive():
    """进入交互模式"""
    interactive_mode()


if __name__ == "__main__":
    main()
