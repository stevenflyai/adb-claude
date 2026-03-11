#!/usr/bin/env python3
"""
Azure Databricks + Claude Opus 4.6 Web Search 测试
===================================================
测试结果 (2026-03-10):
  ❌ Anthropic 原生 server-side web_search → 400 "Missing 'function'"
     (Databricks 不支持 type: web_search_20250305)
  ✅ OpenAI function calling + 客户端搜索 → 完美工作
     (模型发 tool_calls → 你搜索 → 回传结果 → 模型生成回答)

使用方法:
  pip install requests
  # 如需真实搜索: pip install ddgs  (DuckDuckGo)
  python test_databricks_websearch.py
"""

import requests
import json
import time
import os

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置 - 通过 .env 文件或环境变量设置
# ============================================================
DATABRICKS_HOST = os.environ["DATABRICKS_BASE_URL"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
ENDPOINT = "databricks-claude-opus-4-6"

URL = f"{DATABRICKS_HOST}/{ENDPOINT}/invocations"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# Tool 定义 — OpenAI function calling 格式
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for real-time information. Use when the user asks about current events, prices, news, or anything requiring up-to-date data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}


def do_web_search(query: str) -> str:
    """
    执行真实 web search。
    支持: DuckDuckGo (免费) / Bing Search API / Google Custom Search
    """
    print(f"   🔍 搜索: {query}")

    # 方式 1: DuckDuckGo (pip install ddgs)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if results:
            return "\n\n".join([
                f"Title: {r.get('title','')}\nURL: {r.get('href','')}\nSnippet: {r.get('body','')}"
                for r in results
            ])
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️  DuckDuckGo 失败: {e}")

    # 方式 2: Bing Search API (需要 Azure Cognitive Services key)
    # BING_KEY = "YOUR_BING_KEY"
    # r = requests.get("https://api.bing.microsoft.com/v7.0/search",
    #     headers={"Ocp-Apim-Subscription-Key": BING_KEY},
    #     params={"q": query, "count": 5}, timeout=10)
    # ...

    # Fallback: 返回提示
    return f"[搜索暂不可用] 请根据已有知识回答关于 '{query}' 的问题。"


def chat_with_search(user_message: str, max_turns: int = 3) -> str:
    """
    与 Claude Opus 4.6 对话，自动处理 web search tool calling loop。

    流程:
      1. 发送用户消息 + tool 定义
      2. 如果模型返回 tool_calls → 执行搜索 → 回传结果
      3. 重复直到模型返回最终回答
    """
    messages = [{"role": "user", "content": user_message}]

    print(f"\n💬 用户: {user_message}")
    print("-" * 50)

    for turn in range(max_turns):
        payload = {
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.1,
            "tools": [WEB_SEARCH_TOOL],
        }

        print(f"\n📡 第 {turn + 1} 轮请求...")
        t0 = time.time()
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        dt = time.time() - t0

        if r.status_code != 200:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            return f"Error: {r.status_code}"

        data = r.json()
        usage = data.get("usage", {})
        choice = data["choices"][0]
        msg = choice["message"]

        print(f"   ✅ {dt:.2f}s | in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")

        # 检查是否有 tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            # 把 assistant 消息（含 tool_calls）加入对话
            messages.append(msg)

            # 逐个执行 tool call
            for tc in tool_calls:
                func = tc["function"]
                args = json.loads(func["arguments"])
                query = args.get("query", "")

                # 执行搜索
                result = do_web_search(query)
                print(f"   📋 返回 {len(result)} 字符")

                # 回传 tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            continue

        # 没有 tool_calls → 最终回答
        answer = msg.get("content", "")
        print(f"\n🤖 Claude:\n{answer}")
        return answer

    return "达到最大 tool calling 轮数"


def main():
    print("=" * 60)
    print("  Databricks + Claude Opus 4.6 — Web Search 测试")
    print("=" * 60)

    # 测试: 需要搜索的实时问题
    chat_with_search("今天 NVIDIA 的股价是多少？最新市值是多少？")

    print("\n" + "=" * 60)
    print("  📋 总结")
    print("=" * 60)
    print("""
  Databricks Claude Opus 4.6 Web Search:

  ❌ Anthropic 原生 server-side web_search
     → Databricks 返回 400 "Missing 'function'"
     → 因为 Databricks 只接受 OpenAI 格式的 tool 定义

  ✅ OpenAI function calling + 客户端搜索
     → 模型通过 tool_calls 请求搜索 (自动生成 query)
     → 你的代码执行搜索 (DuckDuckGo/Bing/Google)
     → 搜索结果作为 tool result 回传
     → 模型基于搜索结果生成最终回答

  推荐搜索后端:
     免费: DuckDuckGo (pip install ddgs)
     付费: Bing Search API ($3/1k calls) 或 Google Custom Search
    """)


if __name__ == "__main__":
    main()
