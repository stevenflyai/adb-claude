#!/usr/bin/env python3
"""
Azure Databricks + Anthropic Claude Opus 4.6 Prompt Cache 测试
==============================================================
结论：通过 /invocations 端点使用 Anthropic 原生消息格式，可以完美支持 Prompt Cache。

使用方法：
  pip install requests python-dotenv
  python test_databricks_anthropic_cache.py

测试结果 (2026-03-10):
  - OpenAI 兼容格式: ❌ cache 字段始终为 0
  - Anthropic 原生格式 via /invocations: ✅ cache_creation + cache_read 均正常
"""

import os
import requests
import time

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置 - 通过 .env 文件或环境变量设置
# ============================================================
DATABRICKS_HOST = os.environ["DATABRICKS_BASE_URL"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
ENDPOINT = "databricks-claude-opus-4-6"

# ============================================================
# 构造长 system prompt (>1024 tokens 才能触发 cache)
# ============================================================
LONG_SYSTEM = "你是一位资深云计算架构师。\n" + "\n".join([
    f"知识条目 #{i}: NVIDIA Blackwell GB300 NVL72 架构包含72颗B300 GPU，通过NVLink 5.0互联，"
    f"每GPU 1.8TB/s双向带宽，整机架FP4算力1.44 exaFLOPS。Azure使用InfiniBand XDR，"
    f"GCP使用RoCE v2+ConnectX-8，AWS使用EFA v4+SRD协议。条目{i}的额外细节。"
    for i in range(1, 80)
])

QUESTION = "Azure和GCP的GB300网络区别？一句话。"
CACHE_WAIT_SECONDS = 3

# ============================================================
# 公共配置
# ============================================================
URL = f"{DATABRICKS_HOST}/{ENDPOINT}/invocations"
HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}


def test_openai_format():
    """测试 1: OpenAI 兼容格式 (不支持 cache)"""
    print("=" * 60)
    print("  测试 1: OpenAI 兼容格式")
    print("  预期: cache_creation=0, cache_read=0 (不支持)")
    print("=" * 60)

    for i in range(2):
        label = "首次调用" if i == 0 else "第二次调用(测 cache)"
        payload = {
            "messages": [
                {"role": "system", "content": LONG_SYSTEM},
                {"role": "user", "content": QUESTION},
            ],
            "max_tokens": 150,
            "temperature": 0.1,
        }
        print(f"\n📡 {label}...")
        t0 = time.time()
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return
        dt = time.time() - t0

        if r.status_code == 200:
            usage = r.json().get("usage", {})
            cc = usage.get("cache_creation_input_tokens", 0)
            cr = usage.get("cache_read_input_tokens", 0)
            print(f"   ✅ {dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')}")
            print(f"   cache_creation={cc} | cache_read={cr}")
            if cc == 0 and cr == 0:
                print("   ❌ Cache 未生效 (OpenAI 格式不支持)")
        else:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")

        if i == 0:
            time.sleep(CACHE_WAIT_SECONDS)


def test_anthropic_native_format():
    """
    测试 2: Anthropic 原生消息格式 via /invocations
    关键点:
      1. 用 Anthropic 的 system 数组格式 (不是 OpenAI 的 role: system)
      2. 在 system block 中加 cache_control: {"type": "ephemeral"}
      3. Header 中加 anthropic-beta: prompt-caching-2024-07-31
      4. 仍然发到 /invocations 端点 (不是 /anthropic/v1/messages)
    """
    print("\n" + "=" * 60)
    print("  测试 2: Anthropic 原生格式 via /invocations")
    print("  预期: 首次 cache_creation>0, 第二次 cache_read>0")
    print("=" * 60)

    # 加上 anthropic-beta header
    headers_with_beta = {
        **HEADERS,
        "anthropic-beta": "prompt-caching-2024-07-31",
    }

    for i in range(2):
        label = "首次调用 (cache write)" if i == 0 else "第二次调用 (cache read)"

        # ⭐ 关键: 使用 Anthropic 原生消息格式
        payload = {
            "anthropic_version": "2023-06-01",
            "max_tokens": 150,
            "temperature": 0.1,
            # ⭐ system 是数组格式，每个 block 可以加 cache_control
            "system": [
                {
                    "type": "text",
                    "text": LONG_SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # ⭐ 这是关键!
                }
            ],
            # messages 格式和 OpenAI 一样
            "messages": [{"role": "user", "content": QUESTION}],
        }

        print(f"\n📡 {label}...")
        t0 = time.time()
        try:
            r = requests.post(
                URL, headers=headers_with_beta, json=payload, timeout=120
            )
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return
        dt = time.time() - t0

        if r.status_code == 200:
            data = r.json()
            usage = data.get("usage", {})
            cc = usage.get("cache_creation_input_tokens", 0)
            cr = usage.get("cache_read_input_tokens", 0)
            # Anthropic 原生格式用 input_tokens/output_tokens
            it = usage.get("input_tokens", 0)
            ot = usage.get("output_tokens", 0)

            print(f"   ✅ {dt:.2f}s")
            print(f"   input_tokens={it} | output_tokens={ot}")
            print(f"   cache_creation={cc} | cache_read={cr}")

            if i == 0:
                if cc > 0:
                    print(f"   🎯 Cache WRITE 成功! {cc} tokens 已缓存")
                elif cr > 0:
                    print(f"   🎯 Cache 已存在 (之前的运行已缓存), {cr} tokens 从缓存读取")
                else:
                    print("   ❌ Cache WRITE 未生效 (cache_creation=0 且 cache_read=0)")
            if i == 1:
                if cr > 0:
                    print(f"   🎯 Cache READ 成功! {cr} tokens 从缓存读取")
                    print("   💰 这些 tokens 按缓存价格计费，大幅节省!")
                else:
                    print("   ❌ Cache READ 未生效")

            # 打印模型回答
            if data.get("content"):
                answer = data["content"][0].get("text", "")
                print(f"   📝 回答: {answer[:200]}")
        else:
            print(f"   ❌ {r.status_code}: {r.text[:500]}")

        if i == 0:
            print(f"   ⏳ 等待 {CACHE_WAIT_SECONDS} 秒让 cache 生效...")
            time.sleep(CACHE_WAIT_SECONDS)


def main():
    print(f"🔧 Databricks Host: {DATABRICKS_HOST}")
    print(f"🔧 Endpoint: {ENDPOINT}")
    print(f"📝 System prompt: {len(LONG_SYSTEM)} 字符\n")

    test_openai_format()
    test_anthropic_native_format()

    print("\n" + "=" * 60)
    print("  📋 总结")
    print("=" * 60)
    print("""
  OpenAI 兼容格式 (role: system):
    ❌ 不支持 Prompt Cache
    cache_creation 和 cache_read 始终为 0

  Anthropic 原生格式 via /invocations:
    ✅ 完美支持 Prompt Cache!
    关键要素:
      1. payload 用 Anthropic Messages API 格式
      2. system 用数组, 每个 block 加 cache_control
      3. Header 加 anthropic-beta: prompt-caching-2024-07-31
      4. 端点仍然是 /invocations (不需要特殊路径)
    """)


if __name__ == "__main__":
    main()
