#!/usr/bin/env python3
"""
Azure Databricks + Anthropic Claude Opus 4.6 Prompt Cache 测试
==============================================================
测试两种 API 格式的 Prompt Cache 支持情况。

使用方法：
  pip install requests python-dotenv
  python test_databricks_anthropic_cache.py

测试结果:
  2026-03-10: OpenAI 兼容格式 cache 字段不存在; Anthropic 原生格式 ✅
  2026-04-03: OpenAI 兼容格式 (string content) cache 字段存在但值为 0;
              OpenAI 兼容格式 (array content + cache_control) ✅ cache 生效!
              Anthropic 原生格式 ✅ cache_creation + cache_read 均正常
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
    """测试 1: OpenAI 兼容格式 — 检测 cache 字段是否存在及是否生效"""
    print("=" * 60)
    print("  测试 1: OpenAI 兼容格式")
    print("  检测: response 中是否包含 cache 字段，以及 cache 是否生效")
    print("=" * 60)

    results = []  # 收集两次调用结果用于最终判定

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
            cc = usage.get("cache_creation_input_tokens")
            cr = usage.get("cache_read_input_tokens")
            has_cache_fields = cc is not None or cr is not None
            cc = cc or 0
            cr = cr or 0

            print(f"   ✅ {dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')}")
            print(f"   cache 字段存在: {'是' if has_cache_fields else '否'}")
            print(f"   cache_creation={cc} | cache_read={cr}")

            if i == 0 and cc > 0:
                print(f"   🎯 Cache WRITE 成功! {cc} tokens 已缓存")
            if i == 1 and cr > 0:
                print(f"   🎯 Cache READ 成功! {cr} tokens 从缓存读取")

            results.append({
                "has_fields": has_cache_fields,
                "cc": cc, "cr": cr,
            })
        else:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            results.append(None)

        if i == 0:
            time.sleep(CACHE_WAIT_SECONDS)

    # 汇总判定
    if len(results) == 2 and all(results):
        r1, r2 = results
        cache_effective = r1["cc"] > 0 or r2["cr"] > 0
        print(f"\n   📋 OpenAI 格式判定:")
        print(f"      cache 字段返回: {'✅ 是' if r1['has_fields'] else '❌ 否'}")
        if cache_effective:
            print(f"      cache 生效: ✅ (cache_creation={r1['cc']}, cache_read={r2['cr']})")
        else:
            print(f"      cache 生效: ❌ (字段值始终为 0，OpenAI 格式无法指定 cache_control)")


def test_openai_format_cache_control_no_header():
    """测试 2: OpenAI 兼容格式 + cache_control 但无 anthropic-beta header"""
    print("\n" + "=" * 60)
    print("  测试 2: OpenAI 兼容格式 + cache_control (无 beta header)")
    print("  检测: 有 cache_control 但不加 anthropic-beta header，cache 是否生效")
    print("=" * 60)

    results = []

    for i in range(2):
        label = "首次调用" if i == 0 else "第二次调用(测 cache)"
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": LONG_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
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
            cc = usage.get("cache_creation_input_tokens", 0) or 0
            cr = usage.get("cache_read_input_tokens", 0) or 0

            print(f"   ✅ {dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')}")
            print(f"   cache_creation={cc} | cache_read={cr}")

            if i == 0 and cc > 0:
                print(f"   🎯 Cache WRITE 成功! {cc} tokens 已缓存")
            elif i == 0 and cr > 0:
                print(f"   🎯 Cache 已存在, {cr} tokens 从缓存读取")
            if i == 1 and cr > 0:
                print(f"   🎯 Cache READ 成功! {cr} tokens 从缓存读取")

            results.append({"cc": cc, "cr": cr})
        else:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            results.append(None)

        if i == 0:
            time.sleep(CACHE_WAIT_SECONDS)

    if len(results) == 2 and all(results):
        r1, r2 = results
        cache_effective = r1["cc"] > 0 or r2["cr"] > 0
        print(f"\n   📋 cache_control + 无 beta header 判定:")
        if cache_effective:
            print(f"      cache 生效: ✅ (cache_creation={r1['cc']}, cache_read={r2['cr']})")
            print(f"      结论: 不需要 beta header，只需 cache_control 即可!")
        else:
            print(f"      cache 生效: ❌ (有 cache_control 但缺少 beta header)")


def test_openai_format_with_cache_control():
    """
    测试 3: OpenAI 兼容格式 + cache_control + anthropic-beta header
    完整组合: 数组 content + cache_control + beta header
    """
    print("\n" + "=" * 60)
    print("  测试 3: OpenAI 兼容格式 + cache_control + beta header")
    print("  关键: system content 用数组格式 + cache_control + beta header")
    print("=" * 60)

    headers_with_beta = {
        **HEADERS,
        "anthropic-beta": "prompt-caching-2024-07-31",
    }

    results = []

    for i in range(2):
        label = "首次调用" if i == 0 else "第二次调用(测 cache)"
        # ⭐ 关键: content 用数组格式代替字符串, 加 cache_control
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": LONG_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": QUESTION},
            ],
            "max_tokens": 150,
            "temperature": 0.1,
        }
        print(f"\n📡 {label}...")
        t0 = time.time()
        try:
            r = requests.post(URL, headers=headers_with_beta, json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return
        dt = time.time() - t0

        if r.status_code == 200:
            usage = r.json().get("usage", {})
            cc = usage.get("cache_creation_input_tokens", 0) or 0
            cr = usage.get("cache_read_input_tokens", 0) or 0

            print(f"   ✅ {dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')}")
            print(f"   cache_creation={cc} | cache_read={cr}")

            if i == 0 and cc > 0:
                print(f"   🎯 Cache WRITE 成功! {cc} tokens 已缓存")
            elif i == 0 and cr > 0:
                print(f"   🎯 Cache 已存在 (之前的运行已缓存), {cr} tokens 从缓存读取")
            if i == 1 and cr > 0:
                print(f"   🎯 Cache READ 成功! {cr} tokens 从缓存读取")
                print("   💰 这些 tokens 按缓存价格计费，大幅节省!")

            results.append({"cc": cc, "cr": cr})
        else:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            results.append(None)

        if i == 0:
            time.sleep(CACHE_WAIT_SECONDS)

    if len(results) == 2 and all(results):
        r1, r2 = results
        cache_effective = r1["cc"] > 0 or r2["cr"] > 0
        print(f"\n   📋 OpenAI 格式 + cache_control 判定:")
        if cache_effective:
            print(f"      cache 生效: ✅ (cache_creation={r1['cc']}, cache_read={r2['cr']})")
            print(f"      结论: OpenAI 格式也能支持 cache! 只需 content 用数组格式 + cache_control")
        else:
            print(f"      cache 生效: ❌")


def test_openai_format_with_beta_header_only():
    """测试 4: OpenAI 兼容格式 + anthropic-beta header (但 content 仍为字符串)"""
    print("\n" + "=" * 60)
    print("  测试 4: OpenAI 兼容格式 + beta header (无 cache_control)")
    print("  检测: 仅添加 header、不改 content 格式，cache 是否生效")
    print("=" * 60)

    headers_with_beta = {
        **HEADERS,
        "anthropic-beta": "prompt-caching-2024-07-31",
    }

    results = []

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
            r = requests.post(URL, headers=headers_with_beta, json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return
        dt = time.time() - t0

        if r.status_code == 200:
            usage = r.json().get("usage", {})
            cc = usage.get("cache_creation_input_tokens", 0) or 0
            cr = usage.get("cache_read_input_tokens", 0) or 0

            print(f"   ✅ {dt:.2f}s | prompt_tokens={usage.get('prompt_tokens')}")
            print(f"   cache_creation={cc} | cache_read={cr}")

            results.append({"cc": cc, "cr": cr})
        else:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            results.append(None)

        if i == 0:
            time.sleep(CACHE_WAIT_SECONDS)

    if len(results) == 2 and all(results):
        r1, r2 = results
        cache_effective = r1["cc"] > 0 or r2["cr"] > 0
        print(f"\n   📋 beta header + 无 cache_control 判定:")
        if cache_effective:
            print(f"      cache 生效: ✅ (cache_creation={r1['cc']}, cache_read={r2['cr']})")
        else:
            print(f"      cache 生效: ❌ (仅添加 header 不够，还需要 cache_control 指令)")


def test_anthropic_native_format():
    """
    测试 5: Anthropic 原生消息格式 via /invocations
    关键点:
      1. 用 Anthropic 的 system 数组格式 (不是 OpenAI 的 role: system)
      2. 在 system block 中加 cache_control: {"type": "ephemeral"}
      3. Header 中加 anthropic-beta: prompt-caching-2024-07-31
      4. 仍然发到 /invocations 端点 (不是 /anthropic/v1/messages)
    """
    print("\n" + "=" * 60)
    print("  测试 5: Anthropic 原生格式 via /invocations")
    print("  预期: 首次 cache_creation>0, 第二次 cache_read>0")
    print("=" * 60)

    headers_with_beta = {
        **HEADERS,
        "anthropic-beta": "prompt-caching-2024-07-31",
    }

    results = []

    for i in range(2):
        label = "首次调用 (cache write)" if i == 0 else "第二次调用 (cache read)"

        payload = {
            "anthropic_version": "2023-06-01",
            "max_tokens": 150,
            "temperature": 0.1,
            "system": [
                {
                    "type": "text",
                    "text": LONG_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
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
            cc = usage.get("cache_creation_input_tokens", 0) or 0
            cr = usage.get("cache_read_input_tokens", 0) or 0
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

            if data.get("content"):
                answer = data["content"][0].get("text", "")
                print(f"   📝 回答: {answer[:200]}")

            results.append({"cc": cc, "cr": cr})
        else:
            print(f"   ❌ {r.status_code}: {r.text[:500]}")
            results.append(None)

        if i == 0:
            print(f"   ⏳ 等待 {CACHE_WAIT_SECONDS} 秒让 cache 生效...")
            time.sleep(CACHE_WAIT_SECONDS)

    if len(results) == 2 and all(results):
        r1, r2 = results
        cache_effective = r1["cc"] > 0 or r2["cr"] > 0
        print(f"\n   📋 Anthropic 原生格式判定:")
        if cache_effective:
            print(f"      cache 生效: ✅ (cache_creation={r1['cc']}, cache_read={r2['cr']})")
        else:
            print(f"      cache 生效: ❌")


def main():
    print(f"🔧 Databricks Host: {DATABRICKS_HOST}")
    print(f"🔧 Endpoint: {ENDPOINT}")
    print(f"📝 System prompt: {len(LONG_SYSTEM)} 字符\n")

    test_openai_format()
    test_openai_format_cache_control_no_header()
    test_openai_format_with_cache_control()
    test_openai_format_with_beta_header_only()
    test_anthropic_native_format()

    print("\n" + "=" * 60)
    print("  📋 总结")
    print("=" * 60)
    print("""
  ┌─────────┬────────────────────┬──────────────────────────┬──────────────┬────────┐
  │  测试   │  API 格式          │  cache_control           │  beta header │  结果  │
  ├─────────┼────────────────────┼──────────────────────────┼──────────────┼────────┤
  │  1      │  OpenAI 兼容       │  ❌ (string content)     │  ❌          │  ❌    │
  │  2      │  OpenAI 兼容       │  ✅ (array content)      │  ❌          │  ✅    │
  │  3      │  OpenAI 兼容       │  ✅ (array content)      │  ✅          │  ✅    │
  │  4      │  OpenAI 兼容       │  ❌ (string content)     │  ✅          │  ❌    │
  │  5      │  Anthropic 原生    │  ✅ (system array)       │  ✅          │  ✅    │
  └─────────┴────────────────────┴──────────────────────────┴──────────────┴────────┘

  💡 结论:
     唯一必要条件: cache_control: {"type": "ephemeral"}
     anthropic-beta header: 非必须 (测试 2 证明)
     OpenAI 兼容格式: 将 system content 从字符串改为数组即可使用 cache
     示例: {"role": "system", "content": [
              {"type": "text", "text": "...",
               "cache_control": {"type": "ephemeral"}}
           ]}
    """)


if __name__ == "__main__":
    main()
