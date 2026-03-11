#!/usr/bin/env python3
"""
Azure Databricks + Claude Opus 4.6 — Code Execution 引擎
=========================================================
原理：模型通过 function calling 返回 Python 代码 → 本地执行 → 回传结果 → 模型继续

测试结果 (2026-03-10):
  ✅ 复利计算: 模型自动生成计算器代码，精确输出结果
  ✅ 蒙特卡洛模拟: 生成完整模拟 + 统计分析 + 图表
  ✅ 多轮 tool loop: 遇到缺包会自动安装重试
  ✅ 生成图表: 保存到 /tmp/chart.png

使用方法:
  pip install requests numpy matplotlib python-dotenv --break-system-packages
  python test_databricks_code_exec.py
"""

import requests
import json
import time
import subprocess
import tempfile
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

# 执行输出最大字符数
MAX_OUTPUT_CHARS = 5000

# Code execution tool 定义
CODE_EXEC_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a sandboxed environment. "
            "Use for calculations, data analysis, chart generation, simulations, etc. "
            "Available packages: numpy, matplotlib. "
            "Use print() for output. Save charts to /tmp/chart.png. "
            "Do NOT use pandas (not installed). Use numpy instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what the code does",
                },
            },
            "required": ["code"],
        },
    },
}


def _make_sandbox_env() -> dict:
    """构建最小化的子进程环境，避免泄露 secrets"""
    safe_keys = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "SHELL"}
    env = {k: v for k, v in os.environ.items() if k in safe_keys}
    env["MPLBACKEND"] = "Agg"
    return env


def execute_python(code: str, timeout: int = 30) -> str:
    """在本地执行 Python 代码（注意：无真正沙箱，建议在容器中运行）"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_make_sandbox_env(),
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                # 只过滤 Python 的 DeprecationWarning / UserWarning 等标准警告行
                errors = [
                    line
                    for line in result.stderr.split("\n")
                    if line and not line.strip().startswith(("Warning:", "UserWarning:", "DeprecationWarning:"))
                ]
                if errors:
                    output += "\nSTDERR:\n" + "\n".join(errors)
            if result.returncode != 0 and not output.strip():
                output = f"Process exited with code {result.returncode}"
            return output.strip()[:MAX_OUTPUT_CHARS] or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out ({timeout}s limit)"
        except Exception as e:
            return f"ERROR: {e}"
        finally:
            os.unlink(f.name)


def chat_with_code(user_message: str, max_turns: int = 8) -> str:
    """
    与 Claude Opus 4.6 对话，自动处理 code execution tool loop。

    流程:
      1. 发送用户消息 + execute_python tool 定义
      2. 模型返回 tool_calls (Python 代码) → 本地执行 → 回传结果
      3. 重复直到模型给出最终回答
    """
    messages = [{"role": "user", "content": user_message}]

    print(f"\n{'='*60}")
    print(f"  💬 {user_message}")
    print(f"{'='*60}")

    for turn in range(max_turns):
        payload = {
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.1,
            "tools": [CODE_EXEC_TOOL],
        }

        print(f"\n📡 第 {turn + 1} 轮...")
        t0 = time.time()
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        dt = time.time() - t0

        if r.status_code != 200:
            print(f"   ❌ {r.status_code}: {r.text[:300]}")
            raise RuntimeError(f"API request failed: {r.status_code}")

        data = r.json()
        usage = data.get("usage", {})
        choice = data["choices"][0]
        msg = choice["message"]
        print(f"   ✅ {dt:.2f}s | in={usage.get('prompt_tokens',0)} out={usage.get('completion_tokens',0)}")

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as e:
                    print(f"   ❌ Failed to parse tool arguments: {e}")
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": f"ERROR: Malformed JSON arguments: {e}"}
                    )
                    continue
                code = args.get("code", "")
                desc = args.get("description", "")
                print(f"\n   🔧 {desc}")

                # 显示代码前 10 行
                lines = code.split("\n")
                for line in lines[:10]:
                    print(f"   │ {line}")
                if len(lines) > 10:
                    print(f"   │ ... ({len(lines)} lines)")

                # 执行
                result = execute_python(code)
                print(f"   📋 Output: {result[:500]}")

                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )
            continue

        # 最终回答
        answer = msg.get("content", "")
        print(f"\n🤖 回答:\n{answer[:1000]}")
        return answer

    print("⚠️  达到最大轮数")
    return messages[-1].get("content", "") if messages else ""


def main():
    print("🚀 Databricks + Claude Opus 4.6 — Code Execution Engine")
    print(f"🔧 Endpoint: {ENDPOINT}\n")

    # 测试 1: 精确计算
    chat_with_code(
        "投资100万人民币，年化8%，复利20年后是多少？每年追加10万呢？用Python精确计算。"
    )

    # 测试 2: 蒙特卡洛模拟
    chat_with_code(
        "用蒙特卡洛模拟估算圆周率π，分别用1万、10万、100万个随机点，比较精度和误差。"
    )

    # 测试 3: 数据分析
    chat_with_code(
        "生成252个交易日的模拟股票数据，计算20日和60日移动平均线，生成图表保存到/tmp/stock_chart.png。"
    )

    print(f"\n{'='*60}")
    print("  📋 总结")
    print(f"{'='*60}")
    print("""
  Databricks + Claude Opus 4.6 Code Execution:

  ❌ Anthropic 原生 code_execution_20250522 tool
     → Databricks 不支持 (只接受 OpenAI function 格式)

  ✅ OpenAI function calling + 本地 Python 执行
     → 模型生成代码 → subprocess 执行 → 回传结果
     → 多轮 loop: 遇错自动修复重试
     → 支持图表生成 (matplotlib → /tmp/*.png)

  优势 vs Anthropic 原生沙箱:
     ✅ 可以安装任意 pip 包
     ✅ 可以访问本地文件系统
     ✅ 可以访问网络 (如果需要)
     ✅ 在 Databricks cluster 上可以用 Spark
     ✅ 无容器时间/内存限制（你控制）
     ⚠️ 安全性需自己保障（建议在沙箱/容器中执行）
    """)


if __name__ == "__main__":
    main()
