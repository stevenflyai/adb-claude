# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Test/exploration project for Azure Databricks (ADB) serving endpoints with Claude models. Scripts demonstrate calling Claude Opus 4.6 hosted on Databricks through both OpenAI-compatible and Anthropic-native API formats.

## Key Findings (documented in scripts)

- **Prompt caching**: Works in both OpenAI-compatible and Anthropic native formats. The only requirement is `cache_control: {"type": "ephemeral"}` in the content block. For OpenAI format, change system `content` from string to array: `{"role": "system", "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]}`. The `anthropic-beta` header is not required. System prompt must be >1024 tokens to trigger caching.
- **Web search**: Anthropic native `web_search` tool type not supported by Databricks (returns 400). Use OpenAI function-calling format with client-side search instead.
- **MCP tools**: Work via both OpenAI function-calling format (`adb-mcp.py`) and Anthropic native tool_use format (`adb-mcp-anthropic.py`). Convert MCP tool definitions to the target SDK format, then forward tool calls back to MCP server.
- **Code execution**: Databricks doesn't support Anthropic's native `code_execution` tool. Workaround: use OpenAI function-calling to have the model emit Python code, execute locally via `subprocess`, and return results.

## Architecture

All scripts are standalone — no shared modules or build system. Each script loads config from `.env` via `python-dotenv`.

| File | Purpose | API Format |
|------|---------|------------|
| `adb-sample.py` | Minimal chat completion | OpenAI-compatible |
| `adb-mcp.py` | MCP tool integration (GitHub server) with agentic loop | OpenAI-compatible |
| `adb-mcp-anthropic.py` | MCP tool integration using Anthropic SDK | Anthropic native |
| `databricks_websearch.py` | Client-side web search via function calling | Raw HTTP to `/invocations` |
| `test_databricks_code_exec.py` | Local code execution via function calling | Raw HTTP to `/invocations` |
| `test_databricks_anthropic_cache.py` | Prompt caching 5-test matrix (cache_control x beta header x API format) | Both formats |
| `test_ttft.py` | TTFT benchmark across prompt sizes (5 iterations each) | OpenAI-compatible |
| `test_ttft_stress.py` | TTFT stress test (100 runs) with charts | OpenAI-compatible |

## Running Scripts

```bash
# Install dependencies (conda is the default env manager per .vscode/settings.json)
pip install openai python-dotenv mcp anthropic
pip install requests  # for raw HTTP scripts
pip install ddgs      # optional, for DuckDuckGo web search
pip install matplotlib  # for TTFT stress test charts

# Run any script directly
python adb-sample.py
python adb-mcp.py
python test_ttft_stress.py
```

MCP scripts (`adb-mcp.py`, `adb-mcp-anthropic.py`) require `npx` (Node.js) to launch the GitHub MCP server subprocess.

## Environment Variables (via `.env`)

- `DATABRICKS_TOKEN` — Databricks personal access token
- `DATABRICKS_BASE_URL` — Databricks serving endpoint base URL (e.g. `https://<workspace>.azuredatabricks.net/serving-endpoints`)
- `GITHUB_PERSONAL_ACCESS_TOKEN` — GitHub PAT (used by MCP scripts for the GitHub MCP server)

## Databricks API Patterns

All three patterns use model name `databricks-claude-opus-4-6`.

**OpenAI-compatible** (used by `adb-sample.py`, `adb-mcp.py`, `test_ttft*.py`): Use the `openai` Python SDK with `base_url` pointed at Databricks.

**Anthropic native via SDK** (used by `adb-mcp-anthropic.py`): Use the `anthropic` Python SDK with `base_url` set to `{workspace_host}/serving-endpoints/anthropic` (the SDK appends `/v1/messages`). Auth gotcha: `api_key` must be set (use `"unused"`) since the SDK requires it, but actual auth is via `Authorization: Bearer {token}` in `default_headers`.

**Anthropic native via `/invocations`** (used by `test_databricks_anthropic_cache.py`, `databricks_websearch.py`, `test_databricks_code_exec.py`): POST directly to `{base_url}/{endpoint}/invocations` with Anthropic Messages API payload format (`anthropic_version`, `system` as array, `messages`). This endpoint also accepts OpenAI format payloads.
