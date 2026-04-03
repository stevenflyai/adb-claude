# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Test/exploration project for Azure Databricks (ADB) serving endpoints with Claude models. Scripts demonstrate calling Claude Opus 4.6 hosted on Databricks through both OpenAI-compatible and Anthropic-native API formats.

## Key Findings (documented in scripts)

- **Prompt caching**: Works in both OpenAI-compatible and Anthropic native formats. The only requirement is `cache_control: {"type": "ephemeral"}` in the content block. For OpenAI format, change system `content` from string to array: `{"role": "system", "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}]}`. The `anthropic-beta` header is not required.
- **Web search**: Anthropic native `web_search` tool type not supported by Databricks (returns 400). Use OpenAI function-calling format with client-side search instead.
- **MCP tools**: Work via OpenAI function-calling format — convert MCP tool definitions to OpenAI format, then forward tool calls back to MCP server.

## Architecture

All scripts are standalone — no shared modules or build system. Each script loads config from `.env` via `python-dotenv`.

| File | Purpose | API Format |
|------|---------|------------|
| `adb-sample.py` | Minimal chat completion | OpenAI-compatible |
| `adb-mcp.py` | MCP tool integration (GitHub server) with agentic loop | OpenAI-compatible |
| `databricks_websearch.py` | Client-side web search via function calling | Raw HTTP to `/invocations` |
| `test_databricks_anthropic_cache.py` | Prompt caching 2x2 matrix test (cache_control x beta header) | Both formats |
| `test_ttft.py` | TTFT (Time To First Token) benchmark with streaming | OpenAI-compatible |

## Running Scripts

```bash
# Install dependencies (conda is the default env manager per .vscode/settings.json)
pip install openai python-dotenv mcp
pip install requests  # for raw HTTP scripts
pip install ddgs      # optional, for DuckDuckGo web search

# Run any script directly
python adb-sample.py
python adb-mcp.py
python databricks_websearch.py
python test_databricks_anthropic_cache.py
```

## Environment Variables (via `.env`)

- `DATABRICKS_TOKEN` — Databricks personal access token
- `DATABRICKS_BASE_URL` — Databricks serving endpoint base URL (e.g. `https://<workspace>.azuredatabricks.net/serving-endpoints`)
- `GITHUB_PERSONAL_ACCESS_TOKEN` — GitHub PAT (used by `adb-mcp.py` for the GitHub MCP server)

## Databricks API Patterns

**OpenAI-compatible** (used by `adb-sample.py`, `adb-mcp.py`): Use the `openai` Python SDK with `base_url` pointed at Databricks. Model name: `databricks-claude-opus-4-6`.

**Anthropic native via `/invocations`** (used by `test_databricks_anthropic_cache.py`): POST directly to `{base_url}/{endpoint}/invocations` with Anthropic Messages API payload format (`anthropic_version`, `system` as array, `messages`). Supports prompt caching, though OpenAI-compatible format now supports it too (see prompt caching finding above).
