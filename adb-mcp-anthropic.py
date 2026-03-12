"""
MCP tool integration using the Anthropic Python SDK against Databricks serving endpoints.

This is the Anthropic-native equivalent of adb-mcp.py (which uses the OpenAI SDK).
Uses Anthropic's native tool_use format instead of OpenAI function-calling format.

Key difference: the Anthropic SDK connects to Databricks via:
  - base_url pointing to the Databricks serving endpoint
  - api_key set to "unused" (auth is via Bearer token in headers)
  - Authorization header with the Databricks PAT
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

load_dotenv()

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
DATABRICKS_BASE_URL = os.environ.get("DATABRICKS_BASE_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")

# The Anthropic SDK appends /v1/messages to the base_url.
# Databricks exposes a dedicated /serving-endpoints/anthropic gateway for this.
# Derive the workspace host from the generic DATABRICKS_BASE_URL.
workspace_host = DATABRICKS_BASE_URL.split("/serving-endpoints")[0]
anthropic_base_url = f"{workspace_host}/serving-endpoints/anthropic"

client = anthropic.Anthropic(
    api_key="unused",
    base_url=anthropic_base_url,
    default_headers={
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    },
)


def mcp_tool_to_anthropic(tool):
    """Convert an MCP tool definition to Anthropic tool format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
    }


async def main():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_result.tools]
            tool_names = [t.name for t in tools_result.tools]
            print(f"Discovered {len(anthropic_tools)} tools from GitHub MCP server")
            print(f"Tools: {', '.join(tool_names)}\n")

            messages = [
                {
                    "role": "user",
                    "content": "Search for GitHub repositories owned by user 'stevenflyai' and list them.",
                }
            ]

            print("=== Step 1: Sending request to model ===")
            response = client.messages.create(
                model="databricks-claude-opus-4-6",
                messages=messages,
                tools=anthropic_tools,
                max_tokens=5000,
            )

            # Agentic loop: keep going while the model requests tool calls
            while response.stop_reason == "tool_use":
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                print(f"Model requested {len(tool_use_blocks)} tool call(s):")

                # Append the full assistant response (text + tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in tool_use_blocks:
                    print(f"  -> {block.name}({json.dumps(block.input, indent=2)})")

                    # Forward the tool call to the MCP server
                    result = await session.call_tool(block.name, block.input)
                    tool_content = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_content,
                    })

                messages.append({"role": "user", "content": tool_results})

                print("\n=== Sending tool results back to model ===")
                response = client.messages.create(
                    model="databricks-claude-opus-4-6",
                    messages=messages,
                    tools=anthropic_tools,
                    max_tokens=5000,
                )

            # Extract final text response
            print("\n=== Final Response ===")
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)


asyncio.run(main())
