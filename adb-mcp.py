import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
DATABRICKS_BASE_URL = os.environ.get('DATABRICKS_BASE_URL')

# GitHub personal access token for the MCP server
# https://github.com/settings/tokens
GITHUB_TOKEN = os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN')

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL,
)


def mcp_tool_to_openai(tool):
    """Convert an MCP tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
        },
    }


async def main():
    # Launch the GitHub MCP server as a subprocess via npx
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools exposed by the GitHub MCP server
            tools_result = await session.list_tools()
            openai_tools = [mcp_tool_to_openai(t) for t in tools_result.tools]
            tool_names = [t.name for t in tools_result.tools]
            print(f"Discovered {len(openai_tools)} tools from GitHub MCP server")
            print(f"Tools: {', '.join(tool_names)}\n")

            messages = [
                {
                    "role": "user",
                    "content": "Search for GitHub repositories owned by user 'stevenflyai' and list them.",
                }
            ]

            print("=== Step 1: Sending request to model ===")
            response = client.chat.completions.create(
                model="databricks-claude-opus-4-6",
                messages=messages,
                tools=openai_tools,
                max_tokens=5000,
            )

            assistant_message = response.choices[0].message

            # Agentic loop: keep going while the model requests tool calls
            while assistant_message.tool_calls:
                print(f"Model requested {len(assistant_message.tool_calls)} tool call(s):")
                messages.append(assistant_message)

                for tool_call in assistant_message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    print(f"  -> {name}({json.dumps(args, indent=2)})")

                    # Forward the tool call to the GitHub MCP server
                    result = await session.call_tool(name, args)
                    tool_content = "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    })

                print("\n=== Sending tool results back to model ===")
                response = client.chat.completions.create(
                    model="databricks-claude-opus-4-6",
                    messages=messages,
                    tools=openai_tools,
                    max_tokens=5000,
                )
                assistant_message = response.choices[0].message

            print("\n=== Final Response ===")
            print(assistant_message.content)


asyncio.run(main())
