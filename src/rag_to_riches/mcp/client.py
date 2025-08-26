# =============================================================================
#  Filename: client.py
#
#  Short Description: FastMCP client for connecting to MCP protocol servers
#
#  Creation date: 2025-07-13
#  Author: Asif Qamar
# =============================================================================

import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:8000/mcp/") as client:
        # List available tools
        tools = await client.list_tools()
        print("Available tools:", tools)
        
        # Invoke the happy tool
        result = await client.call_tool("happy")
        print("Result from happy tool:", result)

if __name__ == "__main__":
    asyncio.run(example())

#============================================================================================