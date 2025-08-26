# =============================================================================
#  Filename: server.py
#
#  Short Description: FastMCP server for handling MCP protocol requests
#
#  Creation date: 2025-07-13
#  Author: Asif Qamar
# =============================================================================

from fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP()

@mcp.tool(name="happy", description="Happy Messages")
def happy():
    return "Happy Happy Day!!!"

if __name__ == "__main__":
    logger.info("Starting MCP server on http://127.0.0.1:8000/mcp/")
    mcp.run(transport="http")

#============================================================================================