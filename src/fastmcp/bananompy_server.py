from fastmcp import FastMCP
import bananompy

mcp = FastMCP("Bionomia Server")

@mcp.tool
def search_collector(name: str, limit: int = 5) -> dict:
    """Search for a collector by name using Bionomia API."""
    result = bananompy.person.search(name, limit=limit)
    return result

@mcp.tool
def search_collector_with_occurrences(name: str, limit: int = 5) -> dict:
    """Search for collectors with occurrences by name using Bionomia API."""
    result = bananompy.suggest(name, has_occurrences=True, limit=limit)
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
