from fastmcp import FastMCP
import requests

try:
    import bananompy
    HAS_BANANOMPY = True
except ImportError:
    HAS_BANANOMPY = False
    print("Warning: bananompy not available. Using direct API calls as fallback.")

mcp = FastMCP("Bionomia Server")

@mcp.tool
def search_collector(name: str, limit: int = 5) -> dict:
    """Search for a collector by name using Bionomia API."""
    if HAS_BANANOMPY:
        result = bananompy.person.search(name, limit=limit)
        return result
    else:
        # Fallback to direct API call
        try:
            url = f"https://bionomia.net/api/v1/users/search"
            params = {"q": name, "limit": limit}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "message": "Failed to search collectors"}

@mcp.tool
def search_collector_with_occurrences(name: str, limit: int = 5) -> dict:
    """Search for collectors with occurrences by name using Bionomia API."""
    if HAS_BANANOMPY:
        result = bananompy.suggest(name, has_occurrences=True, limit=limit)
        return result
    else:
        # Fallback to direct API call
        try:
            url = f"https://bionomia.net/api/v1/users/search"
            params = {"q": name, "has_occurrences": "true", "limit": limit}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "message": "Failed to search collectors with occurrences"}

if __name__ == "__main__":
    mcp.run(transport="stdio")
