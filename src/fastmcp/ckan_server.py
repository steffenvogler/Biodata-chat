from fastmcp import FastMCP
import requests
from typing import Optional

mcp = FastMCP("CKAN NHM Server")

CKAN_API_BASE = "https://data.nhm.ac.uk/api/3/action"

@mcp.tool
def list_datasets() -> dict:
    """List all datasets from the CKAN NHM API."""
    response = requests.get(f"{CKAN_API_BASE}/package_list")
    return response.json()

@mcp.tool
def search_packages(query: str, limit: int = 5) -> dict:
    """Search for packages matching a query using CKAN NHM API."""
    params = {"q": query, "rows": limit}
    response = requests.get(f"{CKAN_API_BASE}/package_search", params=params)
    return response.json()

@mcp.tool
def get_package_details(package_id: str) -> dict:
    """Get detailed information about a specific package/dataset."""
    params = {"id": package_id}
    response = requests.get(f"{CKAN_API_BASE}/package_show", params=params)
    return response.json()

@mcp.tool
def list_groups() -> dict:
    """List all groups/categories in the CKAN NHM database."""
    response = requests.get(f"{CKAN_API_BASE}/group_list")
    return response.json()

@mcp.tool
def list_organizations() -> dict:
    """List all organizations in the CKAN NHM database."""
    response = requests.get(f"{CKAN_API_BASE}/organization_list")
    return response.json()

@mcp.tool
def get_group_details(group_id: str) -> dict:
    """Get detailed information about a specific group/category."""
    params = {"id": group_id}
    response = requests.get(f"{CKAN_API_BASE}/group_show", params=params)
    return response.json()

@mcp.tool
def get_organization_details(org_id: str) -> dict:
    """Get detailed information about a specific organization."""
    params = {"id": org_id}
    response = requests.get(f"{CKAN_API_BASE}/organization_show", params=params)
    return response.json()

@mcp.tool
def list_tags(limit: int = 100) -> dict:
    """List tags used in the CKAN NHM database."""
    params = {"limit": limit}
    response = requests.get(f"{CKAN_API_BASE}/tag_list", params=params)
    return response.json()

@mcp.tool
def search_resources(query: str, limit: int = 5) -> dict:
    """Search for resources (files/data) matching a query."""
    params = {"query": query, "limit": limit}
    response = requests.get(f"{CKAN_API_BASE}/resource_search", params=params)
    return response.json()

@mcp.tool
def get_resource_details(resource_id: str) -> dict:
    """Get detailed information about a specific resource/file."""
    params = {"id": resource_id}
    response = requests.get(f"{CKAN_API_BASE}/resource_show", params=params)
    return response.json()

@mcp.tool
def get_recent_activity(limit: int = 20) -> dict:
    """Get recent activity/changes in the CKAN NHM database."""
    params = {"limit": limit}
    try:
        response = requests.get(f"{CKAN_API_BASE}/recently_changed_packages_activity_list", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def search_by_tag(tag: str, limit: int = 10) -> dict:
    """Search for packages by a specific tag."""
    params = {"fq": f"tags:{tag}", "rows": limit}
    try:
        response = requests.get(f"{CKAN_API_BASE}/package_search", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def search_by_organization(org_name: str, limit: int = 10) -> dict:
    """Search for packages from a specific organization."""
    params = {"fq": f"organization:{org_name}", "rows": limit}
    try:
        response = requests.get(f"{CKAN_API_BASE}/package_search", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_dataset_stats() -> dict:
    """Get statistics about the CKAN NHM database."""
    try:
        response = requests.get(f"{CKAN_API_BASE}/status_show")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    mcp.run(transport="stdio")
