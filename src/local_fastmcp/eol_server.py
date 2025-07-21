from local_fastmcp import FastMCP
import requests
from typing import Optional, Dict, Any

mcp = FastMCP("EOL (Encyclopedia of Life) Server")

EOL_API_BASE = "https://eol.org/api"
EOL_STRUCTURED_API = "https://eol.org/service"

@mcp.tool
def ping_eol() -> dict:
    """Check if the EOL API is available."""
    try:
        response = requests.get(f"{EOL_API_BASE}/ping.json")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def search_species(query: str, exact: bool = False, filter_by_taxon_concept_id: Optional[str] = None) -> dict:
    """Search for species by name using EOL search API."""
    params = {
        "q": query,
        "exact": str(exact).lower()
    }
    if filter_by_taxon_concept_id:
        params["filter_by_taxon_concept_id"] = filter_by_taxon_concept_id
    
    try:
        response = requests.get(f"{EOL_API_BASE}/search/1.0.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_page_info(page_id: int, images_per_page: int = 75, videos_per_page: int = 75, 
                  sounds_per_page: int = 75, maps_per_page: int = 75, 
                  texts_per_page: int = 75, subjects: str = "all", 
                  licenses: str = "all", details: bool = True, 
                  common_names: bool = True, synonyms: bool = True) -> dict:
    """Get detailed information about a species page."""
    params = {
        "images_per_page": images_per_page,
        "videos_per_page": videos_per_page,
        "sounds_per_page": sounds_per_page,
        "maps_per_page": maps_per_page,
        "texts_per_page": texts_per_page,
        "subjects": subjects,
        "licenses": licenses,
        "details": str(details).lower(),
        "common_names": str(common_names).lower(),
        "synonyms": str(synonyms).lower()
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/pages/1.0/{page_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_hierarchy_entries(page_id: int, common_names: bool = True, synonyms: bool = True) -> dict:
    """Get hierarchy entries for a species page."""
    params = {
        "common_names": str(common_names).lower(),
        "synonyms": str(synonyms).lower()
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/hierarchy_entries/1.0/{page_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_data_objects(page_id: int, subject: str = "all", language: str = "ms") -> dict:
    """Get data objects (text, images, videos, sounds) for a species page."""
    params = {
        "subject": subject,
        "language": language
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/data_objects/1.0/{page_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_collections(page: int = 1, per_page: int = 50, filter_by: str = "", sort_by: str = "newest") -> dict:
    """Get collections from EOL."""
    params = {
        "page": page,
        "per_page": per_page,
        "filter_by": filter_by,
        "sort_by": sort_by
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/collections/1.0.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_collection_details(collection_id: int, page: int = 1, per_page: int = 50, 
                          sort_by: str = "newest", filter_by: str = "") -> dict:
    """Get details of a specific collection."""
    params = {
        "page": page,
        "per_page": per_page,
        "sort_by": sort_by,
        "filter_by": filter_by
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/collections/1.0/{collection_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def search_by_provider(provider_hierarchy_id: int, hierarchy_id: Optional[int] = None, 
                      cache_ttl: Optional[int] = None) -> dict:
    """Search for species by provider hierarchy."""
    params = {}
    if hierarchy_id:
        params["hierarchy_id"] = hierarchy_id
    if cache_ttl:
        params["cache_ttl"] = cache_ttl
    
    try:
        response = requests.get(f"{EOL_API_BASE}/search_by_provider/1.0/{provider_hierarchy_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_hierarchies(page: int = 1, per_page: int = 50, filter_by: str = "") -> dict:
    """Get available taxonomic hierarchies."""
    params = {
        "page": page,
        "per_page": per_page,
        "filter_by": filter_by
    }
    
    try:
        response = requests.get(f"{EOL_API_BASE}/hierarchies/1.0.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_hierarchy_details(hierarchy_id: int, cache_ttl: Optional[int] = None) -> dict:
    """Get details of a specific taxonomic hierarchy."""
    params = {}
    if cache_ttl:
        params["cache_ttl"] = cache_ttl
    
    try:
        response = requests.get(f"{EOL_API_BASE}/hierarchies/1.0/{hierarchy_id}.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False}

@mcp.tool
def get_cypher_query(query: str) -> dict:
    """Execute a Cypher query against EOL's structured data API.
    Note: This requires an API key and special access permissions."""
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "query": query
    }
    
    try:
        response = requests.post(f"{EOL_STRUCTURED_API}/cypher", json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "success": False, "note": "This endpoint may require API key authentication"}

@mcp.tool
def get_species_traits(page_id: int) -> dict:
    """Get trait data for a species using a simplified Cypher query."""
    # Simple query to get traits for a taxon
    query = f"""
    MATCH (t:Taxon)-[:trait]->(trait:Trait)
    WHERE t.page_id = {page_id}
    RETURN t.canonical, trait.predicate, trait.measurement, trait.units
    LIMIT 100
    """
    
    return get_cypher_query(query)

@mcp.tool
def get_species_interactions(page_id: int, interaction_type: Optional[str] = None) -> dict:
    """Get ecological interaction data for a species."""
    base_query = f"""
    MATCH (t1:Taxon)-[r:interacts_with]->(t2:Taxon)
    WHERE t1.page_id = {page_id}
    """
    
    if interaction_type:
        base_query += f" AND r.type = '{interaction_type}'"
    
    query = base_query + """
    RETURN t1.canonical as species1, r.type as interaction, t2.canonical as species2
    LIMIT 50
    """
    
    return get_cypher_query(query)

@mcp.tool
def search_taxa_by_trait(trait_name: str, limit: int = 20) -> dict:
    """Search for taxa that have a specific trait."""
    query = f"""
    MATCH (t:Taxon)-[:trait]->(trait:Trait)
    WHERE trait.predicate CONTAINS '{trait_name}'
    RETURN t.canonical, t.page_id, trait.predicate, trait.measurement, trait.units
    LIMIT {limit}
    """
    
    return get_cypher_query(query)

if __name__ == "__main__":
    mcp.run(transport="stdio")
