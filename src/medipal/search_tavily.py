from tavily import TavilyClient
import os

def search_evidence(query: str, max_results=3) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return ""
    client = TavilyClient(api_key=key)
    resp = client.search(query, search_depth="basic", max_results=max_results)
    snippets = [r["content"] for r in resp["results"]]
    return "\n".join(snippets)