from typing import List, Dict
from duckduckgo_search import DDGS
from .config import settings

class WebSearch:
    def __init__(self):
        self.max_results = settings.MAX_SEARCH_RESULTS

    def search_web(self, query: str, num_results: int = None) -> List[Dict[str, str]]:
        """
        Search the web using DuckDuckGo API

        Args:
            query: Search query string
            num_results: Number of results to return (default: from settings)

        Returns:
            List of search results with title, snippet, and url
        """
        num_results = num_results or self.max_results

        try:
            results = []

            # DDGS() no longer requires context manager in version 8.x
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=num_results)

            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', '')
                })

            return results

        except Exception as e:
            print(f"Error during web search: {e}")
            return []

    def format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results into a readable string for the model"""
        if not results:
            return "No web search results found."

        formatted = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   URL: {result['url']}\n\n"

        return formatted

# Global web search instance
web_search = WebSearch()
