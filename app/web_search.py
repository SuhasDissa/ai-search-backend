from typing import List, Dict, Optional, Tuple
from duckduckgo_search import DDGS
from .config import settings
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ContentCache:
    """Simple in-memory cache for fetched web content"""
    def __init__(self):
        self.cache: Dict[str, tuple[str, datetime]] = {}

    def get(self, url: str) -> Optional[str]:
        """Get cached content if not expired"""
        if url in self.cache:
            content, timestamp = self.cache[url]
            if datetime.now() - timestamp < timedelta(seconds=3600):  # 1 hour TTL
                logger.debug(f"Cache HIT for {url}")
                return content
            else:
                logger.debug(f"Cache EXPIRED for {url}")
                del self.cache[url]
        return None

    def set(self, url: str, content: str):
        """Cache content with timestamp"""
        self.cache[url] = (content, datetime.now())
        logger.debug(f"Cached content for {url} ({len(content)} chars)")
        if len(self.cache) > 100:
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:20]
            for key in oldest_keys:
                del self.cache[key]
            logger.debug(f"Cache cleanup: removed {len(oldest_keys)} old entries")

class ContentRanker:
    """Lightweight TF-IDF based content ranking"""
    def __init__(self):
        self.vectorizer = None

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def rank_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank chunks by relevance using TF-IDF"""
        if not chunks:
            return []

        try:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=5000,
                    min_df=1
                )

            all_texts = [query] + chunks
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[0:1]
            chunk_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(chunks[i], float(similarities[i])) for i in top_indices]
        except:
            return [(chunk, 0.5) for chunk in chunks[:top_k]]

class WebSearch:
    def __init__(self):
        self.max_results = settings.MAX_SEARCH_RESULTS
        self.cache = ContentCache()
        self.ranker = ContentRanker()
        self.model_handler = None

    def set_model_handler(self, handler):
        """Set model handler for query optimization"""
        self.model_handler = handler

    def optimize_query_with_llm(self, user_query: str) -> str:
        """Use Qwen to optimize the search query"""
        if not self.model_handler or not self.model_handler.model:
            logger.debug("Model handler not available, using original query")
            return user_query

        prompt = f"""Convert the following question into an optimized search query with 3-6 keywords.
Remove question words and keep only the essential terms.

Question: {user_query}

Search Query:"""

        try:
            logger.debug(f"Optimizing query: '{user_query}'")
            optimized = self.model_handler.generate_response(prompt, max_tokens=30).strip()
            optimized = optimized.replace('"', '').replace('\n', ' ').strip()
            logger.info(f"Query optimization: '{user_query}' -> '{optimized}'")
            return optimized if optimized else user_query
        except Exception as e:
            logger.error(f"Query optimization error: {e}", exc_info=True)
            return user_query

    async def fetch_webpage(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch and extract webpage content"""
        cached = self.cache.get(url)
        if cached:
            return cached

        try:
            logger.debug(f"Fetching webpage: {url}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            async with session.get(url, headers=headers, timeout=8) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                    element.decompose()

                main_content = soup.find('main') or soup.find('article') or soup.find('body')
                text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)

                if len(text) > 5000:
                    text = text[:5000]
                    logger.debug(f"Truncated content from {url} to 5000 chars")

                self.cache.set(url, text)
                logger.info(f"Successfully fetched {url} ({len(text)} chars)")
                return text
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None

    async def fetch_multiple_pages(self, urls: List[str]) -> List[Dict[str, Optional[str]]]:
        """Fetch multiple pages in parallel"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_webpage(session, url) for url in urls]
            contents = await asyncio.gather(*tasks)
            return [{"url": url, "content": content} for url, content in zip(urls, contents)]

    def search_web(self, query: str, num_results: int = None) -> List[Dict[str, str]]:
        """Search using DuckDuckGo"""
        num_results = num_results or self.max_results
        try:
            logger.info(f"Searching DuckDuckGo for: '{query}' (max_results={num_results})")
            results = []
            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=num_results)
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'snippet': result.get('body', ''),
                    'url': result.get('href', '')
                })
            logger.info(f"Found {len(results)} search results")
            return results
        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return []

    def create_compact_context(self, query: str, search_results: List[Dict], max_chunks_per_url: int = 3) -> str:
        """Create compact context from search results with TF-IDF ranking"""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            content = result.get('content')
            if not content:
                context_parts.append(f"[{i}] {result['title']}\nSource: {result['url']}\n{result['snippet']}\n")
                continue

            chunks = self.ranker.chunk_text(content, chunk_size=500)
            ranked_chunks = self.ranker.rank_chunks(query, chunks, top_k=max_chunks_per_url)

            if ranked_chunks:
                context_parts.append(f"[{i}] {result['title']}\nSource: {result['url']}\n")
                for j, (chunk, score) in enumerate(ranked_chunks, 1):
                    context_parts.append(f"  Excerpt {j} (relevance: {score:.2f}):\n  {chunk}\n")

        return "\n".join(context_parts)

    def enhanced_search(self, user_query: str) -> tuple[str, dict]:
        """
        Perform enhanced web search with LLM query optimization and content ranking
        Returns (compact_context, metadata)
        """
        import time
        start_time = time.time()

        metadata = {
            'original_query': user_query,
            'optimized_query': None,
            'urls_fetched': 0,
            'chunks_ranked': 0,
            'avg_relevance_score': None,
            'cache_used': False,
            'search_time_ms': None
        }

        logger.info(f"Starting enhanced search for: '{user_query}'")

        # Step 1: Optimize query with Qwen
        optimized_query = self.optimize_query_with_llm(user_query)
        metadata['optimized_query'] = optimized_query

        # Step 2: Search
        search_results = self.search_web(optimized_query, num_results=5)

        if not search_results:
            metadata['search_time_ms'] = int((time.time() - start_time) * 1000)
            logger.warning("No web search results found")
            return "No web search results found.", metadata

        # Step 3: Fetch content from top 3 URLs
        top_urls = [r['url'] for r in search_results[:3]]
        logger.info(f"Fetching content from {len(top_urls)} URLs...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fetched_contents = loop.run_until_complete(self.fetch_multiple_pages(top_urls))
        loop.close()

        # Count successful fetches and check cache
        successful_fetches = sum(1 for item in fetched_contents if item['content'])
        metadata['urls_fetched'] = successful_fetches
        metadata['cache_used'] = any(self.cache.get(url) for url in top_urls)
        logger.info(f"Successfully fetched {successful_fetches}/{len(top_urls)} URLs")

        # Map content back to results
        content_map = {item['url']: item['content'] for item in fetched_contents}
        for result in search_results:
            result['content'] = content_map.get(result['url'])

        # Step 4: Create compact context with TF-IDF ranking
        logger.debug("Creating compact context with TF-IDF ranking...")
        compact_context = self.create_compact_context(optimized_query, search_results, max_chunks_per_url=3)

        # Calculate ranking stats
        all_scores = []
        total_chunks = 0
        for result in search_results:
            if result.get('content'):
                chunks = self.ranker.chunk_text(result['content'], chunk_size=500)
                total_chunks += len(chunks)
                ranked = self.ranker.rank_chunks(optimized_query, chunks, top_k=3)
                all_scores.extend([score for _, score in ranked])

        metadata['chunks_ranked'] = total_chunks
        if all_scores:
            metadata['avg_relevance_score'] = round(sum(all_scores) / len(all_scores), 3)

        metadata['search_time_ms'] = int((time.time() - start_time) * 1000)

        logger.info(f"Enhanced search completed in {metadata['search_time_ms']}ms - "
                   f"ranked {total_chunks} chunks, avg score: {metadata['avg_relevance_score']}")

        return compact_context, metadata

# Global web search instance
web_search = WebSearch()
