"""Paper discovery agent using arXiv, Semantic Scholar, and Unpaywall."""

import os
import logging
import time
from typing import List, Dict, Any, Optional
import yaml
import arxiv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class PaperFinderAgent:
    """Agent for discovering papers from multiple sources."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize paper finder agent.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.api_config = self.config["apis"]
        self.rate_limits = self.config["rate_limits"]
        self.max_results = self.config["paper_finder"]["max_results"]

        # API keys
        self.semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        self.unpaywall_email = os.getenv("UNPAYWALL_EMAIL", "")

        # Rate limiting state
        self._last_request_time = {}

    def _rate_limit(self, source: str):
        """Apply rate limiting for API requests.

        Args:
            source: API source name
        """
        if source in self._last_request_time:
            elapsed = time.time() - self._last_request_time[source]
            min_interval = 1.0 / self.rate_limits.get(source, 1)

            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        self._last_request_time[source] = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        logger.info(f"Searching arXiv for: {query}")
        self._rate_limit("arxiv")

        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                paper = {
                    "paper_id": result.entry_id.split("/")[-1],
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "title": result.title,
                    "authors": [{"name": author.name} for author in result.authors],
                    "abstract": result.summary,
                    "year": result.published.year,
                    "publication_date": result.published.isoformat(),
                    "pdf_url": result.pdf_url,
                    "is_open_access": True,
                    "source": "arxiv",
                    "venue": "arXiv",
                    "doi": result.doi if hasattr(result, "doi") else None
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers from arXiv")
            return papers

        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search Semantic Scholar for papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        logger.info(f"Searching Semantic Scholar for: {query}")
        self._rate_limit("semantic_scholar")

        base_url = self.api_config["semantic_scholar"]["base_url"]
        fields = self.api_config["semantic_scholar"]["fields"]

        headers = {}
        if self.semantic_scholar_key:
            headers["x-api-key"] = self.semantic_scholar_key

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{base_url}/paper/search",
                    params={
                        "query": query,
                        "limit": max_results,
                        "fields": fields
                    },
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()

            papers = []
            for item in data.get("data", []):
                # Extract open access PDF URL
                pdf_url = None
                is_open_access = False
                if item.get("openAccessPdf"):
                    pdf_url = item["openAccessPdf"].get("url")
                    is_open_access = True

                paper = {
                    "paper_id": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "authors": [
                        {"name": author.get("name", "")}
                        for author in item.get("authors", [])
                    ],
                    "abstract": item.get("abstract"),
                    "year": item.get("year"),
                    "publication_date": item.get("publicationDate"),
                    "venue": item.get("venue"),
                    "citation_count": item.get("citationCount", 0),
                    "pdf_url": pdf_url,
                    "is_open_access": is_open_access,
                    "source": "semantic_scholar"
                }
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers from Semantic Scholar")
            return papers

        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def find_open_access_pdf(self, doi: str) -> Optional[str]:
        """Find open access PDF URL using Unpaywall.

        Args:
            doi: Paper DOI

        Returns:
            PDF URL if found, None otherwise
        """
        if not doi or not self.unpaywall_email:
            return None

        logger.info(f"Checking Unpaywall for DOI: {doi}")
        self._rate_limit("unpaywall")

        base_url = self.api_config["unpaywall"]["base_url"]

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{base_url}/{doi}",
                    params={"email": self.unpaywall_email}
                )

                if response.status_code == 200:
                    data = response.json()

                    if data.get("is_oa"):
                        best_oa = data.get("best_oa_location")
                        if best_oa and best_oa.get("url_for_pdf"):
                            logger.info(f"Found OA PDF via Unpaywall")
                            return best_oa["url_for_pdf"]

        except Exception as e:
            logger.error(f"Error checking Unpaywall: {e}")

        return None

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for papers across multiple sources.

        Args:
            query: Search query
            max_results: Maximum total results
            sources: List of sources to search (default: all)

        Returns:
            Combined list of paper metadata
        """
        max_results = max_results or self.max_results
        sources = sources or self.config["paper_finder"]["sources"]

        all_papers = []

        # Search each source
        per_source = max_results // len(sources)

        if "arxiv" in sources:
            arxiv_papers = self.search_arxiv(query, per_source)
            all_papers.extend(arxiv_papers)

        if "semantic_scholar" in sources:
            ss_papers = self.search_semantic_scholar(query, per_source)
            all_papers.extend(ss_papers)

        # Deduplicate by title (simple approach)
        seen_titles = set()
        unique_papers = []

        for paper in all_papers:
            title_normalized = paper["title"].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)

        # Enhance with Unpaywall if DOI available and no PDF
        if self.config["paper_finder"].get("prefer_open_access", True):
            for paper in unique_papers:
                if not paper.get("pdf_url") and paper.get("doi"):
                    oa_url = self.find_open_access_pdf(paper["doi"])
                    if oa_url:
                        paper["pdf_url"] = oa_url
                        paper["is_open_access"] = True

        logger.info(f"Total unique papers found: {len(unique_papers)}")
        return unique_papers[:max_results]

    def get_paper_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Get paper metadata by DOI.

        Args:
            doi: Paper DOI

        Returns:
            Paper metadata if found
        """
        # Try Semantic Scholar first
        papers = self.search_semantic_scholar(f"doi:{doi}", max_results=1)

        if papers:
            paper = papers[0]
            # Try to find OA PDF
            if not paper.get("pdf_url"):
                oa_url = self.find_open_access_pdf(doi)
                if oa_url:
                    paper["pdf_url"] = oa_url
                    paper["is_open_access"] = True
            return paper

        return None

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by arXiv ID.

        Args:
            arxiv_id: arXiv identifier

        Returns:
            Paper metadata if found
        """
        papers = self.search_arxiv(f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None
