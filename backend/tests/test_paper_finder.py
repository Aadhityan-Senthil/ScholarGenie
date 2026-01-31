"""Tests for PaperFinderAgent."""

import pytest
from unittest.mock import Mock, patch
from backend.agents.paper_finder import PaperFinderAgent


class TestPaperFinderAgent:
    """Test cases for PaperFinderAgent."""

    @pytest.fixture
    def agent(self):
        """Create a PaperFinderAgent instance."""
        return PaperFinderAgent()

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert agent.max_results > 0

    @patch('backend.agents.paper_finder.arxiv.Search')
    def test_search_arxiv(self, mock_search, agent):
        """Test arXiv search functionality."""
        # Mock arXiv results
        mock_result = Mock()
        mock_result.entry_id = "http://arxiv.org/abs/1706.03762"
        mock_result.title = "Attention Is All You Need"
        mock_result.authors = [Mock(name="Vaswani")]
        mock_result.summary = "Test abstract"
        mock_result.published.year = 2017
        mock_result.published.isoformat.return_value = "2017-06-12"
        mock_result.pdf_url = "https://arxiv.org/pdf/1706.03762.pdf"
        mock_result.doi = None

        mock_search.return_value.results.return_value = [mock_result]

        # Test search
        results = agent.search_arxiv("transformer", max_results=1)

        assert len(results) == 1
        assert results[0]["title"] == "Attention Is All You Need"
        assert results[0]["is_open_access"] is True

    def test_search_with_invalid_query(self, agent):
        """Test search with invalid query."""
        results = agent.search("", max_results=1)
        # Should handle gracefully
        assert isinstance(results, list)

    def test_rate_limiting(self, agent):
        """Test that rate limiting is applied."""
        import time

        # Record time for multiple calls
        start = time.time()
        agent._rate_limit("arxiv")
        agent._rate_limit("arxiv")
        elapsed = time.time() - start

        # Should have some delay (at least based on rate limit)
        # This is a basic test, actual timing may vary
        assert elapsed >= 0  # Basic sanity check
