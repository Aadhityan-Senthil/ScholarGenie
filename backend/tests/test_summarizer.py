"""Tests for SummarizerAgent."""

import pytest
from backend.agents.summarizer import SummarizerAgent
from backend.utils.metadata import PaperMetadata, Author, Section


class TestSummarizerAgent:
    """Test cases for SummarizerAgent."""

    @pytest.fixture
    def agent(self):
        """Create a SummarizerAgent instance."""
        return SummarizerAgent()

    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        return PaperMetadata(
            paper_id="test_001",
            title="Test Paper on Transformers",
            authors=[Author(name="John Doe"), Author(name="Jane Smith")],
            abstract="This is a test abstract about transformer models and their applications in NLP.",
            year=2023,
            sections=[
                Section(
                    title="Introduction",
                    content="Transformers have revolutionized natural language processing. They use attention mechanisms.",
                    level=1
                ),
                Section(
                    title="Methods",
                    content="We employ a transformer-based architecture with multi-head attention.",
                    level=1
                ),
                Section(
                    title="Results",
                    content="Our model achieves state-of-the-art performance on multiple benchmarks.",
                    level=1
                )
            ]
        )

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert agent.model_manager is not None

    def test_summarize_paper(self, agent, sample_paper):
        """Test paper summarization."""
        summary = agent.summarize_paper(sample_paper)

        assert summary is not None
        assert summary.paper_id == sample_paper.paper_id
        assert summary.tldr is not None
        assert len(summary.tldr) > 0
        assert summary.short_summary is not None
        assert summary.full_summary is not None

    def test_generate_tldr(self, agent, sample_paper):
        """Test TL;DR generation."""
        tldr = agent._generate_tldr(sample_paper)

        assert tldr is not None
        assert len(tldr) > 0
        assert tldr.endswith('.')

    def test_extract_keypoints(self, agent):
        """Test keypoint extraction."""
        full_text = "Transformers use attention. They are powerful. They work well for NLP tasks."
        summary = "Transformers are effective models."

        keypoints = agent._extract_keypoints(full_text, summary)

        assert isinstance(keypoints, list)
        # May or may not find points depending on model, but should return a list
        assert len(keypoints) >= 0

    def test_summarize_sections(self, agent, sample_paper):
        """Test section summarization."""
        section_summaries = agent._summarize_sections(sample_paper)

        assert isinstance(section_summaries, dict)
        # Should have summaries for some sections
        assert len(section_summaries) > 0
