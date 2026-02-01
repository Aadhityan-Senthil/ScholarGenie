"""End-to-end pipeline tests."""

import pytest
import os
from backend.agents.paper_finder import PaperFinderAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.extractor import ExtractorAgent
from backend.agents.presenter import PresenterAgent
from backend.agents.evaluator import EvaluatorAgent
from backend.utils.metadata import PaperMetadata, Author, Section


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def sample_paper(self):
        """Create a sample paper for testing."""
        return PaperMetadata(
            paper_id="pipeline_test_001",
            title="Attention Is All You Need",
            authors=[
                Author(name="Ashish Vaswani"),
                Author(name="Noam Shazeer")
            ],
            abstract=(
                "The dominant sequence transduction models are based on complex recurrent "
                "or convolutional neural networks that include an encoder and a decoder. "
                "The best performing models also connect the encoder and decoder through "
                "an attention mechanism. We propose a new simple network architecture, "
                "the Transformer, based solely on attention mechanisms, dispensing with "
                "recurrence and convolutions entirely."
            ),
            year=2017,
            venue="NeurIPS",
            sections=[
                Section(
                    title="Introduction",
                    content=(
                        "Recurrent neural networks, long short-term memory and gated "
                        "recurrent neural networks in particular, have been firmly established "
                        "as state of the art approaches in sequence modeling and transduction "
                        "problems such as language modeling and machine translation."
                    ),
                    level=1
                ),
                Section(
                    title="Model Architecture",
                    content=(
                        "Most competitive neural sequence transduction models have an "
                        "encoder-decoder structure. Here, the encoder maps an input sequence "
                        "to a sequence of continuous representations. The decoder then generates "
                        "an output sequence of symbols one element at a time."
                    ),
                    level=1
                ),
                Section(
                    title="Results",
                    content=(
                        "On the WMT 2014 English-to-German translation task, the big transformer "
                        "model outperforms the best previously reported models by more than 2.0 BLEU. "
                        "On the WMT 2014 English-to-French translation task, our model achieves "
                        "a BLEU score of 41.0, outperforming all previously published models."
                    ),
                    level=1
                )
            ]
        )

    def test_complete_pipeline(self, sample_paper, tmp_path):
        """Test the complete pipeline from ingestion to presentation."""
        # Initialize agents
        summarizer = SummarizerAgent()
        extractor = ExtractorAgent()
        presenter = PresenterAgent()
        evaluator = EvaluatorAgent()

        # Step 1: Summarize paper
        summary = summarizer.summarize_paper(sample_paper)
        assert summary is not None
        assert summary.tldr is not None
        assert summary.full_summary is not None
        assert len(summary.keypoints) > 0

        # Step 2: Extract structured data
        extracted_data = extractor.extract(sample_paper)
        assert extracted_data is not None
        assert extracted_data.paper_id == sample_paper.paper_id

        # Step 3: Generate presentation
        pptx_path = os.path.join(tmp_path, "test_presentation.pptx")
        result_path = presenter.generate_pptx(
            sample_paper,
            summary,
            extracted_data,
            output_path=pptx_path
        )
        assert os.path.exists(result_path)
        assert result_path == pptx_path

        # Step 4: Generate report
        md_path = os.path.join(tmp_path, "test_report.md")
        result_path = presenter.generate_markdown_report(
            sample_paper,
            summary,
            extracted_data,
            output_path=md_path
        )
        assert os.path.exists(result_path)

        # Verify report content
        with open(result_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
            assert sample_paper.title in report_content
            assert summary.tldr in report_content

        # Step 5: Evaluate summary
        evaluation = evaluator.evaluate_summary(sample_paper, summary)
        assert evaluation is not None
        assert "quality_checks" in evaluation
        assert "metrics" in evaluation

    def test_paper_search_integration(self):
        """Test paper search integration (requires internet)."""
        finder = PaperFinderAgent()

        # Search for a well-known paper
        results = finder.search("attention is all you need", max_results=3)

        # Should find some results
        assert isinstance(results, list)
        # Note: May be empty if APIs are down or rate limited

    def test_summarizer_model_loading(self):
        """Test that summarizer can load models."""
        summarizer = SummarizerAgent()

        # Check model manager
        assert summarizer.model_manager is not None

        # Test model info
        info = summarizer.model_manager.get_model_info()
        assert "device" in info
        assert "summarization_model" in info

    def test_extractor_patterns(self, sample_paper):
        """Test extractor pattern matching."""
        extractor = ExtractorAgent()

        extracted = extractor.extract(sample_paper)

        # Should extract some information
        assert extracted is not None

        # Check for methods (may or may not find depending on content)
        assert isinstance(extracted.methods, list)
        assert isinstance(extracted.datasets, list)
        assert isinstance(extracted.models, list)

    def test_presenter_markdown_generation(self, sample_paper, tmp_path):
        """Test markdown report generation."""
        summarizer = SummarizerAgent()
        presenter = PresenterAgent()

        summary = summarizer.summarize_paper(sample_paper)

        md_path = os.path.join(tmp_path, "test.md")
        result = presenter.generate_markdown_report(
            sample_paper,
            summary,
            output_path=md_path
        )

        assert os.path.exists(result)

        # Check content
        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "# Attention Is All You Need" in content
            assert "## TL;DR" in content
            assert "## Summary" in content

    def test_evaluator_quality_checks(self, sample_paper):
        """Test evaluator quality checks."""
        summarizer = SummarizerAgent()
        evaluator = EvaluatorAgent()

        summary = summarizer.summarize_paper(sample_paper)
        evaluation = evaluator.evaluate_summary(sample_paper, summary)

        # Check structure
        assert "quality_checks" in evaluation
        qc = evaluation["quality_checks"]

        assert "tldr_length" in qc
        assert "full_summary_length" in qc
        assert "keypoints_count" in qc

        # Check that quality checks have proper structure
        assert "word_count" in qc["tldr_length"]
        assert "pass" in qc["tldr_length"]


@pytest.mark.slow
class TestSlowPipeline:
    """Slow integration tests that may take longer."""

    def test_large_paper_processing(self):
        """Test processing a large paper (marked as slow)."""
        # This would test with a real large PDF
        # Skipped in quick test runs
        pytest.skip("Slow test - run with pytest -m slow")
