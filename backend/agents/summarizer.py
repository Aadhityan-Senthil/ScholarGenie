"""Summarization agent with multi-granularity support."""

import logging
from typing import Dict, List, Optional
import yaml

from backend.utils.metadata import PaperMetadata, Summary
from backend.utils.models import ModelManager

logger = logging.getLogger(__name__)


class SummarizerAgent:
    """Agent for multi-granularity paper summarization."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize summarizer agent.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sum_config = self.config["summarization"]
        self.model_manager = ModelManager(config_path)

    def summarize_paper(self, paper: PaperMetadata) -> Summary:
        """Generate multi-granularity summary of a paper.

        Args:
            paper: Paper metadata with full text

        Returns:
            Multi-level summary
        """
        logger.info(f"Summarizing paper: {paper.title}")

        # Get full text
        full_text = paper.get_full_text()

        # Generate different granularity summaries
        tldr = self._generate_tldr(paper)
        short_summary = self._generate_short_summary(paper)
        full_summary = self._generate_full_summary(full_text)

        # Section summaries
        section_summaries = self._summarize_sections(paper)

        # Extract key points
        keypoints = self._extract_keypoints(full_text, full_summary)

        # Extract specific sections
        methods = self._extract_methods_summary(paper)
        results = self._extract_results_summary(paper)
        limitations = self._extract_limitations(paper)

        return Summary(
            paper_id=paper.paper_id,
            tldr=tldr,
            short_summary=short_summary,
            full_summary=full_summary,
            section_summaries=section_summaries,
            keypoints=keypoints,
            methods=methods,
            results=results,
            limitations=limitations,
            model_used=self.sum_config["model_name"]
        )

    def _generate_tldr(self, paper: PaperMetadata) -> str:
        """Generate one-sentence TL;DR summary.

        Args:
            paper: Paper metadata

        Returns:
            TL;DR summary
        """
        # Use abstract if available, otherwise first section
        text = paper.abstract if paper.abstract else paper.get_full_text()[:1000]

        try:
            summary = self.model_manager.summarize_text(
                text,
                max_length=self.sum_config.get("tldr_max_length", 50),
                min_length=20
            )

            # Ensure it's one sentence
            summary = summary.split('.')[0] + '.'

            return summary

        except Exception as e:
            logger.error(f"Error generating TL;DR: {e}")
            # Fallback: use first sentence of abstract
            if paper.abstract:
                return paper.abstract.split('.')[0] + '.'
            return "Summary not available."

    def _generate_short_summary(self, paper: PaperMetadata) -> str:
        """Generate 2-3 sentence summary.

        Args:
            paper: Paper metadata

        Returns:
            Short summary
        """
        text = paper.abstract if paper.abstract else paper.get_full_text()[:2000]

        try:
            summary = self.model_manager.summarize_text(
                text,
                max_length=100,
                min_length=40
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating short summary: {e}")
            if paper.abstract:
                sentences = paper.abstract.split('.')[:3]
                return '. '.join(sentences) + '.'
            return "Summary not available."

    def _generate_full_summary(self, text: str) -> str:
        """Generate detailed summary (200-400 words).

        Args:
            text: Full paper text

        Returns:
            Detailed summary
        """
        try:
            # For very long texts, use chunking strategy
            max_input = self.sum_config.get("max_input_length", 4096)
            chunks = self.model_manager.chunk_text(text, max_tokens=max_input, overlap=100)

            if len(chunks) == 1:
                # Single chunk - direct summarization
                summary = self.model_manager.summarize_text(
                    chunks[0],
                    max_length=self.sum_config.get("full_summary_max_length", 512),
                    min_length=200
                )
            else:
                # Multiple chunks - hierarchical summarization
                chunk_summaries = []

                for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
                    logger.debug(f"Summarizing chunk {i+1}/{min(len(chunks), 5)}")
                    chunk_summary = self.model_manager.summarize_text(
                        chunk,
                        max_length=200,
                        min_length=50
                    )
                    chunk_summaries.append(chunk_summary)

                # Combine and summarize again
                combined = ' '.join(chunk_summaries)
                summary = self.model_manager.summarize_text(
                    combined,
                    max_length=self.sum_config.get("full_summary_max_length", 512),
                    min_length=200
                )

            return summary

        except Exception as e:
            logger.error(f"Error generating full summary: {e}")
            return "Detailed summary could not be generated."

    def _summarize_sections(self, paper: PaperMetadata) -> Dict[str, str]:
        """Generate summaries for each section.

        Args:
            paper: Paper metadata

        Returns:
            Dictionary mapping section titles to summaries
        """
        section_summaries = {}

        for section in paper.sections:
            if len(section.content) < 100:
                # Too short to summarize
                section_summaries[section.title] = section.content
                continue

            try:
                summary = self.model_manager.summarize_text(
                    section.content,
                    max_length=self.sum_config.get("section_summary_max_length", 200),
                    min_length=50
                )
                section_summaries[section.title] = summary

            except Exception as e:
                logger.error(f"Error summarizing section '{section.title}': {e}")
                # Use first few sentences as fallback
                sentences = section.content.split('.')[:3]
                section_summaries[section.title] = '. '.join(sentences) + '.'

        return section_summaries

    def _extract_keypoints(self, full_text: str, summary: str) -> List[str]:
        """Extract key bullet points from paper.

        Args:
            full_text: Full paper text
            summary: Generated summary

        Returns:
            List of key points
        """
        keypoints = []

        # Strategy: Generate a bulleted summary
        prompt_text = f"Key findings and contributions:\n\n{summary}\n\nFull context:\n{full_text[:2000]}"

        try:
            # Generate summary optimized for bullet points
            bullets_text = self.model_manager.summarize_text(
                prompt_text,
                max_length=300,
                min_length=100
            )

            # Split into sentences and format as bullets
            sentences = bullets_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:
                    keypoints.append(sentence)

            # Limit to configured count
            max_keypoints = self.sum_config.get("keypoints_count", 12)
            keypoints = keypoints[:max_keypoints]

        except Exception as e:
            logger.error(f"Error extracting keypoints: {e}")
            # Fallback: extract from summary
            sentences = summary.split('.')
            for sentence in sentences[:5]:
                if sentence.strip():
                    keypoints.append(sentence.strip())

        return keypoints

    def _extract_methods_summary(self, paper: PaperMetadata) -> Optional[str]:
        """Extract summary of methodology section.

        Args:
            paper: Paper metadata

        Returns:
            Methods summary if found
        """
        # Look for methods/methodology section
        methods_section = None

        for section in paper.sections:
            title_lower = section.title.lower()
            if any(keyword in title_lower for keyword in ['method', 'approach', 'technique']):
                methods_section = section
                break

        if not methods_section:
            return None

        try:
            summary = self.model_manager.summarize_text(
                methods_section.content,
                max_length=200,
                min_length=50
            )
            return summary

        except Exception as e:
            logger.error(f"Error summarizing methods: {e}")
            return None

    def _extract_results_summary(self, paper: PaperMetadata) -> Optional[str]:
        """Extract summary of results section.

        Args:
            paper: Paper metadata

        Returns:
            Results summary if found
        """
        # Look for results/experiments section
        results_section = None

        for section in paper.sections:
            title_lower = section.title.lower()
            if any(keyword in title_lower for keyword in ['result', 'experiment', 'evaluation', 'finding']):
                results_section = section
                break

        if not results_section:
            return None

        try:
            summary = self.model_manager.summarize_text(
                results_section.content,
                max_length=200,
                min_length=50
            )
            return summary

        except Exception as e:
            logger.error(f"Error summarizing results: {e}")
            return None

    def _extract_limitations(self, paper: PaperMetadata) -> Optional[str]:
        """Extract limitations and future work.

        Args:
            paper: Paper metadata

        Returns:
            Limitations summary if found
        """
        # Look for limitations, discussion, or conclusion sections
        relevant_section = None

        for section in paper.sections:
            title_lower = section.title.lower()
            if any(keyword in title_lower for keyword in ['limitation', 'discussion', 'conclusion', 'future']):
                relevant_section = section
                break

        if not relevant_section:
            # Check last section
            if paper.sections:
                relevant_section = paper.sections[-1]

        if not relevant_section:
            return None

        try:
            summary = self.model_manager.summarize_text(
                relevant_section.content,
                max_length=150,
                min_length=30
            )
            return summary

        except Exception as e:
            logger.error(f"Error extracting limitations: {e}")
            return None
