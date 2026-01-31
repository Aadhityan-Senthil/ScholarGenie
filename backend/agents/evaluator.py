"""Evaluation agent for quality assessment."""

import logging
from typing import Dict, Any, Optional, List
import yaml
import numpy as np
from rouge_score import rouge_scorer
import bert_score

from backend.utils.metadata import PaperMetadata, Summary

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """Agent for evaluating summary and extraction quality."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator agent.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.eval_config = self.config["evaluation"]
        self.rouge_types = self.eval_config.get("rouge_types", ["rouge1", "rouge2", "rougeL"])

    def evaluate_summary(
        self,
        paper: PaperMetadata,
        summary: Summary
    ) -> Dict[str, Any]:
        """Evaluate summary quality.

        Args:
            paper: Original paper metadata
            summary: Generated summary

        Returns:
            Evaluation metrics and results
        """
        logger.info(f"Evaluating summary for: {paper.title}")

        results = {
            "paper_id": paper.paper_id,
            "metrics": {},
            "quality_checks": {},
            "warnings": []
        }

        # Compute ROUGE scores (using abstract as reference)
        if paper.abstract:
            rouge_scores = self._compute_rouge(
                reference=paper.abstract,
                hypothesis=summary.full_summary
            )
            results["metrics"]["rouge"] = rouge_scores

            # ROUGE for TL;DR
            rouge_tldr = self._compute_rouge(
                reference=paper.abstract,
                hypothesis=summary.tldr
            )
            results["metrics"]["rouge_tldr"] = rouge_tldr

        # Compute BERTScore
        if "bertscore" in self.eval_config.get("metrics", []):
            if paper.abstract:
                bert_scores = self._compute_bertscore(
                    reference=paper.abstract,
                    hypothesis=summary.full_summary
                )
                results["metrics"]["bertscore"] = bert_scores

        # Quality checks
        results["quality_checks"] = self._run_quality_checks(paper, summary)

        # Generate warnings
        results["warnings"] = self._generate_warnings(paper, summary, results["quality_checks"])

        return results

    def _compute_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            reference: Reference text
            hypothesis: Generated text

        Returns:
            Dictionary of ROUGE scores
        """
        try:
            scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
            scores = scorer.score(reference, hypothesis)

            # Extract F-measures
            rouge_scores = {}
            for metric in self.rouge_types:
                rouge_scores[metric] = {
                    "precision": scores[metric].precision,
                    "recall": scores[metric].recall,
                    "fmeasure": scores[metric].fmeasure
                }

            return rouge_scores

        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")
            return {}

    def _compute_bertscore(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Compute BERTScore.

        Args:
            reference: Reference text
            hypothesis: Generated text

        Returns:
            Dictionary of BERTScore metrics
        """
        try:
            model_type = self.config["evaluation"].get(
                "bertscore_model",
                "microsoft/deberta-base-mnli"
            )

            P, R, F1 = bert_score.score(
                [hypothesis],
                [reference],
                model_type=model_type,
                verbose=False
            )

            return {
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": float(F1[0])
            }

        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return {}

    def _run_quality_checks(
        self,
        paper: PaperMetadata,
        summary: Summary
    ) -> Dict[str, Any]:
        """Run quality checks on the summary.

        Args:
            paper: Original paper
            summary: Generated summary

        Returns:
            Dictionary of quality check results
        """
        checks = {}

        # Length checks
        tldr_words = len(summary.tldr.split())
        short_words = len(summary.short_summary.split())
        full_words = len(summary.full_summary.split())

        checks["tldr_length"] = {
            "word_count": tldr_words,
            "pass": 5 <= tldr_words <= 30
        }

        checks["short_summary_length"] = {
            "word_count": short_words,
            "pass": 20 <= short_words <= 100
        }

        checks["full_summary_length"] = {
            "word_count": full_words,
            "pass": 100 <= full_words <= 600
        }

        # Keypoint count check
        checks["keypoints_count"] = {
            "count": len(summary.keypoints),
            "pass": 5 <= len(summary.keypoints) <= 20
        }

        # Completeness check
        checks["has_methods"] = summary.methods is not None
        checks["has_results"] = summary.results is not None
        checks["has_section_summaries"] = len(summary.section_summaries) > 0

        # Hallucination detection (basic)
        hallucination_score = self._detect_hallucinations(paper, summary)
        checks["hallucination_score"] = {
            "score": hallucination_score,
            "pass": hallucination_score < 0.3  # Threshold
        }

        # Redundancy check
        redundancy_score = self._check_redundancy(summary)
        checks["redundancy_score"] = {
            "score": redundancy_score,
            "pass": redundancy_score < 0.5
        }

        return checks

    def _detect_hallucinations(
        self,
        paper: PaperMetadata,
        summary: Summary
    ) -> float:
        """Detect potential hallucinations in summary.

        Args:
            paper: Original paper
            summary: Generated summary

        Returns:
            Hallucination score (0-1, higher = more likely hallucination)
        """
        # Simple heuristic: check if key terms in summary appear in original text
        full_text = paper.get_full_text().lower()
        summary_text = summary.full_summary.lower()

        # Extract important words from summary (nouns, numbers)
        import re

        # Extract capitalized words and numbers
        important_words = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\.?\d*\b', summary.full_summary)

        if not important_words:
            return 0.0

        # Check how many appear in original
        found_count = sum(1 for word in important_words if word.lower() in full_text)

        # Hallucination score = proportion not found
        hallucination_score = 1.0 - (found_count / len(important_words))

        return hallucination_score

    def _check_redundancy(self, summary: Summary) -> float:
        """Check for redundancy in summary.

        Args:
            summary: Generated summary

        Returns:
            Redundancy score (0-1, higher = more redundant)
        """
        # Simple check: compare similarity between different summary levels
        from difflib import SequenceMatcher

        tldr_words = set(summary.tldr.lower().split())
        short_words = set(summary.short_summary.lower().split())
        full_words = set(summary.full_summary.lower().split())

        # Check overlap
        tldr_short_overlap = len(tldr_words & short_words) / max(len(tldr_words), 1)
        short_full_overlap = len(short_words & full_words) / max(len(short_words), 1)

        # Average overlap (high overlap is expected, but not 100%)
        avg_overlap = (tldr_short_overlap + short_full_overlap) / 2

        # Redundancy is high if overlap is very high (>0.8) or very low (<0.3)
        if avg_overlap > 0.8:
            return 0.8
        elif avg_overlap < 0.3:
            return 0.7
        else:
            return 0.2  # Good balance

    def _generate_warnings(
        self,
        paper: PaperMetadata,
        summary: Summary,
        quality_checks: Dict[str, Any]
    ) -> List[str]:
        """Generate warnings based on quality checks.

        Args:
            paper: Original paper
            summary: Generated summary
            quality_checks: Results from quality checks

        Returns:
            List of warning messages
        """
        warnings = []

        # Length warnings
        if not quality_checks["tldr_length"]["pass"]:
            warnings.append(
                f"TL;DR length unusual: {quality_checks['tldr_length']['word_count']} words"
            )

        if not quality_checks["full_summary_length"]["pass"]:
            warnings.append(
                f"Full summary length unusual: {quality_checks['full_summary_length']['word_count']} words"
            )

        # Completeness warnings
        if not quality_checks["has_methods"]:
            warnings.append("Methods summary is missing")

        if not quality_checks["has_results"]:
            warnings.append("Results summary is missing")

        # Hallucination warning
        if not quality_checks["hallucination_score"]["pass"]:
            warnings.append(
                f"Potential hallucinations detected (score: {quality_checks['hallucination_score']['score']:.2f})"
            )

        # Redundancy warning
        if not quality_checks["redundancy_score"]["pass"]:
            warnings.append(
                f"High redundancy detected (score: {quality_checks['redundancy_score']['score']:.2f})"
            )

        return warnings

    def compare_summaries(
        self,
        summary1: Summary,
        summary2: Summary
    ) -> Dict[str, Any]:
        """Compare two summaries.

        Args:
            summary1: First summary
            summary2: Second summary

        Returns:
            Comparison results
        """
        comparison = {
            "rouge_scores": {},
            "length_comparison": {},
            "keypoints_comparison": {}
        }

        # ROUGE between summaries
        rouge_scores = self._compute_rouge(
            reference=summary1.full_summary,
            hypothesis=summary2.full_summary
        )
        comparison["rouge_scores"] = rouge_scores

        # Length comparison
        comparison["length_comparison"] = {
            "summary1_words": len(summary1.full_summary.split()),
            "summary2_words": len(summary2.full_summary.split()),
            "difference": abs(
                len(summary1.full_summary.split()) - len(summary2.full_summary.split())
            )
        }

        # Keypoints comparison
        comparison["keypoints_comparison"] = {
            "summary1_count": len(summary1.keypoints),
            "summary2_count": len(summary2.keypoints),
            "overlap": len(set(summary1.keypoints) & set(summary2.keypoints))
        }

        return comparison

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report.

        Args:
            evaluation_results: Results from evaluate_summary

        Returns:
            Formatted report string
        """
        report = "# Summary Evaluation Report\n\n"

        # Metrics
        if "rouge" in evaluation_results["metrics"]:
            report += "## ROUGE Scores\n\n"
            for metric, scores in evaluation_results["metrics"]["rouge"].items():
                report += f"**{metric.upper()}:**\n"
                report += f"- Precision: {scores['precision']:.3f}\n"
                report += f"- Recall: {scores['recall']:.3f}\n"
                report += f"- F-measure: {scores['fmeasure']:.3f}\n\n"

        if "bertscore" in evaluation_results["metrics"]:
            report += "## BERTScore\n\n"
            bs = evaluation_results["metrics"]["bertscore"]
            report += f"- Precision: {bs['precision']:.3f}\n"
            report += f"- Recall: {bs['recall']:.3f}\n"
            report += f"- F1: {bs['f1']:.3f}\n\n"

        # Quality Checks
        report += "## Quality Checks\n\n"
        qc = evaluation_results["quality_checks"]

        report += f"- TL;DR Length: {qc['tldr_length']['word_count']} words "
        report += f"({'✓' if qc['tldr_length']['pass'] else '✗'})\n"

        report += f"- Full Summary Length: {qc['full_summary_length']['word_count']} words "
        report += f"({'✓' if qc['full_summary_length']['pass'] else '✗'})\n"

        report += f"- Keypoints Count: {qc['keypoints_count']['count']} "
        report += f"({'✓' if qc['keypoints_count']['pass'] else '✗'})\n"

        report += f"- Has Methods: {'✓' if qc['has_methods'] else '✗'}\n"
        report += f"- Has Results: {'✓' if qc['has_results'] else '✗'}\n"

        report += f"- Hallucination Score: {qc['hallucination_score']['score']:.3f} "
        report += f"({'✓' if qc['hallucination_score']['pass'] else '✗'})\n"

        report += f"- Redundancy Score: {qc['redundancy_score']['score']:.3f} "
        report += f"({'✓' if qc['redundancy_score']['pass'] else '✗'})\n\n"

        # Warnings
        if evaluation_results["warnings"]:
            report += "## Warnings\n\n"
            for warning in evaluation_results["warnings"]:
                report += f"- ⚠️  {warning}\n"

        return report
