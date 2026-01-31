"""Information extraction agent for structured data."""

import re
import logging
from typing import List, Dict, Any, Optional
import yaml

from backend.utils.metadata import PaperMetadata, ExtractedData

logger = logging.getLogger(__name__)


class ExtractorAgent:
    """Agent for extracting structured information from papers."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize extractor agent.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def extract(self, paper: PaperMetadata) -> ExtractedData:
        """Extract structured data from paper.

        Args:
            paper: Paper metadata

        Returns:
            Extracted structured data
        """
        logger.info(f"Extracting structured data from: {paper.title}")

        full_text = paper.get_full_text()

        return ExtractedData(
            paper_id=paper.paper_id,
            research_question=self._extract_research_question(full_text),
            hypothesis=self._extract_hypothesis(full_text),
            methods=self._extract_methods(paper),
            datasets=self._extract_datasets(full_text),
            models=self._extract_models(full_text),
            key_findings=self._extract_findings(paper),
            metrics=self._extract_metrics(full_text),
            limitations=self._extract_limitations_list(paper),
            future_work=self._extract_future_work(paper)
        )

    def _extract_research_question(self, text: str) -> Optional[str]:
        """Extract research question using patterns.

        Args:
            text: Paper text

        Returns:
            Research question if found
        """
        patterns = [
            r'(?i)(?:research question|we (?:ask|investigate|explore))[\s:]+([^.]+\.)',
            r'(?i)(?:this (?:paper|work) (?:addresses|investigates))[\s:]+([^.]+\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _extract_hypothesis(self, text: str) -> Optional[str]:
        """Extract hypothesis.

        Args:
            text: Paper text

        Returns:
            Hypothesis if found
        """
        patterns = [
            r'(?i)(?:we hypothesize|our hypothesis|hypothesis)[\s:]+([^.]+\.)',
            r'(?i)(?:we (?:expect|predict|propose) that)[\s:]+([^.]+\.)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        return None

    def _extract_methods(self, paper: PaperMetadata) -> List[str]:
        """Extract methodology mentions.

        Args:
            paper: Paper metadata

        Returns:
            List of methods
        """
        methods = []

        # Find methods section
        methods_section = paper.get_section_by_title("methods")
        if not methods_section:
            methods_section = paper.get_section_by_title("methodology")
        if not methods_section:
            methods_section = paper.get_section_by_title("approach")

        if methods_section:
            text = methods_section.content
        else:
            text = paper.get_full_text()

        # Common ML/AI methods
        method_keywords = [
            r'\b(?:neural network|CNN|RNN|LSTM|GRU|transformer|attention|BERT|GPT)\b',
            r'\b(?:SVM|random forest|decision tree|gradient boosting|XGBoost)\b',
            r'\b(?:regression|classification|clustering|reinforcement learning)\b',
            r'\b(?:supervised|unsupervised|semi-supervised) learning\b',
            r'\b(?:deep learning|machine learning|transfer learning)\b',
        ]

        for pattern in method_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            methods.extend([m.strip() for m in matches])

        # Deduplicate and clean
        methods = list(set([m.lower() for m in methods]))
        return methods[:20]  # Limit to top 20

    def _extract_datasets(self, text: str) -> List[str]:
        """Extract dataset mentions.

        Args:
            text: Paper text

        Returns:
            List of datasets
        """
        datasets = []

        # Common dataset patterns
        dataset_patterns = [
            r'\b([A-Z][A-Za-z0-9]+(?:-[A-Z0-9]+)*)\s+dataset\b',
            r'\b(?:dataset|corpus)[\s:]+([A-Z][A-Za-z0-9-]+)\b',
            r'\b(ImageNet|COCO|MNIST|CIFAR|WikiText|SQuAD|GLUE|SuperGLUE)\b',
        ]

        for pattern in dataset_patterns:
            matches = re.findall(pattern, text)
            datasets.extend(matches)

        # Deduplicate
        datasets = list(set(datasets))
        return datasets[:15]

    def _extract_models(self, text: str) -> List[str]:
        """Extract model/architecture mentions.

        Args:
            text: Paper text

        Returns:
            List of models
        """
        models = []

        # Common model names
        model_patterns = [
            r'\b(BERT|GPT-[0-9]|T5|RoBERTa|ELECTRA|ALBERT|XLNet|DistilBERT)\b',
            r'\b(ResNet|VGG|Inception|EfficientNet|MobileNet|DenseNet)\b',
            r'\b(LSTM|GRU|Transformer|Attention)\b',
            r'\b([A-Z][a-z]+(?:-[A-Z0-9]+)*)\s+(?:model|architecture|network)\b',
        ]

        for pattern in model_patterns:
            matches = re.findall(pattern, text)
            if isinstance(matches[0] if matches else None, tuple):
                models.extend([m[0] for m in matches])
            else:
                models.extend(matches)

        # Deduplicate
        models = list(set(models))
        return models[:15]

    def _extract_findings(self, paper: PaperMetadata) -> List[str]:
        """Extract key findings.

        Args:
            paper: Paper metadata

        Returns:
            List of findings
        """
        findings = []

        # Find results section
        results_section = paper.get_section_by_title("results")
        if not results_section:
            results_section = paper.get_section_by_title("experiments")
        if not results_section:
            results_section = paper.get_section_by_title("evaluation")

        if results_section:
            text = results_section.content
        else:
            # Use abstract as fallback
            text = paper.abstract if paper.abstract else ""

        # Extract sentences with performance indicators
        patterns = [
            r'(?i)(?:we (?:find|found|show|demonstrate|observe)|our (?:results|experiments) show)[\s:]+([^.]+\.)',
            r'(?i)(?:achieves?|outperforms?|improves?)[\s:]+([^.]+\.)',
            r'(?i)(?:accuracy|precision|recall|F1|performance|score) of ([^.]+\.)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            findings.extend([m.strip() for m in matches])

        return findings[:10]

    def _extract_metrics(self, text: str) -> Dict[str, Any]:
        """Extract performance metrics.

        Args:
            text: Paper text

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Metric patterns (metric name and value)
        patterns = [
            r'(?i)(accuracy|precision|recall|F1[-\s]score)[\s:]+([0-9.]+)%?',
            r'(?i)(BLEU|ROUGE|METEOR)[-\s]?([0-9.]+)',
            r'(?i)(perplexity|loss)[\s:]+([0-9.]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for metric_name, value in matches:
                try:
                    metrics[metric_name.lower()] = float(value)
                except:
                    pass

        return metrics

    def _extract_limitations_list(self, paper: PaperMetadata) -> List[str]:
        """Extract limitations as a list.

        Args:
            paper: Paper metadata

        Returns:
            List of limitations
        """
        limitations = []

        # Find limitations section
        lim_section = paper.get_section_by_title("limitations")
        if not lim_section:
            lim_section = paper.get_section_by_title("discussion")

        if lim_section:
            text = lim_section.content

            # Extract sentences mentioning limitations
            patterns = [
                r'(?i)(?:limitation|drawback|weakness|challenge)[\s:]+([^.]+\.)',
                r'(?i)(?:however|unfortunately|cannot|unable to)[\s:]+([^.]+\.)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text)
                limitations.extend([m.strip() for m in matches])

        return limitations[:8]

    def _extract_future_work(self, paper: PaperMetadata) -> List[str]:
        """Extract future work directions.

        Args:
            paper: Paper metadata

        Returns:
            List of future work items
        """
        future_work = []

        # Check conclusion/future work sections
        future_section = paper.get_section_by_title("future work")
        if not future_section:
            future_section = paper.get_section_by_title("conclusion")

        if future_section:
            text = future_section.content

            # Extract future-oriented sentences
            patterns = [
                r'(?i)(?:future (?:work|research|direction)|plan to|intend to)[\s:]+([^.]+\.)',
                r'(?i)(?:could|should|would) (?:be|explore|investigate)[\s:]+([^.]+\.)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text)
                future_work.extend([m.strip() for m in matches])

        return future_work[:8]

    def extract_tables(self, paper: PaperMetadata) -> List[Dict[str, Any]]:
        """Extract tables from paper (basic version).

        Args:
            paper: Paper metadata

        Returns:
            List of table metadata
        """
        # This is a placeholder for table extraction
        # Full implementation would require more sophisticated PDF parsing
        tables = []

        full_text = paper.get_full_text()

        # Look for table references
        table_pattern = r'(?i)table\s+(\d+)(?::|\s)([^\n]+)'
        matches = re.findall(table_pattern, full_text)

        for table_num, caption in matches:
            tables.append({
                "number": int(table_num),
                "caption": caption.strip(),
                "data": None  # Would need actual extraction
            })

        return tables

    def extract_equations(self, paper: PaperMetadata) -> List[str]:
        """Extract equations from paper (basic version).

        Args:
            paper: Paper metadata

        Returns:
            List of equation strings
        """
        # This is a placeholder
        # Full implementation would parse LaTeX math
        equations = []

        # This would need proper math extraction
        return equations
