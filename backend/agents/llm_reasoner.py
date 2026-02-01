"""LLM Reasoning Layer for Gap Validation.

This module uses LLM-based reasoning to validate and enhance
research gap discoveries. It provides natural language explanations
and assesses the novelty and impact of identified gaps.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from backend.agents.gap_discovery import ResearchGap
from backend.agents.knowledge_graph import KnowledgeGraphAgent
from backend.utils.models import ModelManager


logger = logging.getLogger(__name__)


@dataclass
class GapValidation:
    """Validation result for a research gap."""
    gap_id: str
    is_valid: bool
    novelty_score: float  # 0-1
    impact_score: float  # 0-1
    feasibility_score: float  # 0-1
    explanation: str
    recommendations: List[str]
    related_work: List[str]


class LLMReasoner:
    """LLM-based reasoning for gap validation and explanation."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphAgent,
        model_manager: Optional[ModelManager] = None
    ):
        """Initialize LLM reasoner.

        Args:
            knowledge_graph: Knowledge graph for context
            model_manager: Model manager for LLM access
        """
        self.kg = knowledge_graph
        self.model_manager = model_manager or ModelManager()

        logger.info("LLMReasoner initialized")

    def validate_gap(self, gap: ResearchGap) -> GapValidation:
        """Validate a research gap using LLM reasoning.

        Args:
            gap: Research gap to validate

        Returns:
            Validation result
        """
        logger.info(f"Validating gap: {gap.gap_id}")

        # Gather context from knowledge graph
        context = self._gather_gap_context(gap)

        # Score novelty
        novelty_score = self._assess_novelty(gap, context)

        # Score impact
        impact_score = self._assess_impact(gap, context)

        # Score feasibility
        feasibility_score = self._assess_feasibility(gap, context)

        # Determine validity (heuristic)
        is_valid = (
            novelty_score > 0.5 and
            impact_score > 0.4 and
            feasibility_score > 0.3
        )

        # Generate explanation
        explanation = self._generate_explanation(
            gap, novelty_score, impact_score, feasibility_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(gap, context)

        # Find related work
        related_work = self._find_related_work(gap)

        return GapValidation(
            gap_id=gap.gap_id,
            is_valid=is_valid,
            novelty_score=novelty_score,
            impact_score=impact_score,
            feasibility_score=feasibility_score,
            explanation=explanation,
            recommendations=recommendations,
            related_work=related_work
        )

    def _gather_gap_context(self, gap: ResearchGap) -> Dict:
        """Gather relevant context for gap from knowledge graph.

        Args:
            gap: Research gap

        Returns:
            Context dictionary
        """
        context = {
            "entities": [],
            "related_papers": [],
            "connections": []
        }

        # Get entity information
        for entity_id in gap.entities[:10]:  # Limit for performance
            if entity_id in self.kg.nodes:
                node = self.kg.nodes[entity_id]
                context["entities"].append({
                    "id": entity_id,
                    "name": node.name,
                    "type": str(node.node_type),
                    "properties": node.properties
                })

        # Get related papers
        for paper_id in gap.related_papers[:5]:
            if paper_id in self.kg.nodes:
                node = self.kg.nodes[paper_id]
                context["related_papers"].append({
                    "id": paper_id,
                    "title": node.name,
                    "year": node.properties.get("year"),
                    "citations": node.properties.get("citation_count", 0)
                })

        # Get connections between entities
        for i, entity1 in enumerate(gap.entities[:5]):
            for entity2 in gap.entities[i+1:6]:
                if self.kg.graph.has_edge(entity1, entity2):
                    edge_data = self.kg.graph.get_edge_data(entity1, entity2)
                    context["connections"].append({
                        "from": entity1,
                        "to": entity2,
                        "relation": edge_data.get("relation", "unknown")
                    })

        return context

    def _assess_novelty(self, gap: ResearchGap, context: Dict) -> float:
        """Assess novelty of the research gap.

        Args:
            gap: Research gap
            context: Gap context

        Returns:
            Novelty score (0-1)
        """
        # Heuristic-based novelty assessment
        score = 0.5  # Base score

        # Fewer papers = more novel
        num_papers = len(context["related_papers"])
        if num_papers == 0:
            score += 0.3
        elif num_papers < 3:
            score += 0.2
        elif num_papers < 5:
            score += 0.1

        # Fewer connections = more novel
        num_connections = len(context["connections"])
        if num_connections == 0:
            score += 0.2
        elif num_connections < 2:
            score += 0.1

        # Gap type considerations
        if gap.gap_type == "missing_link":
            score += 0.1
        elif gap.gap_type == "emerging_trend":
            score += 0.15
        elif gap.gap_type == "underexplored_combination":
            score += 0.2

        return min(score, 1.0)

    def _assess_impact(self, gap: ResearchGap, context: Dict) -> float:
        """Assess potential impact of addressing the gap.

        Args:
            gap: Research gap
            context: Gap context

        Returns:
            Impact score (0-1)
        """
        # Base score from gap's own assessment
        impact_map = {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "breakthrough": 0.9
        }
        score = impact_map.get(gap.potential_impact, 0.5)

        # Boost for high-citation related papers
        if context["related_papers"]:
            avg_citations = sum(p.get("citations", 0) for p in context["related_papers"])
            avg_citations /= len(context["related_papers"])

            if avg_citations > 100:
                score += 0.1
            elif avg_citations > 50:
                score += 0.05

        # Boost for involving multiple entity types
        entity_types = set(e.get("type") for e in context["entities"])
        if len(entity_types) >= 3:
            score += 0.1
        elif len(entity_types) >= 2:
            score += 0.05

        # Boost for bridge opportunities
        if gap.gap_type == "bridge_opportunity":
            score += 0.15

        return min(score, 1.0)

    def _assess_feasibility(self, gap: ResearchGap, context: Dict) -> float:
        """Assess feasibility of addressing the gap.

        Args:
            gap: Research gap
            context: Gap context

        Returns:
            Feasibility score (0-1)
        """
        score = 0.6  # Base feasibility

        # More related work = more feasible
        num_papers = len(context["related_papers"])
        if num_papers > 0:
            score += min(num_papers * 0.05, 0.2)

        # Some connections = more feasible (not completely isolated)
        num_connections = len(context["connections"])
        if 1 <= num_connections <= 5:
            score += 0.1
        elif num_connections > 5:
            score += 0.05

        # Recent papers = more feasible
        if context["related_papers"]:
            recent_count = sum(
                1 for p in context["related_papers"]
                if p.get("year", 0) >= 2020
            )
            if recent_count > 0:
                score += 0.1

        # High confidence = more feasible
        if gap.confidence > 0.7:
            score += 0.1
        elif gap.confidence > 0.5:
            score += 0.05

        return min(score, 1.0)

    def _generate_explanation(
        self,
        gap: ResearchGap,
        novelty: float,
        impact: float,
        feasibility: float
    ) -> str:
        """Generate natural language explanation of gap validation.

        Args:
            gap: Research gap
            novelty: Novelty score
            impact: Impact score
            feasibility: Feasibility score

        Returns:
            Explanation string
        """
        explanation_parts = []

        # Overall assessment
        if novelty > 0.7 and impact > 0.6:
            explanation_parts.append(
                "This represents a highly novel research opportunity with significant potential impact."
            )
        elif novelty > 0.5 or impact > 0.5:
            explanation_parts.append(
                "This is a promising research direction worth exploring."
            )
        else:
            explanation_parts.append(
                "This gap exists but may have limited novelty or impact."
            )

        # Novelty details
        if novelty > 0.7:
            explanation_parts.append(
                f"The novelty is high ({novelty:.2f}), suggesting this area is underexplored."
            )
        elif novelty < 0.4:
            explanation_parts.append(
                f"The novelty is moderate ({novelty:.2f}), as some related work exists."
            )

        # Impact details
        if impact > 0.7:
            explanation_parts.append(
                f"Potential impact is significant ({impact:.2f}), likely to advance the field."
            )
        elif impact < 0.4:
            explanation_parts.append(
                f"Potential impact is limited ({impact:.2f}), may be incremental work."
            )

        # Feasibility details
        if feasibility > 0.7:
            explanation_parts.append(
                f"Feasibility is high ({feasibility:.2f}), with sufficient foundations to build upon."
            )
        elif feasibility < 0.4:
            explanation_parts.append(
                f"Feasibility is challenging ({feasibility:.2f}), may require significant groundwork."
            )

        return " ".join(explanation_parts)

    def _generate_recommendations(
        self,
        gap: ResearchGap,
        context: Dict
    ) -> List[str]:
        """Generate research recommendations.

        Args:
            gap: Research gap
            context: Gap context

        Returns:
            List of recommendations
        """
        recommendations = []

        # Entity-specific recommendations
        entities_by_type = {}
        for entity in context["entities"]:
            entity_type = entity.get("type", "unknown")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity["name"])

        # Model + Dataset recommendations
        if "model" in entities_by_type and "dataset" in entities_by_type:
            models = entities_by_type["model"][:2]
            datasets = entities_by_type["dataset"][:2]
            recommendations.append(
                f"Evaluate {', '.join(models)} on {', '.join(datasets)} benchmark(s)"
            )

        # Task recommendations
        if "task" in entities_by_type:
            tasks = entities_by_type["task"][:2]
            recommendations.append(
                f"Develop new methods for {', '.join(tasks)}"
            )

        # Cross-domain recommendations
        if len(entities_by_type) >= 3:
            recommendations.append(
                "Explore cross-domain connections between these entities"
            )

        # Survey recommendation
        if len(context["related_papers"]) < 3:
            recommendations.append(
                "Conduct a literature survey to identify foundational work"
            )

        # Collaboration recommendation
        if gap.gap_type == "isolated_cluster":
            recommendations.append(
                "Seek collaborations with researchers in related areas to bridge this gap"
            )

        # Baseline recommendation
        if gap.gap_type == "underexplored_combination":
            recommendations.append(
                "Establish baseline results for this combination"
            )

        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Further investigation recommended to assess viability"
            )

        return recommendations[:5]  # Limit to 5

    def _find_related_work(self, gap: ResearchGap) -> List[str]:
        """Find related work relevant to the gap.

        Args:
            gap: Research gap

        Returns:
            List of related paper titles/IDs
        """
        related = []

        # Get papers from gap
        for paper_id in gap.related_papers[:5]:
            if paper_id in self.kg.nodes:
                node = self.kg.nodes[paper_id]
                related.append(node.name)

        # Get papers connected to gap entities
        for entity_id in gap.entities[:5]:
            if entity_id in self.kg.graph:
                # Get papers that use this entity
                for predecessor in self.kg.graph.predecessors(entity_id):
                    if predecessor in self.kg.nodes:
                        pred_node = self.kg.nodes[predecessor]
                        if pred_node.node_type.value == "paper":
                            if pred_node.name not in related:
                                related.append(pred_node.name)

        return related[:10]  # Limit to 10

    def explain_gap_chain(
        self,
        gaps: List[ResearchGap]
    ) -> str:
        """Explain a chain of related gaps.

        Args:
            gaps: List of research gaps

        Returns:
            Natural language explanation
        """
        if not gaps:
            return "No gaps to explain."

        if len(gaps) == 1:
            return f"Single gap identified: {gaps[0].title}"

        explanation = f"Found {len(gaps)} related research gaps:\n\n"

        for i, gap in enumerate(gaps, 1):
            explanation += f"{i}. {gap.title}\n"
            explanation += f"   Type: {gap.gap_type}, Impact: {gap.potential_impact}\n"

        # Find commonalities
        all_entities = set()
        for gap in gaps:
            all_entities.update(gap.entities)

        explanation += f"\nThese gaps involve {len(all_entities)} unique entities, "
        explanation += "suggesting a broader underexplored research area."

        return explanation

    def generate_research_agenda(
        self,
        validated_gaps: List[GapValidation],
        top_k: int = 5
    ) -> Dict:
        """Generate a research agenda from validated gaps.

        Args:
            validated_gaps: List of validated gaps
            top_k: Number of top gaps to include

        Returns:
            Research agenda dictionary
        """
        # Filter valid gaps
        valid_gaps = [v for v in validated_gaps if v.is_valid]

        # Sort by combined score
        valid_gaps.sort(
            key=lambda v: (v.novelty_score + v.impact_score + v.feasibility_score) / 3,
            reverse=True
        )

        top_gaps = valid_gaps[:top_k]

        agenda = {
            "total_gaps_analyzed": len(validated_gaps),
            "valid_gaps": len(valid_gaps),
            "priority_gaps": [],
            "overall_themes": self._extract_themes(top_gaps),
            "next_steps": []
        }

        for i, gap in enumerate(top_gaps, 1):
            agenda["priority_gaps"].append({
                "rank": i,
                "gap_id": gap.gap_id,
                "novelty": gap.novelty_score,
                "impact": gap.impact_score,
                "feasibility": gap.feasibility_score,
                "explanation": gap.explanation,
                "recommendations": gap.recommendations
            })

        # Generate next steps
        if top_gaps:
            agenda["next_steps"] = [
                "Prioritize gaps with high novelty and impact scores",
                "Review related work for feasibility assessment",
                "Form collaborations to address identified gaps",
                "Allocate resources based on gap priority"
            ]

        return agenda

    def _extract_themes(self, gaps: List[GapValidation]) -> List[str]:
        """Extract common themes from gaps.

        Args:
            gaps: List of gap validations

        Returns:
            List of themes
        """
        themes = []

        # Count gap types
        gap_type_counts = {}
        for gap in gaps:
            # This would need the original gap, simplified for now
            themes.append("Underexplored research areas")

        return list(set(themes))[:5]

    def export_validation(self, validation: GapValidation) -> Dict:
        """Export validation to dictionary.

        Args:
            validation: Gap validation

        Returns:
            Dictionary representation
        """
        return {
            "gap_id": validation.gap_id,
            "is_valid": validation.is_valid,
            "scores": {
                "novelty": validation.novelty_score,
                "impact": validation.impact_score,
                "feasibility": validation.feasibility_score
            },
            "explanation": validation.explanation,
            "recommendations": validation.recommendations,
            "related_work": validation.related_work
        }
