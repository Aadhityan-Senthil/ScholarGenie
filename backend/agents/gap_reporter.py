"""Research Gap Report Generator.

Automatically generates comprehensive research gap reports in
multiple formats (Markdown, PDF, HTML) with visualizations.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import os

from backend.agents.gap_discovery import ResearchGap, GapDiscoveryAgent
from backend.agents.llm_reasoner import LLMReasoner, GapValidation
from backend.agents.causal_reasoning import CausalGraphReasoner, BreakthroughPrediction
from backend.agents.knowledge_graph import KnowledgeGraphAgent


logger = logging.getLogger(__name__)


class GapReportGenerator:
    """Generates comprehensive research gap reports."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphAgent,
        gap_discovery: GapDiscoveryAgent,
        llm_reasoner: LLMReasoner,
        causal_reasoner: Optional[CausalGraphReasoner] = None
    ):
        """Initialize gap report generator.

        Args:
            knowledge_graph: Knowledge graph instance
            gap_discovery: Gap discovery agent
            llm_reasoner: LLM reasoning agent
            causal_reasoner: Optional causal reasoning engine
        """
        self.kg = knowledge_graph
        self.gap_discovery = gap_discovery
        self.llm_reasoner = llm_reasoner
        self.causal_reasoner = causal_reasoner

        logger.info("GapReportGenerator initialized")

    def generate_full_report(
        self,
        output_path: str,
        format: str = "markdown",
        include_visualizations: bool = True
    ) -> str:
        """Generate comprehensive gap analysis report.

        Args:
            output_path: Output file path
            format: Report format ("markdown", "html", "pdf")
            include_visualizations: Whether to include charts

        Returns:
            Path to generated report
        """
        logger.info(f"Generating full gap report in {format} format...")

        # Discover gaps
        gaps = self.gap_discovery.discover_all_gaps()

        # Validate gaps
        validations = []
        for gap in gaps[:20]:  # Validate top 20
            validation = self.llm_reasoner.validate_gap(gap)
            validations.append(validation)

        # Get breakthrough predictions if causal reasoner available
        breakthroughs = []
        if self.causal_reasoner:
            breakthroughs = self.causal_reasoner.predict_breakthroughs(top_k=10)

        # Generate report content
        if format == "markdown":
            content = self._generate_markdown_report(gaps, validations, breakthroughs)
        elif format == "html":
            content = self._generate_html_report(gaps, validations, breakthroughs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Report generated: {output_path}")
        return output_path

    def _generate_markdown_report(
        self,
        gaps: List[ResearchGap],
        validations: List[GapValidation],
        breakthroughs: List[BreakthroughPrediction]
    ) -> str:
        """Generate Markdown format report.

        Args:
            gaps: List of research gaps
            validations: List of gap validations
            breakthroughs: List of breakthrough predictions

        Returns:
            Markdown content
        """
        md = []

        # Title and metadata
        md.append("# Research Gap Analysis Report\n")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append(f"**Total Gaps Discovered:** {len(gaps)}\n")
        md.append(f"**Validated Gaps:** {sum(1 for v in validations if v.is_valid)}\n")
        md.append(f"**Breakthrough Opportunities:** {len(breakthroughs)}\n")
        md.append("\n---\n\n")

        # Executive Summary
        md.append("## Executive Summary\n\n")
        md.append(self._generate_executive_summary(gaps, validations, breakthroughs))
        md.append("\n")

        # Knowledge Graph Statistics
        md.append("## Knowledge Graph Overview\n\n")
        kg_stats = self.kg.get_graph_statistics()
        md.append(f"- **Total Entities:** {kg_stats['total_nodes']}\n")
        md.append(f"- **Total Relationships:** {kg_stats['total_edges']}\n")
        md.append(f"- **Entity Types:**\n")
        for entity_type, count in kg_stats['node_types'].items():
            md.append(f"  - {entity_type}: {count}\n")
        md.append("\n")

        # Top Gaps by Category
        md.append("## Discovered Research Gaps\n\n")

        # Group gaps by type
        gaps_by_type = {}
        for gap in gaps:
            if gap.gap_type not in gaps_by_type:
                gaps_by_type[gap.gap_type] = []
            gaps_by_type[gap.gap_type].append(gap)

        for gap_type, type_gaps in gaps_by_type.items():
            md.append(f"### {gap_type.replace('_', ' ').title()}\n\n")

            for i, gap in enumerate(type_gaps[:5], 1):  # Top 5 per type
                md.append(f"#### {i}. {gap.title}\n\n")
                md.append(f"**Description:** {gap.description}\n\n")
                md.append(f"**Confidence:** {gap.confidence:.2f} | **Impact:** {gap.potential_impact}\n\n")

                if gap.supporting_evidence:
                    md.append("**Evidence:**\n")
                    for evidence in gap.supporting_evidence[:3]:
                        md.append(f"- {evidence}\n")
                    md.append("\n")

        # Validated Gaps
        md.append("## Gap Validation Results\n\n")

        valid_gaps = [v for v in validations if v.is_valid]
        md.append(f"**Valid Gaps:** {len(valid_gaps)} out of {len(validations)} analyzed\n\n")

        if valid_gaps:
            md.append("### Top Validated Gaps\n\n")
            # Sort by combined score
            valid_gaps.sort(
                key=lambda v: (v.novelty_score + v.impact_score + v.feasibility_score) / 3,
                reverse=True
            )

            for i, validation in enumerate(valid_gaps[:10], 1):
                md.append(f"#### {i}. Gap {validation.gap_id}\n\n")
                md.append(f"**Novelty:** {validation.novelty_score:.2f} | ")
                md.append(f"**Impact:** {validation.impact_score:.2f} | ")
                md.append(f"**Feasibility:** {validation.feasibility_score:.2f}\n\n")
                md.append(f"**Assessment:** {validation.explanation}\n\n")

                if validation.recommendations:
                    md.append("**Recommendations:**\n")
                    for rec in validation.recommendations:
                        md.append(f"- {rec}\n")
                    md.append("\n")

        # Breakthrough Predictions
        if breakthroughs:
            md.append("## Breakthrough Opportunities\n\n")
            md.append(f"Identified {len(breakthroughs)} potential breakthrough opportunities ")
            md.append("using causal reasoning.\n\n")

            for i, breakthrough in enumerate(breakthroughs[:10], 1):
                md.append(f"### {i}. {breakthrough.title}\n\n")
                md.append(f"**Description:** {breakthrough.description}\n\n")
                md.append(f"**Expected Impact:** {breakthrough.expected_impact:.2f} | ")
                md.append(f"**Confidence:** {breakthrough.confidence:.2f} | ")
                md.append(f"**Timeline:** {breakthrough.timeline}\n\n")

                if breakthrough.causal_chain:
                    md.append("**Causal Chain:**\n")
                    for source, relation, target in breakthrough.causal_chain:
                        md.append(f"- {source} → {relation} → {target}\n")
                    md.append("\n")

                md.append(f"**Reasoning:** {breakthrough.reasoning}\n\n")

                if breakthrough.prerequisites:
                    md.append("**Prerequisites:**\n")
                    for prereq in breakthrough.prerequisites:
                        md.append(f"- {prereq}\n")
                    md.append("\n")

        # Recommendations
        md.append("## Strategic Recommendations\n\n")
        md.append(self._generate_strategic_recommendations(gaps, validations, breakthroughs))
        md.append("\n")

        # Research Agenda
        md.append("## Proposed Research Agenda\n\n")
        agenda = self.llm_reasoner.generate_research_agenda(validations, top_k=5)

        md.append(f"**Priority Gaps:** {len(agenda['priority_gaps'])}\n\n")

        for gap in agenda['priority_gaps']:
            md.append(f"### Priority {gap['rank']}: Gap {gap['gap_id']}\n\n")
            md.append(f"**Scores:** Novelty: {gap['novelty']:.2f}, ")
            md.append(f"Impact: {gap['impact']:.2f}, ")
            md.append(f"Feasibility: {gap['feasibility']:.2f}\n\n")
            md.append(f"{gap['explanation']}\n\n")

        # Next Steps
        md.append("## Next Steps\n\n")
        if agenda['next_steps']:
            for step in agenda['next_steps']:
                md.append(f"1. {step}\n")
        md.append("\n")

        # Appendix
        md.append("## Appendix\n\n")
        md.append("### Methodology\n\n")
        md.append("This report was generated using:\n")
        md.append("- **Knowledge Graph Construction:** Entity extraction and relationship mapping\n")
        md.append("- **Gap Discovery:** Graph algorithms, semantic clustering, and temporal analysis\n")
        md.append("- **LLM Validation:** Novelty, impact, and feasibility assessment\n")
        md.append("- **Causal Reasoning:** Causal graph analysis for breakthrough prediction\n")
        md.append("\n")

        return "".join(md)

    def _generate_html_report(
        self,
        gaps: List[ResearchGap],
        validations: List[GapValidation],
        breakthroughs: List[BreakthroughPrediction]
    ) -> str:
        """Generate HTML format report.

        Args:
            gaps: List of research gaps
            validations: List of gap validations
            breakthroughs: List of breakthrough predictions

        Returns:
            HTML content
        """
        html = []

        # HTML header
        html.append("<!DOCTYPE html>\n")
        html.append("<html lang='en'>\n")
        html.append("<head>\n")
        html.append("  <meta charset='UTF-8'>\n")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
        html.append("  <title>Research Gap Analysis Report</title>\n")
        html.append("  <style>\n")
        html.append(self._get_html_styles())
        html.append("  </style>\n")
        html.append("</head>\n")
        html.append("<body>\n")

        # Content
        html.append("  <div class='container'>\n")
        html.append("    <h1>Research Gap Analysis Report</h1>\n")
        html.append(f"    <p class='meta'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")

        # Statistics cards
        html.append("    <div class='stats-grid'>\n")
        html.append(f"      <div class='stat-card'>\n")
        html.append(f"        <div class='stat-value'>{len(gaps)}</div>\n")
        html.append(f"        <div class='stat-label'>Gaps Discovered</div>\n")
        html.append(f"      </div>\n")
        html.append(f"      <div class='stat-card'>\n")
        html.append(f"        <div class='stat-value'>{sum(1 for v in validations if v.is_valid)}</div>\n")
        html.append(f"        <div class='stat-label'>Valid Gaps</div>\n")
        html.append(f"      </div>\n")
        html.append(f"      <div class='stat-card'>\n")
        html.append(f"        <div class='stat-value'>{len(breakthroughs)}</div>\n")
        html.append(f"        <div class='stat-label'>Breakthroughs</div>\n")
        html.append(f"      </div>\n")
        html.append("    </div>\n")

        # Executive Summary
        html.append("    <section>\n")
        html.append("      <h2>Executive Summary</h2>\n")
        summary = self._generate_executive_summary(gaps, validations, breakthroughs)
        for paragraph in summary.split('\n'):
            if paragraph.strip():
                html.append(f"      <p>{paragraph}</p>\n")
        html.append("    </section>\n")

        # Top gaps
        html.append("    <section>\n")
        html.append("      <h2>Top Research Gaps</h2>\n")
        for i, gap in enumerate(gaps[:10], 1):
            html.append("      <div class='gap-card'>\n")
            html.append(f"        <h3>{i}. {gap.title}</h3>\n")
            html.append(f"        <p>{gap.description}</p>\n")
            html.append(f"        <div class='gap-meta'>\n")
            html.append(f"          <span class='badge'>{gap.gap_type}</span>\n")
            html.append(f"          <span class='badge'>{gap.potential_impact} impact</span>\n")
            html.append(f"          <span class='badge'>Confidence: {gap.confidence:.2f}</span>\n")
            html.append(f"        </div>\n")
            html.append("      </div>\n")
        html.append("    </section>\n")

        # Breakthrough opportunities
        if breakthroughs:
            html.append("    <section>\n")
            html.append("      <h2>Breakthrough Opportunities</h2>\n")
            for i, breakthrough in enumerate(breakthroughs[:5], 1):
                html.append("      <div class='breakthrough-card'>\n")
                html.append(f"        <h3>{i}. {breakthrough.title}</h3>\n")
                html.append(f"        <p>{breakthrough.description}</p>\n")
                html.append(f"        <p><strong>Reasoning:</strong> {breakthrough.reasoning}</p>\n")
                html.append(f"        <div class='gap-meta'>\n")
                html.append(f"          <span class='badge'>Impact: {breakthrough.expected_impact:.2f}</span>\n")
                html.append(f"          <span class='badge'>Timeline: {breakthrough.timeline}</span>\n")
                html.append(f"        </div>\n")
                html.append("      </div>\n")
            html.append("    </section>\n")

        html.append("  </div>\n")
        html.append("</body>\n")
        html.append("</html>\n")

        return "".join(html)

    def _get_html_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      line-height: 1.6;
      color: #333;
      background: #f5f5f5;
      margin: 0;
      padding: 20px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      background: white;
      padding: 40px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    h1 {
      color: #2c3e50;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
    }
    h2 {
      color: #34495e;
      margin-top: 40px;
    }
    h3 {
      color: #555;
    }
    .meta {
      color: #7f8c8d;
      font-size: 0.9em;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin: 30px 0;
    }
    .stat-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 30px;
      border-radius: 8px;
      text-align: center;
    }
    .stat-value {
      font-size: 3em;
      font-weight: bold;
    }
    .stat-label {
      font-size: 1.1em;
      opacity: 0.9;
    }
    .gap-card, .breakthrough-card {
      background: #f8f9fa;
      border-left: 4px solid #3498db;
      padding: 20px;
      margin: 20px 0;
      border-radius: 4px;
    }
    .breakthrough-card {
      border-left-color: #e74c3c;
    }
    .gap-meta {
      margin-top: 15px;
    }
    .badge {
      display: inline-block;
      background: #ecf0f1;
      color: #2c3e50;
      padding: 5px 12px;
      border-radius: 12px;
      font-size: 0.85em;
      margin-right: 10px;
    }
    section {
      margin: 40px 0;
    }
"""

    def _generate_executive_summary(
        self,
        gaps: List[ResearchGap],
        validations: List[GapValidation],
        breakthroughs: List[BreakthroughPrediction]
    ) -> str:
        """Generate executive summary.

        Args:
            gaps: Research gaps
            validations: Gap validations
            breakthroughs: Breakthrough predictions

        Returns:
            Summary text
        """
        summary_parts = []

        # Overview
        valid_count = sum(1 for v in validations if v.is_valid)
        summary_parts.append(
            f"This research gap analysis identified {len(gaps)} potential research opportunities "
            f"in the analyzed knowledge base. Through rigorous validation, {valid_count} gaps "
            f"were confirmed as novel, impactful, and feasible research directions."
        )

        # Gap types
        gap_types = {}
        for gap in gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1

        top_types = sorted(gap_types.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_types:
            types_str = ", ".join([f"{t.replace('_', ' ')} ({c})" for t, c in top_types])
            summary_parts.append(
                f"The most common gap types are: {types_str}."
            )

        # Breakthrough potential
        if breakthroughs:
            high_impact = sum(1 for b in breakthroughs if b.expected_impact > 0.7)
            summary_parts.append(
                f"Causal reasoning identified {len(breakthroughs)} breakthrough opportunities, "
                f"with {high_impact} classified as high-impact."
            )

        # Key finding
        if validations:
            top_validation = max(validations, key=lambda v: (v.novelty_score + v.impact_score) / 2)
            if top_validation.is_valid:
                summary_parts.append(
                    f"The highest-priority gap combines strong novelty ({top_validation.novelty_score:.2f}) "
                    f"and significant impact potential ({top_validation.impact_score:.2f})."
                )

        return "\n\n".join(summary_parts)

    def _generate_strategic_recommendations(
        self,
        gaps: List[ResearchGap],
        validations: List[GapValidation],
        breakthroughs: List[BreakthroughPrediction]
    ) -> str:
        """Generate strategic recommendations.

        Args:
            gaps: Research gaps
            validations: Gap validations
            breakthroughs: Breakthrough predictions

        Returns:
            Recommendations text
        """
        recs = []

        # Priority gaps
        valid_gaps = [v for v in validations if v.is_valid]
        if valid_gaps:
            top_gap = max(valid_gaps, key=lambda v: (v.novelty_score + v.impact_score + v.feasibility_score) / 3)
            recs.append(
                f"1. **Prioritize high-scoring gaps:** Focus on gaps with combined scores above 0.6. "
                f"The top gap has a combined score of "
                f"{(top_gap.novelty_score + top_gap.impact_score + top_gap.feasibility_score) / 3:.2f}."
            )

        # Collaboration
        isolated_gaps = [g for g in gaps if g.gap_type == "isolated_cluster"]
        if isolated_gaps:
            recs.append(
                f"2. **Foster cross-domain collaboration:** {len(isolated_gaps)} gaps involve isolated "
                f"research clusters that would benefit from interdisciplinary collaboration."
            )

        # Emerging trends
        emerging_gaps = [g for g in gaps if g.gap_type == "emerging_trend"]
        if emerging_gaps:
            recs.append(
                f"3. **Capitalize on emerging trends:** {len(emerging_gaps)} gaps represent emerging "
                f"research areas with high growth potential."
            )

        # Breakthroughs
        if breakthroughs:
            short_term = [b for b in breakthroughs if b.timeline == "short-term"]
            if short_term:
                recs.append(
                    f"4. **Pursue short-term breakthroughs:** {len(short_term)} breakthrough opportunities "
                    f"can be pursued in the short term with existing resources."
                )

        # Resource allocation
        recs.append(
            "5. **Allocate resources strategically:** Invest in gaps with high feasibility scores first "
            "to build momentum, then tackle higher-risk, higher-impact opportunities."
        )

        return "\n\n".join(recs)

    def generate_gap_summary(self, gap: ResearchGap) -> str:
        """Generate a summary for a single gap.

        Args:
            gap: Research gap

        Returns:
            Summary text
        """
        summary = f"**{gap.title}**\n\n"
        summary += f"Type: {gap.gap_type}\n"
        summary += f"Impact: {gap.potential_impact}\n"
        summary += f"Confidence: {gap.confidence:.2f}\n\n"
        summary += f"{gap.description}\n\n"

        if gap.supporting_evidence:
            summary += "Evidence:\n"
            for evidence in gap.supporting_evidence:
                summary += f"- {evidence}\n"

        return summary
