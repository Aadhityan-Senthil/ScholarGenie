"""Analysis crew for knowledge graph and gap discovery."""

from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from backend.crews.base import BaseScholarGenieCrew
from backend.agents.knowledge_graph import KnowledgeGraphAgent
from backend.agents.gap_discovery import GapDiscoveryAgent
from backend.agents.llm_reasoner import LLMReasoner


class AnalysisCrew(BaseScholarGenieCrew):
    """
    Crew for deep knowledge analysis and gap discovery.

    Coordinates agents to:
    1. Build knowledge graphs
    2. Discover research gaps
    3. Reason about relationships
    4. Generate actionable insights
    """

    def __init__(self, **kwargs):
        """Initialize analysis crew."""
        super().__init__(**kwargs)

        # Initialize ScholarGenie agents
        self.kg_agent = KnowledgeGraphAgent()
        self.gap_discovery = GapDiscoveryAgent(self.kg_agent)
        self.llm_reasoner = LLMReasoner(self.kg_agent)

        # Create tools
        self.tools = self._create_tools()

        # Create CrewAI agents
        self.kg_builder = self._create_kg_builder()
        self.gap_analyst = self._create_gap_analyst()
        self.reasoning_agent = self._create_reasoning_agent()

    def _create_tools(self) -> List:
        """Create tools from ScholarGenie agents."""

        @tool("build_knowledge_graph")
        def build_kg(papers_data: str) -> str:
            """
            Build knowledge graph from papers.

            Args:
                papers_data: JSON string of papers

            Returns:
                Knowledge graph statistics
            """
            import json
            papers = json.loads(papers_data)
            self.kg_agent.build_from_papers(papers)
            stats = self.kg_agent.get_statistics()
            return str(stats)

        @tool("discover_gaps")
        def discover_gaps(method: str = "unexplored_pairs") -> str:
            """
            Discover research gaps in knowledge graph.

            Args:
                method: Gap discovery method

            Returns:
                List of discovered gaps
            """
            gaps = self.gap_discovery.discover_gaps([method])
            return str(gaps[:10])  # Top 10 gaps

        @tool("reason_about_relationships")
        def reason(entities: str) -> str:
            """
            Use LLM to reason about entity relationships.

            Args:
                entities: List of entities to analyze

            Returns:
                Inferred relationships and patterns
            """
            import json
            entity_list = json.loads(entities)
            patterns = self.llm_reasoner.find_semantic_patterns(entity_list[:5])
            return str(patterns)

        return [build_kg, discover_gaps, reason]

    def _create_kg_builder(self) -> Agent:
        """Create knowledge graph builder agent."""
        return self.create_agent(
            role="Knowledge Graph Architect",
            goal="Build comprehensive knowledge graphs from research papers",
            backstory="""You are an expert in knowledge representation and ontology design.
            You excel at extracting entities (concepts, methods, findings) and their
            relationships from scientific text. You understand how to structure knowledge
            for maximum queryability and insight generation.""",
            tools=[self.tools[0]],
            allow_delegation=False
        )

    def _create_gap_analyst(self) -> Agent:
        """Create gap discovery agent."""
        return self.create_agent(
            role="Research Gap Discovery Specialist",
            goal="Identify high-value research gaps and opportunities",
            backstory="""You are a research strategist who excels at identifying gaps
            in scientific knowledge. You can spot unexplored combinations, under-investigated
            areas, and emerging opportunities. You understand what makes a gap valuable:
            novelty, impact potential, and feasibility.""",
            tools=[self.tools[1]],
            allow_delegation=False
        )

    def _create_reasoning_agent(self) -> Agent:
        """Create reasoning specialist agent."""
        return self.create_agent(
            role="Knowledge Reasoning Expert",
            goal="Infer implicit relationships and discover hidden patterns",
            backstory="""You are a research analyst with strong logical reasoning skills.
            You can identify implicit connections between concepts, detect patterns across
            domains, and generate hypotheses about relationships. You think both inductively
            and deductively to uncover insights.""",
            tools=[self.tools[2]],
            allow_delegation=True
        )

    def analyze_research_landscape(
        self,
        papers: List[Dict[str, Any]],
        focus_area: str = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of research landscape.

        Args:
            papers: List of papers to analyze
            focus_area: Optional focus area for analysis

        Returns:
            Complete analysis with KG, gaps, and insights
        """
        # Create tasks
        kg_task = self.create_task(
            description=f"""Build a comprehensive knowledge graph from {len(papers)} research papers.

            Extract:
            - Key concepts and their definitions
            - Methods and techniques
            - Findings and results
            - Relationships between entities
            - Temporal evolution of concepts

            Organize the knowledge graph for maximum insight generation.
            {f'Focus especially on: {focus_area}' if focus_area else ''}""",
            agent=self.kg_builder,
            expected_output="Complete knowledge graph with statistics and key entities"
        )

        gap_task = self.create_task(
            description=f"""Discover high-value research gaps using multiple methods:

            1. Unexplored entity combinations
            2. Under-researched areas
            3. Methodological gaps
            4. Cross-domain opportunities
            5. Temporal gaps (emerging vs established)

            For each gap, assess:
            - Novelty (1-10)
            - Impact potential (1-10)
            - Feasibility (1-10)

            Prioritize gaps by overall value.
            {f'Focus on gaps related to: {focus_area}' if focus_area else ''}""",
            agent=self.gap_analyst,
            expected_output="Prioritized list of research gaps with assessments",
            context=[kg_task]
        )

        reasoning_task = self.create_task(
            description=f"""Analyze the knowledge graph to discover hidden patterns and insights:

            1. Identify implicit relationships not stated in papers
            2. Detect emerging themes and trends
            3. Find analogies across domains
            4. Generate novel hypotheses
            5. Suggest interdisciplinary connections

            Create a strategic research intelligence report with actionable insights.
            {f'Focus insights on: {focus_area}' if focus_area else ''}""",
            agent=self.reasoning_agent,
            expected_output="Strategic insights and novel hypotheses",
            context=[kg_task, gap_task]
        )

        # Create and execute crew
        crew = self.create_crew(
            agents=[self.kg_builder, self.gap_analyst, self.reasoning_agent],
            tasks=[kg_task, gap_task, reasoning_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={"papers": str(papers)})

        return {
            "papers_analyzed": len(papers),
            "focus_area": focus_area,
            "knowledge_graph_stats": self.kg_agent.get_statistics(),
            "analysis_report": result
        }

    def deep_dive_analysis(
        self,
        topic: str,
        specific_questions: List[str]
    ) -> Dict[str, Any]:
        """
        Deep dive analysis on specific research questions.

        Args:
            topic: Research topic
            specific_questions: Specific questions to answer

        Returns:
            Detailed answers with evidence
        """
        analysis_task = self.create_task(
            description=f"""Deep dive analysis on: {topic}

            Answer these specific questions:
            {chr(10).join(f'{i+1}. {q}' for i, q in enumerate(specific_questions))}

            For each question:
            1. Search knowledge graph for relevant evidence
            2. Reason about relationships and patterns
            3. Provide clear, evidence-based answers
            4. Note confidence level and supporting papers
            5. Identify what's unknown or uncertain

            Create a structured report with detailed answers.""",
            agent=self.reasoning_agent,
            expected_output="Detailed answers to all questions with evidence"
        )

        crew = self.create_crew(
            agents=[self.reasoning_agent],
            tasks=[analysis_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={"topic": topic, "questions": specific_questions})

        return {
            "topic": topic,
            "questions": specific_questions,
            "answers": result
        }
