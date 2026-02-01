"""Research crew for paper discovery and analysis."""

from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from backend.crews.base import BaseScholarGenieCrew
from backend.agents.paper_finder import PaperFinderAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.extractor import ExtractorAgent


class ResearchCrew(BaseScholarGenieCrew):
    """
    Crew for comprehensive research paper discovery and analysis.

    Coordinates multiple agents to:
    1. Find relevant papers
    2. Summarize findings
    3. Extract structured data
    4. Synthesize insights
    """

    def __init__(self, **kwargs):
        """Initialize research crew."""
        super().__init__(**kwargs)

        # Initialize ScholarGenie agents
        self.paper_finder = PaperFinderAgent()
        self.summarizer = SummarizerAgent()
        self.extractor = ExtractorAgent()

        # Create tools from ScholarGenie agents
        self.tools = self._create_tools()

        # Create CrewAI agents
        self.search_agent = self._create_search_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.synthesis_agent = self._create_synthesis_agent()

    def _create_tools(self) -> List:
        """Create tools from ScholarGenie agents."""

        @tool("search_papers")
        def search_papers(query: str, max_results: int = 10) -> str:
            """
            Search for research papers on a given topic.

            Args:
                query: Research query
                max_results: Maximum number of results

            Returns:
                List of papers with titles, authors, abstracts
            """
            results = self.paper_finder.search_papers(query, max_results=max_results)
            return str(results)

        @tool("summarize_paper")
        def summarize_paper(paper_id: str, abstract: str) -> str:
            """
            Generate summary of a research paper.

            Args:
                paper_id: Paper identifier
                abstract: Paper abstract

            Returns:
                Multi-level summary
            """
            summary = self.summarizer.summarize_abstract(paper_id, abstract, "")
            return str(summary)

        @tool("extract_data")
        def extract_data(paper_id: str, text: str) -> str:
            """
            Extract structured data from paper text.

            Args:
                paper_id: Paper identifier
                text: Paper full text

            Returns:
                Extracted methods, findings, limitations
            """
            data = self.extractor.extract_structured_data(paper_id, text)
            return str(data)

        return [search_papers, summarize_paper, extract_data]

    def _create_search_agent(self) -> Agent:
        """Create paper search specialist agent."""
        return self.create_agent(
            role="Research Paper Search Specialist",
            goal="Find the most relevant and high-quality research papers for given topics",
            backstory="""You are an expert research librarian with deep knowledge of
            academic databases and search strategies. You excel at finding relevant papers
            across multiple sources including Semantic Scholar, arXiv, and PubMed. You
            understand research methodologies and can identify foundational papers, recent
            advances, and emerging trends.""",
            tools=self.tools[:1],  # Only search tool
            allow_delegation=False
        )

    def _create_analysis_agent(self) -> Agent:
        """Create paper analysis specialist agent."""
        return self.create_agent(
            role="Research Paper Analyst",
            goal="Analyze papers deeply to extract key insights, methods, and findings",
            backstory="""You are a research analyst with expertise in multiple scientific
            domains. You can quickly read and understand complex research papers, identify
            novel contributions, assess methodological rigor, and extract actionable insights.
            You have a keen eye for experimental design and statistical analysis.""",
            tools=self.tools[1:],  # Summary and extraction tools
            allow_delegation=False
        )

    def _create_synthesis_agent(self) -> Agent:
        """Create synthesis specialist agent."""
        return self.create_agent(
            role="Research Synthesis Expert",
            goal="Synthesize findings across multiple papers into coherent insights",
            backstory="""You are a senior researcher who excels at connecting ideas across
            papers, identifying patterns, detecting gaps, and synthesizing knowledge. You
            can identify contradictions, convergence of evidence, and emerging themes. Your
            summaries are clear, concise, and actionable.""",
            tools=[],  # No tools needed, uses outputs from other agents
            allow_delegation=True
        )

    def research_topic(
        self,
        topic: str,
        max_papers: int = 10,
        depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Execute comprehensive research on a topic.

        Args:
            topic: Research topic or question
            max_papers: Maximum papers to analyze
            depth: Analysis depth (quick, standard, comprehensive)

        Returns:
            Research report with papers, summaries, and synthesis
        """
        # Create tasks
        search_task = self.create_task(
            description=f"""Search for the top {max_papers} most relevant research papers on: {topic}

            Focus on:
            - Recent papers (last 3 years preferred)
            - High-impact publications
            - Diverse perspectives
            - Foundational works if applicable

            Return a list of papers with titles, authors, years, and abstracts.""",
            agent=self.search_agent,
            expected_output=f"List of {max_papers} relevant papers with complete metadata"
        )

        analysis_task = self.create_task(
            description=f"""Analyze each paper found and extract:

            1. Key findings and contributions
            2. Methods and datasets used
            3. Limitations and future work
            4. Citations and related work

            Create detailed analysis for each paper.""",
            agent=self.analysis_agent,
            expected_output="Detailed analysis of each paper with structured data",
            context=[search_task]
        )

        synthesis_task = self.create_task(
            description=f"""Synthesize findings across all analyzed papers on: {topic}

            Create a comprehensive research report including:
            1. Executive Summary
            2. Main Themes and Trends
            3. Key Findings (with evidence from papers)
            4. Methodological Approaches
            5. Gaps and Opportunities
            6. Future Research Directions
            7. Practical Implications

            The report should be clear, well-organized, and actionable.""",
            agent=self.synthesis_agent,
            expected_output="Comprehensive research synthesis report",
            context=[search_task, analysis_task]
        )

        # Create and execute crew
        crew = self.create_crew(
            agents=[self.search_agent, self.analysis_agent, self.synthesis_agent],
            tasks=[search_task, analysis_task, synthesis_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={"topic": topic, "max_papers": max_papers})

        return {
            "topic": topic,
            "papers_analyzed": max_papers,
            "report": result,
            "depth": depth
        }

    def comparative_analysis(
        self,
        topics: List[str],
        max_papers_per_topic: int = 5
    ) -> Dict[str, Any]:
        """
        Compare research across multiple topics.

        Args:
            topics: List of topics to compare
            max_papers_per_topic: Papers per topic

        Returns:
            Comparative analysis report
        """
        # Create comparison task
        comparison_task = self.create_task(
            description=f"""Compare research across these topics: {', '.join(topics)}

            For each topic, find {max_papers_per_topic} key papers and:
            1. Identify common themes
            2. Note unique aspects
            3. Compare methodologies
            4. Assess maturity of field
            5. Identify cross-pollination opportunities

            Create a comparative analysis highlighting similarities, differences, and synergies.""",
            agent=self.synthesis_agent,
            expected_output="Detailed comparative analysis across all topics"
        )

        crew = self.create_crew(
            agents=[self.search_agent, self.analysis_agent, self.synthesis_agent],
            tasks=[comparison_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={"topics": topics})

        return {
            "topics": topics,
            "comparison_report": result
        }
