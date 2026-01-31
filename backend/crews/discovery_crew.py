"""Discovery crew for breakthrough prediction and hypothesis generation."""

from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from backend.crews.base import BaseScholarGenieCrew
from backend.agents.hypothesis_tree import HypothesisTreeGenerator
from backend.agents.causal_reasoning import CausalGraphReasoner
from backend.agents.domain_transfer import DomainTransferAgent, Domain


class DiscoveryCrew(BaseScholarGenieCrew):
    """
    Crew for scientific discovery and breakthrough prediction.

    Coordinates agents to:
    1. Generate research hypotheses
    2. Perform causal reasoning
    3. Find cross-domain analogies
    4. Predict breakthroughs
    """

    def __init__(self, kg_agent=None, **kwargs):
        """Initialize discovery crew."""
        super().__init__(**kwargs)

        # Initialize ScholarGenie agents
        from backend.agents.knowledge_graph import KnowledgeGraphAgent
        self.kg_agent = kg_agent or KnowledgeGraphAgent()
        self.hypothesis_gen = HypothesisTreeGenerator(self.kg_agent, None)
        self.causal_reasoner = CausalGraphReasoner(self.kg_agent)
        self.domain_transfer = DomainTransferAgent()

        # Create tools
        self.tools = self._create_tools()

        # Create CrewAI agents
        self.hypothesis_agent = self._create_hypothesis_agent()
        self.causal_agent = self._create_causal_agent()
        self.innovation_agent = self._create_innovation_agent()

    def _create_tools(self) -> List:
        """Create tools from ScholarGenie agents."""

        @tool("generate_hypotheses")
        def generate_hypotheses(research_question: str, max_hypotheses: int = 5) -> str:
            """
            Generate research hypotheses from a question.

            Args:
                research_question: Research question
                max_hypotheses: Maximum hypotheses to generate

            Returns:
                Hypothesis tree with evidence
            """
            tree = self.hypothesis_gen.generate_hypothesis_tree(
                research_question,
                max_depth=2,
                max_hypotheses=max_hypotheses
            )
            return str(tree)

        @tool("analyze_causality")
        def analyze_causality(concepts: str) -> str:
            """
            Analyze causal relationships between concepts.

            Args:
                concepts: JSON list of concepts

            Returns:
                Causal graph with pathways
            """
            import json
            concept_list = json.loads(concepts)
            causal_graph = self.causal_reasoner.build_causal_graph(concept_list[:10])
            return str(causal_graph)

        @tool("find_cross_domain_analogies")
        def find_analogies(problem: str, target_domain: str) -> str:
            """
            Find solutions from other scientific domains.

            Args:
                problem: Problem description
                target_domain: Target domain

            Returns:
                Cross-domain transfers and solutions
            """
            try:
                domain_enum = Domain(target_domain.lower())
                transfers = self.domain_transfer.find_analogies(
                    problem=problem,
                    target_domain=domain_enum
                )
                return str(transfers[:5])  # Top 5 transfers
            except:
                return "Invalid domain"

        return [generate_hypotheses, analyze_causality, find_analogies]

    def _create_hypothesis_agent(self) -> Agent:
        """Create hypothesis generation agent."""
        return self.create_agent(
            role="Research Hypothesis Generator",
            goal="Generate novel, testable research hypotheses",
            backstory="""You are a creative research scientist who excels at generating
            innovative hypotheses. You can think divergently to propose multiple plausible
            explanations, then convergently to refine and prioritize them. You understand
            what makes a good hypothesis: specific, testable, falsifiable, and grounded
            in existing knowledge while pushing boundaries.""",
            tools=[self.tools[0]],
            allow_delegation=False
        )

    def _create_causal_agent(self) -> Agent:
        """Create causal reasoning agent."""
        return self.create_agent(
            role="Causal Reasoning Specialist",
            goal="Identify and analyze causal relationships in research",
            backstory="""You are an expert in causal inference and scientific reasoning.
            You can distinguish correlation from causation, identify confounders, and
            map causal pathways. You understand experimental design, statistical methods,
            and logical reasoning needed to establish causality.""",
            tools=[self.tools[1]],
            allow_delegation=False
        )

    def _create_innovation_agent(self) -> Agent:
        """Create innovation and discovery agent."""
        return self.create_agent(
            role="Scientific Innovation Catalyst",
            goal="Find breakthrough ideas through cross-domain thinking",
            backstory="""You are an innovation expert who specializes in finding solutions
            from unexpected places. You excel at analogical reasoning, connecting ideas
            across disciplines, and identifying how concepts from one field can solve
            problems in another. You've studied many scientific breakthroughs and understand
            common patterns of innovation.""",
            tools=[self.tools[2]],
            allow_delegation=True
        )

    def discover_breakthroughs(
        self,
        research_area: str,
        current_challenges: List[str],
        target_domain: str = "computer_science"
    ) -> Dict[str, Any]:
        """
        Discover potential breakthrough research directions.

        Args:
            research_area: Area of research
            current_challenges: List of current challenges
            target_domain: Primary research domain

        Returns:
            Breakthrough predictions with evidence
        """
        # Create tasks
        hypothesis_task = self.create_task(
            description=f"""Generate breakthrough research hypotheses for: {research_area}

            Current challenges:
            {chr(10).join(f'- {c}' for c in current_challenges)}

            For each challenge, generate 3-5 innovative hypotheses that could lead to breakthroughs.

            Each hypothesis should:
            1. Be specific and testable
            2. Build on existing knowledge
            3. Propose a novel approach or perspective
            4. Have clear potential impact
            5. Be feasible to investigate

            Organize hypotheses by challenge and priority.""",
            agent=self.hypothesis_agent,
            expected_output="Structured set of breakthrough hypotheses for each challenge"
        )

        causal_task = self.create_task(
            description=f"""Analyze causal relationships in: {research_area}

            Examine the hypotheses and:
            1. Identify key causal pathways
            2. Find potential interventions
            3. Assess mechanistic understanding
            4. Predict cascading effects
            5. Identify critical experiments

            Build a causal model showing how different approaches could lead to breakthroughs.""",
            agent=self.causal_agent,
            expected_output="Causal analysis with intervention points and predictions",
            context=[hypothesis_task]
        )

        innovation_task = self.create_task(
            description=f"""Find innovative solutions through cross-domain thinking:

            Research area: {research_area}
            Target domain: {target_domain}

            For each challenge and hypothesis:
            1. Search for analogous problems in other domains
            2. Identify successful solutions from those domains
            3. Adapt solutions to this research area
            4. Assess novelty and feasibility
            5. Generate concrete research proposals

            Focus on truly innovative, interdisciplinary approaches that could lead to breakthroughs.

            Create a breakthrough discovery report with prioritized opportunities.""",
            agent=self.innovation_agent,
            expected_output="Breakthrough opportunities with cross-domain solutions",
            context=[hypothesis_task, causal_task]
        )

        # Create and execute crew
        crew = self.create_crew(
            agents=[self.hypothesis_agent, self.causal_agent, self.innovation_agent],
            tasks=[hypothesis_task, causal_task, innovation_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={
            "research_area": research_area,
            "challenges": current_challenges,
            "target_domain": target_domain
        })

        return {
            "research_area": research_area,
            "challenges_analyzed": len(current_challenges),
            "breakthrough_report": result
        }

    def validate_hypothesis(
        self,
        hypothesis: str,
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate a research hypothesis against evidence.

        Args:
            hypothesis: Hypothesis to validate
            evidence: List of supporting/contradicting evidence

        Returns:
            Validation report with recommendations
        """
        validation_task = self.create_task(
            description=f"""Validate this research hypothesis:

            Hypothesis: {hypothesis}

            Evidence provided:
            {chr(10).join(f'- {e.get("description", str(e))}' for e in evidence)}

            Analyze:
            1. Supporting evidence strength
            2. Contradicting evidence
            3. Alternative explanations
            4. Causal mechanisms
            5. Testability and falsifiability
            6. Required experiments
            7. Confidence assessment

            Provide a comprehensive validation report with recommendations.""",
            agent=self.causal_agent,
            expected_output="Detailed hypothesis validation with confidence score"
        )

        crew = self.create_crew(
            agents=[self.causal_agent],
            tasks=[validation_task],
            process=Process.sequential
        )

        result = self.kickoff(crew, inputs={"hypothesis": hypothesis, "evidence": evidence})

        return {
            "hypothesis": hypothesis,
            "evidence_count": len(evidence),
            "validation_report": result
        }
