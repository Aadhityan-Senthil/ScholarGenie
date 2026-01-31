"""
Cross-Domain Knowledge Transfer Agent

Finds solutions from other scientific domains using analogical reasoning:
1. Maps concepts across disciplines
2. Identifies analogous problems in different fields
3. Generates cross-domain solution proposals
4. Tracks successful knowledge transfers

Usage:
    agent = DomainTransferAgent()
    transfers = agent.find_analogies(
        problem="Optimize neural architecture search",
        source_domains=["biology", "physics"]
    )
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re


class Domain(Enum):
    """Scientific domains"""
    COMPUTER_SCIENCE = "computer_science"
    BIOLOGY = "biology"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    NEUROSCIENCE = "neuroscience"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"


@dataclass
class ConceptMapping:
    """Maps concepts between domains"""
    source_domain: Domain
    target_domain: Domain
    source_concept: str
    target_concept: str
    similarity_score: float  # 0-1
    mapping_type: str  # "direct", "analogical", "metaphorical"


@dataclass
class CrossDomainTransfer:
    """A knowledge transfer from one domain to another"""
    transfer_id: str
    source_domain: Domain
    target_domain: Domain

    # Problem mapping
    source_problem: str
    target_problem: str
    problem_analogy: str

    # Solution mapping
    source_solution: str
    adapted_solution: str

    # Concept mappings
    concept_mappings: List[ConceptMapping] = field(default_factory=list)

    # Feasibility
    novelty_score: float = 0.0  # 0-1
    feasibility_score: float = 0.0  # 0-1
    impact_score: float = 0.0  # 0-1

    # Research proposal
    research_title: str = ""
    hypothesis: str = ""
    methodology: str = ""
    expected_outcomes: List[str] = field(default_factory=list)

    # Historical precedents
    similar_transfers: List[str] = field(default_factory=list)


class DomainTransferAgent:
    """
    Discovers cross-domain knowledge transfer opportunities
    """

    def __init__(self):
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.concept_mappings = self._initialize_concept_mappings()
        self.successful_transfers = self._initialize_historical_transfers()

    def _initialize_domain_knowledge(self) -> Dict[Domain, Dict]:
        """Initialize knowledge base for each domain"""
        return {
            Domain.COMPUTER_SCIENCE: {
                "core_concepts": ["algorithm", "optimization", "search", "learning", "network", "architecture"],
                "problems": ["efficiency", "scalability", "accuracy", "generalization", "robustness"],
                "methods": ["gradient descent", "backpropagation", "dynamic programming", "greedy search"],
                "entities": ["neural network", "tree", "graph", "array", "function"]
            },
            Domain.BIOLOGY: {
                "core_concepts": ["evolution", "selection", "adaptation", "mutation", "fitness", "population"],
                "problems": ["survival", "reproduction", "resource allocation", "adaptation"],
                "methods": ["natural selection", "genetic variation", "inheritance", "mutation"],
                "entities": ["organism", "gene", "protein", "cell", "population", "species"]
            },
            Domain.PHYSICS: {
                "core_concepts": ["energy", "force", "momentum", "equilibrium", "dynamics", "optimization"],
                "problems": ["minimization", "stability", "conservation", "equilibrium"],
                "methods": ["energy minimization", "least action", "statistical mechanics", "thermodynamics"],
                "entities": ["particle", "system", "state", "field", "wave"]
            },
            Domain.NEUROSCIENCE: {
                "core_concepts": ["learning", "plasticity", "memory", "attention", "perception"],
                "problems": ["pattern recognition", "decision making", "learning", "adaptation"],
                "methods": ["hebbian learning", "reinforcement", "attention mechanisms"],
                "entities": ["neuron", "synapse", "network", "circuit"]
            },
            Domain.ECONOMICS: {
                "core_concepts": ["optimization", "equilibrium", "utility", "allocation", "game theory"],
                "problems": ["resource allocation", "equilibrium finding", "strategy", "pricing"],
                "methods": ["game theory", "mechanism design", "optimization", "equilibrium analysis"],
                "entities": ["agent", "market", "utility", "strategy"]
            }
        }

    def _initialize_concept_mappings(self) -> List[ConceptMapping]:
        """Initialize known concept mappings between domains"""
        return [
            # CS ↔ Biology
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.BIOLOGY, "neural network", "brain", 0.8, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.BIOLOGY, "learning algorithm", "evolution", 0.7, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.BIOLOGY, "optimization", "natural selection", 0.75, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.BIOLOGY, "mutation (genetic algorithm)", "genetic mutation", 0.9, "direct"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.BIOLOGY, "population", "generation", 0.85, "direct"),

            # CS ↔ Physics
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.PHYSICS, "optimization", "energy minimization", 0.8, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.PHYSICS, "gradient descent", "potential descent", 0.75, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.PHYSICS, "local minimum", "energy well", 0.8, "analogical"),
            ConceptMapping(Domain.COMPUTER_SCIENCE, Domain.PHYSICS, "simulated annealing", "thermodynamic annealing", 0.95, "direct"),

            # Biology ↔ Engineering
            ConceptMapping(Domain.BIOLOGY, Domain.ENGINEERING, "evolution", "optimization", 0.7, "analogical"),
            ConceptMapping(Domain.BIOLOGY, Domain.ENGINEERING, "adaptation", "control system", 0.65, "analogical"),

            # Physics ↔ ML
            ConceptMapping(Domain.PHYSICS, Domain.COMPUTER_SCIENCE, "Boltzmann distribution", "softmax", 0.85, "analogical"),
            ConceptMapping(Domain.PHYSICS, Domain.COMPUTER_SCIENCE, "statistical mechanics", "probabilistic models", 0.75, "analogical"),
        ]

    def _initialize_historical_transfers(self) -> Dict[str, Dict]:
        """Initialize database of successful historical transfers"""
        return {
            "genetic_algorithms": {
                "source_domain": Domain.BIOLOGY,
                "target_domain": Domain.COMPUTER_SCIENCE,
                "source_concept": "natural selection and evolution",
                "application": "optimization algorithms",
                "success": "widely used for optimization problems",
                "key_papers": ["Holland 1975"]
            },
            "simulated_annealing": {
                "source_domain": Domain.PHYSICS,
                "target_domain": Domain.COMPUTER_SCIENCE,
                "source_concept": "thermodynamic annealing",
                "application": "global optimization",
                "success": "effective for combinatorial optimization",
                "key_papers": ["Kirkpatrick 1983"]
            },
            "ant_colony_optimization": {
                "source_domain": Domain.BIOLOGY,
                "target_domain": Domain.COMPUTER_SCIENCE,
                "source_concept": "ant foraging behavior",
                "application": "routing and scheduling",
                "success": "used in logistics and networking",
                "key_papers": ["Dorigo 1992"]
            },
            "particle_swarm": {
                "source_domain": Domain.BIOLOGY,
                "target_domain": Domain.COMPUTER_SCIENCE,
                "source_concept": "bird flocking / fish schooling",
                "application": "optimization",
                "success": "popular metaheuristic algorithm",
                "key_papers": ["Kennedy 1995"]
            },
            "neural_networks": {
                "source_domain": Domain.NEUROSCIENCE,
                "target_domain": Domain.COMPUTER_SCIENCE,
                "source_concept": "biological neurons and synapses",
                "application": "machine learning",
                "success": "foundation of deep learning",
                "key_papers": ["McCulloch 1943", "Rosenblatt 1958"]
            }
        }

    def find_analogies(
        self,
        problem: str,
        target_domain: Domain,
        source_domains: Optional[List[Domain]] = None,
        min_similarity: float = 0.5
    ) -> List[CrossDomainTransfer]:
        """
        Find analogous solutions from other domains

        Args:
            problem: Description of the problem in target domain
            target_domain: Domain where problem exists
            source_domains: Domains to search for analogies (None = all)
            min_similarity: Minimum similarity threshold

        Returns:
            List of potential knowledge transfers
        """

        if source_domains is None:
            source_domains = [d for d in Domain if d != target_domain]

        transfers = []

        # Extract problem characteristics
        problem_keywords = self._extract_keywords(problem)
        problem_type = self._classify_problem_type(problem, problem_keywords)

        for source_domain in source_domains:
            # Find similar problems in source domain
            similar_problems = self._find_similar_problems(
                problem_type, problem_keywords, source_domain
            )

            for source_problem, similarity in similar_problems:
                if similarity < min_similarity:
                    continue

                # Get solution from source domain
                source_solution = self._get_domain_solution(source_domain, source_problem)

                # Adapt solution to target domain
                adapted_solution, concept_maps = self._adapt_solution(
                    source_solution, source_domain, target_domain
                )

                # Generate research proposal
                transfer = self._create_transfer(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_problem=source_problem,
                    target_problem=problem,
                    source_solution=source_solution,
                    adapted_solution=adapted_solution,
                    concept_mappings=concept_maps
                )

                transfers.append(transfer)

        # Sort by feasibility * impact
        transfers.sort(key=lambda t: t.feasibility_score * t.impact_score, reverse=True)

        return transfers[:10]  # Top 10

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction (in production, use NLP)
        keywords = []
        text_lower = text.lower()

        all_concepts = set()
        for domain_info in self.domain_knowledge.values():
            all_concepts.update(domain_info["core_concepts"])
            all_concepts.update(domain_info["problems"])

        for concept in all_concepts:
            if concept in text_lower:
                keywords.append(concept)

        return keywords

    def _classify_problem_type(self, problem: str, keywords: List[str]) -> str:
        """Classify the type of problem"""
        problem_lower = problem.lower()

        if any(word in problem_lower for word in ["optimize", "optimization", "minimize", "maximize"]):
            return "optimization"
        elif any(word in problem_lower for word in ["search", "find", "discover"]):
            return "search"
        elif any(word in problem_lower for word in ["learn", "learning", "adapt"]):
            return "learning"
        elif any(word in problem_lower for word in ["allocate", "allocation", "distribute"]):
            return "resource_allocation"
        elif any(word in problem_lower for word in ["stable", "equilibrium", "balance"]):
            return "equilibrium"
        else:
            return "general"

    def _find_similar_problems(
        self,
        problem_type: str,
        keywords: List[str],
        source_domain: Domain
    ) -> List[Tuple[str, float]]:
        """Find similar problems in source domain"""

        domain_info = self.domain_knowledge.get(source_domain, {})
        similar_problems = []

        # Map problem type to source domain
        if problem_type == "optimization":
            if source_domain == Domain.BIOLOGY:
                similar_problems.append(("evolution for survival and reproduction", 0.8))
                similar_problems.append(("resource allocation in populations", 0.75))
            elif source_domain == Domain.PHYSICS:
                similar_problems.append(("energy minimization in physical systems", 0.85))
                similar_problems.append(("finding equilibrium states", 0.75))
            elif source_domain == Domain.ECONOMICS:
                similar_problems.append(("utility maximization", 0.8))
                similar_problems.append(("market equilibrium", 0.7))

        elif problem_type == "search":
            if source_domain == Domain.BIOLOGY:
                similar_problems.append(("foraging for resources", 0.75))
                similar_problems.append(("exploration-exploitation in animal behavior", 0.8))
            elif source_domain == Domain.PHYSICS:
                similar_problems.append(("particle path finding", 0.7))

        elif problem_type == "learning":
            if source_domain == Domain.BIOLOGY:
                similar_problems.append(("adaptation through evolution", 0.8))
                similar_problems.append(("behavioral learning", 0.75))
            elif source_domain == Domain.NEUROSCIENCE:
                similar_problems.append(("synaptic plasticity and learning", 0.9))
                similar_problems.append(("memory formation", 0.75))

        return similar_problems

    def _get_domain_solution(self, domain: Domain, problem: str) -> str:
        """Get how the source domain solves the problem"""

        domain_info = self.domain_knowledge.get(domain, {})

        if domain == Domain.BIOLOGY and "evolution" in problem:
            return "Natural selection through variation, selection, and inheritance"
        elif domain == Domain.BIOLOGY and "foraging" in problem:
            return "Swarm intelligence and pheromone-based communication"
        elif domain == Domain.PHYSICS and "energy" in problem:
            return "Gradient descent on energy landscape with thermal fluctuations"
        elif domain == Domain.PHYSICS and "equilibrium" in problem:
            return "Statistical mechanics and Boltzmann distribution"
        elif domain == Domain.NEUROSCIENCE and "learning" in problem:
            return "Hebbian learning and synaptic weight adjustment"
        else:
            return f"Domain-specific approach using {', '.join(domain_info.get('methods', [])[:2])}"

    def _adapt_solution(
        self,
        source_solution: str,
        source_domain: Domain,
        target_domain: Domain
    ) -> Tuple[str, List[ConceptMapping]]:
        """Adapt solution from source to target domain"""

        # Find relevant concept mappings
        relevant_mappings = [
            m for m in self.concept_mappings
            if (m.source_domain == source_domain and m.target_domain == target_domain)
        ]

        adapted_solution = source_solution

        # Apply concept mappings
        for mapping in relevant_mappings:
            adapted_solution = adapted_solution.replace(
                mapping.source_concept,
                f"{mapping.target_concept} (adapted from {mapping.source_concept})"
            )

        return adapted_solution, relevant_mappings

    def _create_transfer(
        self,
        source_domain: Domain,
        target_domain: Domain,
        source_problem: str,
        target_problem: str,
        source_solution: str,
        adapted_solution: str,
        concept_mappings: List[ConceptMapping]
    ) -> CrossDomainTransfer:
        """Create complete knowledge transfer"""

        # Generate research proposal
        title = f"{source_domain.value.replace('_', ' ').title()}-Inspired Approach to {target_problem[:50]}"

        hypothesis = f"Adapting {source_domain.value} principles of {source_solution[:50]} can improve {target_problem[:50]}"

        methodology = f"""
1. Formalize the mapping between {source_domain.value} concepts and {target_domain.value} entities
2. Implement adapted algorithm based on {source_solution[:50]}
3. Validate on benchmark problems
4. Compare with existing {target_domain.value} approaches
"""

        expected_outcomes = [
            f"Novel {target_domain.value} algorithm inspired by {source_domain.value}",
            "Improved performance on specific problem classes",
            "Theoretical understanding of cross-domain mappings",
            "Open-source implementation"
        ]

        # Score novelty, feasibility, impact
        novelty_score = 0.7 + len(concept_mappings) * 0.05  # More mappings = more novel
        feasibility_score = 0.8 if concept_mappings else 0.5  # Higher if we have mappings
        impact_score = 0.7  # Default moderate impact

        # Find similar historical transfers
        similar = [
            name for name, transfer in self.successful_transfers.items()
            if transfer["source_domain"] == source_domain and transfer["target_domain"] == target_domain
        ]

        return CrossDomainTransfer(
            transfer_id=f"{source_domain.value}_{target_domain.value}_{hash(target_problem) % 10000}",
            source_domain=source_domain,
            target_domain=target_domain,
            source_problem=source_problem,
            target_problem=target_problem,
            problem_analogy=f"{source_problem} is analogous to {target_problem}",
            source_solution=source_solution,
            adapted_solution=adapted_solution,
            concept_mappings=concept_mappings,
            novelty_score=min(1.0, novelty_score),
            feasibility_score=feasibility_score,
            impact_score=impact_score,
            research_title=title,
            hypothesis=hypothesis,
            methodology=methodology,
            expected_outcomes=expected_outcomes,
            similar_transfers=similar
        )

    def get_historical_transfers(self) -> List[Dict]:
        """Get list of successful historical transfers"""
        return [
            {
                "name": name,
                **transfer
            }
            for name, transfer in self.successful_transfers.items()
        ]


if __name__ == "__main__":
    # Example usage
    print("Cross-Domain Knowledge Transfer - Example")

    agent = DomainTransferAgent()

    # Find analogies for a CS problem from biology
    transfers = agent.find_analogies(
        problem="Optimize neural architecture search for deep learning models",
        target_domain=Domain.COMPUTER_SCIENCE,
        source_domains=[Domain.BIOLOGY, Domain.PHYSICS]
    )

    print(f"\nFound {len(transfers)} potential knowledge transfers:")

    for i, transfer in enumerate(transfers, 1):
        print(f"\n{i}. {transfer.research_title}")
        print(f"   Source: {transfer.source_domain.value} → Target: {transfer.target_domain.value}")
        print(f"   Analogy: {transfer.problem_analogy}")
        print(f"   Scores: Novelty={transfer.novelty_score:.2f}, Feasibility={transfer.feasibility_score:.2f}, Impact={transfer.impact_score:.2f}")
        print(f"   Concept Mappings: {len(transfer.concept_mappings)}")
        if transfer.similar_transfers:
            print(f"   Historical Precedents: {', '.join(transfer.similar_transfers)}")

    # Show historical transfers
    print(f"\n\nHistorical Successful Transfers:")
    for transfer in agent.get_historical_transfers()[:3]:
        print(f"- {transfer['name']}: {transfer['source_domain'].value} → {transfer['target_domain'].value}")
        print(f"  {transfer['source_concept']} applied to {transfer['application']}")
