"""Causal Hypothesis Tree Generator.

Generates a tree of testable hypotheses based on causal reasoning.
Each hypothesis is a potential research direction with:
- Clear causal claim
- Required experiments
- Expected outcomes
- Falsification criteria
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import uuid

from backend.agents.causal_reasoning import CausalGraphReasoner, BreakthroughPrediction
from backend.agents.knowledge_graph import KnowledgeGraphAgent, NodeType


logger = logging.getLogger(__name__)


@dataclass
class CausalHypothesis:
    """A testable causal hypothesis."""
    hypothesis_id: str
    claim: str  # "X causes Y"
    null_hypothesis: str  # "X does not cause Y"
    independent_variable: str  # X
    dependent_variable: str  # Y
    confounders: List[str]  # Potential confounding variables
    required_experiments: List[Dict]
    expected_effect_size: float
    confidence_level: float
    falsification_criteria: List[str]
    parent_hypothesis: Optional[str] = None  # For tree structure
    children: List[str] = None  # Child hypothesis IDs

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "null_hypothesis": self.null_hypothesis,
            "independent_variable": self.independent_variable,
            "dependent_variable": self.dependent_variable,
            "confounders": self.confounders,
            "required_experiments": self.required_experiments,
            "expected_effect_size": self.expected_effect_size,
            "confidence_level": self.confidence_level,
            "falsification_criteria": self.falsification_criteria,
            "parent_hypothesis": self.parent_hypothesis,
            "children": self.children
        }


class HypothesisTreeGenerator:
    """Generates hypothesis trees for research planning."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphAgent,
        causal_reasoner: CausalGraphReasoner
    ):
        """Initialize hypothesis tree generator.

        Args:
            knowledge_graph: Knowledge graph instance
            causal_reasoner: Causal reasoning engine
        """
        self.kg = knowledge_graph
        self.causal = causal_reasoner
        self.hypotheses: Dict[str, CausalHypothesis] = {}

        logger.info("HypothesisTreeGenerator initialized")

    def generate_hypothesis_tree(
        self,
        root_claim: str,
        max_depth: int = 3,
        max_breadth: int = 3
    ) -> Dict[str, CausalHypothesis]:
        """Generate a tree of related hypotheses.

        Args:
            root_claim: Root hypothesis claim (e.g., "Model X improves Task Y")
            max_depth: Maximum tree depth
            max_breadth: Maximum children per node

        Returns:
            Dictionary of hypothesis ID -> Hypothesis
        """
        logger.info(f"Generating hypothesis tree for: {root_claim}")

        # Create root hypothesis
        root = self._create_hypothesis_from_claim(root_claim)
        self.hypotheses[root.hypothesis_id] = root

        # Recursively expand
        self._expand_hypothesis(root.hypothesis_id, current_depth=0, max_depth=max_depth, max_breadth=max_breadth)

        logger.info(f"Generated {len(self.hypotheses)} hypotheses")
        return self.hypotheses

    def generate_from_breakthrough(
        self,
        breakthrough: BreakthroughPrediction,
        max_depth: int = 2
    ) -> Dict[str, CausalHypothesis]:
        """Generate hypothesis tree from a breakthrough prediction.

        Args:
            breakthrough: Breakthrough prediction
            max_depth: Maximum tree depth

        Returns:
            Dictionary of hypotheses
        """
        logger.info(f"Generating hypotheses from breakthrough: {breakthrough.title}")

        # Create root hypothesis from breakthrough
        root = self._create_hypothesis_from_breakthrough(breakthrough)
        self.hypotheses[root.hypothesis_id] = root

        # Expand with sub-hypotheses
        self._expand_hypothesis(root.hypothesis_id, current_depth=0, max_depth=max_depth, max_breadth=3)

        return self.hypotheses

    def _create_hypothesis_from_claim(self, claim: str) -> CausalHypothesis:
        """Create hypothesis from a causal claim.

        Args:
            claim: Causal claim string

        Returns:
            Causal hypothesis
        """
        # Parse claim (simplified - could use NLP)
        # Expected format: "X causes/improves/enables Y"

        hypothesis_id = str(uuid.uuid4())[:8]

        # Extract variables (simplified)
        if "improves" in claim.lower():
            parts = claim.lower().split("improves")
            iv = parts[0].strip() if len(parts) > 0 else "Unknown"
            dv = parts[1].strip() if len(parts) > 1 else "Unknown"
            verb = "improves"
        elif "causes" in claim.lower():
            parts = claim.lower().split("causes")
            iv = parts[0].strip() if len(parts) > 0 else "Unknown"
            dv = parts[1].strip() if len(parts) > 1 else "Unknown"
            verb = "causes"
        elif "enables" in claim.lower():
            parts = claim.lower().split("enables")
            iv = parts[0].strip() if len(parts) > 0 else "Unknown"
            dv = parts[1].strip() if len(parts) > 1 else "Unknown"
            verb = "enables"
        else:
            iv = "Unknown"
            dv = "Unknown"
            verb = "affects"

        return CausalHypothesis(
            hypothesis_id=hypothesis_id,
            claim=claim,
            null_hypothesis=f"{iv} does not {verb} {dv}",
            independent_variable=iv,
            dependent_variable=dv,
            confounders=self._identify_confounders(iv, dv),
            required_experiments=self._design_experiments(iv, dv),
            expected_effect_size=0.5,  # Placeholder
            confidence_level=0.7,
            falsification_criteria=self._generate_falsification_criteria(iv, dv)
        )

    def _create_hypothesis_from_breakthrough(
        self,
        breakthrough: BreakthroughPrediction
    ) -> CausalHypothesis:
        """Create hypothesis from breakthrough prediction.

        Args:
            breakthrough: Breakthrough prediction

        Returns:
            Causal hypothesis
        """
        hypothesis_id = str(uuid.uuid4())[:8]

        # Extract variables from breakthrough
        if breakthrough.required_elements and len(breakthrough.required_elements) >= 2:
            iv = breakthrough.required_elements[0]
            dv = breakthrough.required_elements[-1]

            iv_name = self.kg.nodes[iv].name if iv in self.kg.nodes else iv
            dv_name = self.kg.nodes[dv].name if dv in self.kg.nodes else dv

            claim = f"{iv_name} enables breakthroughs in {dv_name}"
            null = f"{iv_name} does not significantly impact {dv_name}"
        else:
            iv_name = "Unknown"
            dv_name = "Unknown"
            claim = breakthrough.title
            null = f"Null hypothesis for: {breakthrough.title}"

        return CausalHypothesis(
            hypothesis_id=hypothesis_id,
            claim=claim,
            null_hypothesis=null,
            independent_variable=iv_name,
            dependent_variable=dv_name,
            confounders=self._identify_confounders(iv_name, dv_name),
            required_experiments=self._design_experiments_for_breakthrough(breakthrough),
            expected_effect_size=breakthrough.expected_impact,
            confidence_level=breakthrough.confidence,
            falsification_criteria=self._generate_falsification_criteria(iv_name, dv_name)
        )

    def _expand_hypothesis(
        self,
        hypothesis_id: str,
        current_depth: int,
        max_depth: int,
        max_breadth: int
    ) -> None:
        """Recursively expand hypothesis with sub-hypotheses.

        Args:
            hypothesis_id: Hypothesis to expand
            current_depth: Current tree depth
            max_depth: Maximum depth
            max_breadth: Maximum children per node
        """
        if current_depth >= max_depth:
            return

        hypothesis = self.hypotheses[hypothesis_id]

        # Generate sub-hypotheses
        children = self._generate_sub_hypotheses(hypothesis, max_breadth)

        for child in children:
            child.parent_hypothesis = hypothesis_id
            self.hypotheses[child.hypothesis_id] = child
            hypothesis.children.append(child.hypothesis_id)

            # Recursively expand
            self._expand_hypothesis(
                child.hypothesis_id,
                current_depth + 1,
                max_depth,
                max_breadth
            )

    def _generate_sub_hypotheses(
        self,
        parent: CausalHypothesis,
        max_count: int
    ) -> List[CausalHypothesis]:
        """Generate sub-hypotheses for a parent hypothesis.

        Sub-hypotheses explore:
        1. Mechanisms (how does X cause Y?)
        2. Moderators (when does X cause Y?)
        3. Boundary conditions (where does X cause Y?)

        Args:
            parent: Parent hypothesis
            max_count: Maximum sub-hypotheses

        Returns:
            List of sub-hypotheses
        """
        sub_hypotheses = []

        # Mechanism hypotheses
        mechanism = self._generate_mechanism_hypothesis(parent)
        if mechanism:
            sub_hypotheses.append(mechanism)

        # Moderator hypotheses
        moderator = self._generate_moderator_hypothesis(parent)
        if moderator:
            sub_hypotheses.append(moderator)

        # Boundary condition hypotheses
        boundary = self._generate_boundary_hypothesis(parent)
        if boundary:
            sub_hypotheses.append(boundary)

        return sub_hypotheses[:max_count]

    def _generate_mechanism_hypothesis(
        self,
        parent: CausalHypothesis
    ) -> Optional[CausalHypothesis]:
        """Generate hypothesis about mechanism.

        Args:
            parent: Parent hypothesis

        Returns:
            Mechanism hypothesis or None
        """
        mechanism_claim = f"The mechanism by which {parent.independent_variable} affects {parent.dependent_variable} is through intermediate factor M"

        return CausalHypothesis(
            hypothesis_id=str(uuid.uuid4())[:8],
            claim=mechanism_claim,
            null_hypothesis=f"No significant mediating mechanism exists",
            independent_variable=parent.independent_variable,
            dependent_variable="Mediator M",
            confounders=parent.confounders,
            required_experiments=[
                {
                    "type": "mediation_analysis",
                    "description": "Test if removing M eliminates the effect of X on Y",
                    "variables": [parent.independent_variable, "M", parent.dependent_variable]
                }
            ],
            expected_effect_size=parent.expected_effect_size * 0.7,
            confidence_level=parent.confidence_level * 0.8,
            falsification_criteria=[
                "Effect persists when M is controlled for",
                "M is not correlated with Y"
            ]
        )

    def _generate_moderator_hypothesis(
        self,
        parent: CausalHypothesis
    ) -> Optional[CausalHypothesis]:
        """Generate hypothesis about moderating conditions.

        Args:
            parent: Parent hypothesis

        Returns:
            Moderator hypothesis or None
        """
        moderator_claim = f"The effect of {parent.independent_variable} on {parent.dependent_variable} is stronger under condition C"

        return CausalHypothesis(
            hypothesis_id=str(uuid.uuid4())[:8],
            claim=moderator_claim,
            null_hypothesis=f"Condition C does not moderate the relationship",
            independent_variable=parent.independent_variable,
            dependent_variable=parent.dependent_variable,
            confounders=parent.confounders + ["Moderator C"],
            required_experiments=[
                {
                    "type": "interaction_test",
                    "description": "Test X*C interaction on Y",
                    "variables": [parent.independent_variable, "C", parent.dependent_variable]
                }
            ],
            expected_effect_size=parent.expected_effect_size * 1.2,
            confidence_level=parent.confidence_level * 0.7,
            falsification_criteria=[
                "No significant interaction between X and C",
                "Effect size same across C conditions"
            ]
        )

    def _generate_boundary_hypothesis(
        self,
        parent: CausalHypothesis
    ) -> Optional[CausalHypothesis]:
        """Generate hypothesis about boundary conditions.

        Args:
            parent: Parent hypothesis

        Returns:
            Boundary hypothesis or None
        """
        boundary_claim = f"The relationship between {parent.independent_variable} and {parent.dependent_variable} holds only in domain D"

        return CausalHypothesis(
            hypothesis_id=str(uuid.uuid4())[:8],
            claim=boundary_claim,
            null_hypothesis=f"The relationship generalizes beyond domain D",
            independent_variable=parent.independent_variable,
            dependent_variable=parent.dependent_variable,
            confounders=parent.confounders,
            required_experiments=[
                {
                    "type": "cross_domain_validation",
                    "description": "Test relationship in multiple domains",
                    "variables": [parent.independent_variable, parent.dependent_variable, "Domain"]
                }
            ],
            expected_effect_size=parent.expected_effect_size,
            confidence_level=parent.confidence_level * 0.9,
            falsification_criteria=[
                "Effect replicates in all tested domains",
                "No domain-specific patterns observed"
            ]
        )

    def _identify_confounders(self, iv: str, dv: str) -> List[str]:
        """Identify potential confounding variables.

        Args:
            iv: Independent variable
            dv: Dependent variable

        Returns:
            List of potential confounders
        """
        confounders = []

        # Common confounders in ML research
        common_confounders = [
            "Dataset size",
            "Model capacity",
            "Training duration",
            "Hyperparameter tuning",
            "Hardware resources",
            "Implementation quality"
        ]

        # Add domain-specific confounders
        if "model" in iv.lower() or "model" in dv.lower():
            confounders.extend(["Model architecture", "Training data quality"])

        if "dataset" in iv.lower() or "dataset" in dv.lower():
            confounders.extend(["Dataset bias", "Label quality"])

        # Add common confounders
        confounders.extend(common_confounders[:3])

        return list(set(confounders))[:5]  # Max 5 confounders

    def _design_experiments(self, iv: str, dv: str) -> List[Dict]:
        """Design experiments to test hypothesis.

        Args:
            iv: Independent variable
            dv: Dependent variable

        Returns:
            List of experiment designs
        """
        experiments = []

        # Experiment 1: Controlled comparison
        experiments.append({
            "type": "controlled_comparison",
            "description": f"Compare {dv} with and without {iv}, controlling for confounders",
            "variables": [iv, dv],
            "sample_size": "Minimum 100 trials",
            "controls": ["Randomization", "Blinding if possible"]
        })

        # Experiment 2: Dose-response
        experiments.append({
            "type": "dose_response",
            "description": f"Vary levels of {iv} and measure effect on {dv}",
            "variables": [iv, dv],
            "sample_size": "Minimum 50 trials per level",
            "controls": ["Multiple levels of IV", "Consistent DV measurement"]
        })

        # Experiment 3: Ablation
        if "model" in iv.lower():
            experiments.append({
                "type": "ablation_study",
                "description": f"Remove {iv} and measure change in {dv}",
                "variables": [iv, dv],
                "sample_size": "Minimum 50 trials",
                "controls": ["Baseline comparison", "Statistical significance testing"]
            })

        return experiments[:3]  # Max 3 experiments

    def _design_experiments_for_breakthrough(
        self,
        breakthrough: BreakthroughPrediction
    ) -> List[Dict]:
        """Design experiments for breakthrough prediction.

        Args:
            breakthrough: Breakthrough prediction

        Returns:
            List of experiment designs
        """
        experiments = []

        # Pilot study
        experiments.append({
            "type": "pilot_study",
            "description": f"Initial feasibility study for: {breakthrough.title}",
            "objectives": [
                "Verify required elements are accessible",
                "Establish baseline measurements",
                "Identify technical challenges"
            ],
            "sample_size": "Small scale (10-20 trials)",
            "duration": "2-4 weeks"
        })

        # Main experiment
        experiments.append({
            "type": "main_experiment",
            "description": f"Full-scale test of: {breakthrough.title}",
            "objectives": [
                "Test causal hypothesis",
                "Measure effect size",
                "Compare to baselines"
            ],
            "sample_size": "Full scale (100+ trials)",
            "duration": "2-3 months"
        })

        # Replication study
        experiments.append({
            "type": "replication",
            "description": "Independent replication to verify findings",
            "objectives": [
                "Confirm results in different setting",
                "Test generalizability",
                "Build confidence"
            ],
            "sample_size": "Similar to main experiment",
            "duration": "1-2 months"
        })

        return experiments

    def _generate_falsification_criteria(self, iv: str, dv: str) -> List[str]:
        """Generate falsification criteria for hypothesis.

        Args:
            iv: Independent variable
            dv: Dependent variable

        Returns:
            List of falsification criteria
        """
        criteria = []

        # Statistical criteria
        criteria.append("No statistically significant difference (p > 0.05)")

        # Effect size criteria
        criteria.append("Effect size below minimum threshold (Cohen's d < 0.2)")

        # Replication criteria
        criteria.append("Failure to replicate in independent study")

        # Mechanism criteria
        criteria.append("Proposed mechanism does not hold under scrutiny")

        # Alternative explanation criteria
        criteria.append("Alternative explanation better fits the data")

        return criteria[:5]

    def get_hypothesis_tree_structure(self) -> Dict:
        """Get tree structure of hypotheses.

        Returns:
            Tree structure dictionary
        """
        tree = {
            "total_hypotheses": len(self.hypotheses),
            "root_hypotheses": [],
            "max_depth": 0,
            "hypotheses": {}
        }

        for hyp_id, hyp in self.hypotheses.items():
            tree["hypotheses"][hyp_id] = {
                "claim": hyp.claim,
                "parent": hyp.parent_hypothesis,
                "children": hyp.children,
                "depth": self._calculate_depth(hyp_id)
            }

            if hyp.parent_hypothesis is None:
                tree["root_hypotheses"].append(hyp_id)

            depth = tree["hypotheses"][hyp_id]["depth"]
            tree["max_depth"] = max(tree["max_depth"], depth)

        return tree

    def _calculate_depth(self, hypothesis_id: str) -> int:
        """Calculate depth of hypothesis in tree.

        Args:
            hypothesis_id: Hypothesis ID

        Returns:
            Depth (0 for root)
        """
        depth = 0
        current = hypothesis_id

        while current in self.hypotheses:
            parent = self.hypotheses[current].parent_hypothesis
            if parent is None:
                break
            depth += 1
            current = parent

        return depth

    def export_hypotheses(self) -> List[Dict]:
        """Export all hypotheses to dictionary format.

        Returns:
            List of hypothesis dictionaries
        """
        return [hyp.to_dict() for hyp in self.hypotheses.values()]

    def generate_research_plan(self) -> Dict:
        """Generate a research plan from hypothesis tree.

        Returns:
            Research plan dictionary
        """
        plan = {
            "overview": f"Research plan with {len(self.hypotheses)} testable hypotheses",
            "phases": [],
            "estimated_duration": "6-12 months",
            "resource_requirements": []
        }

        # Group hypotheses by depth (phases)
        by_depth = defaultdict(list)
        for hyp in self.hypotheses.values():
            depth = self._calculate_depth(hyp.hypothesis_id)
            by_depth[depth].append(hyp)

        # Create phases
        for depth in sorted(by_depth.keys()):
            phase_hyps = by_depth[depth]

            phase = {
                "phase_number": depth + 1,
                "hypotheses": [h.claim for h in phase_hyps],
                "experiments": [],
                "expected_duration": "1-2 months"
            }

            # Collect unique experiments
            all_exps = []
            for hyp in phase_hyps:
                all_exps.extend(hyp.required_experiments)

            # Deduplicate by type
            seen_types = set()
            for exp in all_exps:
                if exp["type"] not in seen_types:
                    phase["experiments"].append(exp)
                    seen_types.add(exp["type"])

            plan["phases"].append(phase)

        # Resource requirements
        plan["resource_requirements"] = [
            "Computing resources for experiments",
            "Datasets and benchmarks",
            "Research team (2-3 people)",
            "Analysis tools and software"
        ]

        return plan
