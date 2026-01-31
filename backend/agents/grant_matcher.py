"""
Research Funding & Grant Matcher

This agent matches research gaps to active funding opportunities:
1. Maintains database of grants (NSF, NIH, EU, DARPA, etc.)
2. Scores grant-research fit using keyword matching
3. Generates proposal outlines matching grant requirements
4. Tracks deadlines and eligibility
5. Provides budget templates

Usage:
    matcher = GrantMatcher()
    matches = matcher.find_matching_grants(
        research_gap="Apply transformers to protein folding",
        keywords=["AI", "biology", "transformers"]
    )
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


class GrantAgency(Enum):
    """Funding agencies"""
    NSF = "nsf"  # National Science Foundation
    NIH = "nih"  # National Institutes of Health
    DARPA = "darpa"  # Defense Advanced Research Projects Agency
    DOE = "doe"  # Department of Energy
    EU_HORIZON = "eu_horizon"  # European Research Council
    GOOGLE_RESEARCH = "google"
    MICROSOFT_RESEARCH = "microsoft"
    META_RESEARCH = "meta"


class GrantType(Enum):
    """Types of grants"""
    RESEARCH = "research"
    TRAINING = "training"
    EQUIPMENT = "equipment"
    CONFERENCE = "conference"
    COLLABORATIVE = "collaborative"
    EARLY_CAREER = "early_career"


@dataclass
class Grant:
    """Funding opportunity"""
    grant_id: str
    title: str
    agency: GrantAgency
    grant_type: GrantType

    # Funding details
    funding_amount_min: int  # USD
    funding_amount_max: int  # USD
    duration_years: float

    # Deadlines
    deadline: datetime
    award_announcement: Optional[datetime] = None

    # Focus areas
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    disciplines: List[str] = field(default_factory=list)

    # Requirements
    eligibility: List[str] = field(default_factory=list)
    required_sections: List[str] = field(default_factory=list)
    page_limit: Optional[int] = None

    # URLs
    info_url: Optional[str] = None
    application_url: Optional[str] = None

    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GrantMatch:
    """Match between research and grant"""
    grant: Grant
    match_score: float  # 0-1

    # Matching details
    matched_keywords: List[str] = field(default_factory=list)
    matched_topics: List[str] = field(default_factory=list)

    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Urgency
    days_until_deadline: int = 0
    urgency: str = "low"  # low, medium, high, urgent


@dataclass
class ProposalOutline:
    """Generated proposal outline"""
    grant_id: str
    grant_title: str
    research_question: str

    # Sections (following grant requirements)
    project_summary: str
    project_description: Dict[str, str]  # section_name -> content
    timeline: List[Dict]  # milestones
    budget: Dict[str, float]
    personnel: List[Dict]

    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0


class GrantMatcher:
    """
    Matches research to funding opportunities
    """

    def __init__(self):
        self.grants: Dict[str, Grant] = {}
        self._initialize_grant_database()

    def _initialize_grant_database(self):
        """Initialize with sample grants (in production, load from database/API)"""

        # NSF AI Institutes
        self.grants["nsf_ai_2024"] = Grant(
            grant_id="nsf_ai_2024",
            title="NSF National Artificial Intelligence Research Institutes",
            agency=GrantAgency.NSF,
            grant_type=GrantType.RESEARCH,
            funding_amount_min=5_000_000,
            funding_amount_max=20_000_000,
            duration_years=5,
            deadline=datetime(2024, 8, 15),
            keywords=["artificial intelligence", "machine learning", "AI", "deep learning", "transformers"],
            topics=["AI research", "foundational AI", "AI applications"],
            disciplines=["computer science", "engineering", "cognitive science"],
            eligibility=["US universities", "research institutions"],
            required_sections=["project summary", "project description", "broader impacts", "budget", "timeline"],
            page_limit=15,
            info_url="https://nsf.gov/ai-institutes",
            is_active=True
        )

        # NIH Bioinformatics
        self.grants["nih_bioinfo_2024"] = Grant(
            grant_id="nih_bioinfo_2024",
            title="NIH Bioinformatics and Computational Biology Research",
            agency=GrantAgency.NIH,
            grant_type=GrantType.RESEARCH,
            funding_amount_min=500_000,
            funding_amount_max=2_500_000,
            duration_years=3,
            deadline=datetime(2024, 6, 30),
            keywords=["bioinformatics", "computational biology", "genomics", "AI", "machine learning"],
            topics=["protein structure prediction", "drug discovery", "genomics analysis"],
            disciplines=["biology", "computer science", "bioinformatics"],
            eligibility=["US institutions", "non-profit research organizations"],
            required_sections=["specific aims", "research strategy", "budget justification"],
            page_limit=12,
            info_url="https://nih.gov/grants/bioinformatics",
            is_active=True
        )

        # DARPA AI Next
        self.grants["darpa_ai_2024"] = Grant(
            grant_id="darpa_ai_2024",
            title="DARPA AI Next Campaign",
            agency=GrantAgency.DARPA,
            grant_type=GrantType.RESEARCH,
            funding_amount_min=1_000_000,
            funding_amount_max=10_000_000,
            duration_years=4,
            deadline=datetime(2024, 12, 1),
            keywords=["AI", "machine learning", "autonomous systems", "robust AI", "explainable AI"],
            topics=["next-generation AI", "AI robustness", "AI explainability"],
            disciplines=["computer science", "robotics", "cognitive science"],
            eligibility=["US entities", "allies"],
            required_sections=["technical approach", "team qualifications", "management plan", "budget"],
            page_limit=20,
            info_url="https://darpa.mil/ai-next",
            is_active=True
        )

        # EU Horizon Research
        self.grants["eu_horizon_2024"] = Grant(
            grant_id="eu_horizon_2024",
            title="Horizon Europe - AI and Robotics",
            agency=GrantAgency.EU_HORIZON,
            grant_type=GrantType.COLLABORATIVE,
            funding_amount_min=2_000_000,
            funding_amount_max=8_000_000,
            duration_years=4,
            deadline=datetime(2024, 9, 20),
            keywords=["AI", "robotics", "machine learning", "autonomous systems"],
            topics=["AI for society", "human-centric AI", "trustworthy AI"],
            disciplines=["computer science", "robotics", "social sciences"],
            eligibility=["EU member states", "associated countries"],
            required_sections=["excellence", "impact", "implementation", "budget"],
            page_limit=50,
            info_url="https://ec.europa.eu/horizon",
            is_active=True
        )

        # Google Research Award
        self.grants["google_ai_2024"] = Grant(
            grant_id="google_ai_2024",
            title="Google Research Scholar Program - AI",
            agency=GrantAgency.GOOGLE_RESEARCH,
            grant_type=GrantType.RESEARCH,
            funding_amount_min=60_000,
            funding_amount_max=150_000,
            duration_years=1,
            deadline=datetime(2024, 4, 1),
            keywords=["AI", "machine learning", "NLP", "computer vision", "deep learning"],
            topics=["AI research", "ML applications"],
            disciplines=["computer science"],
            eligibility=["faculty at accredited universities"],
            required_sections=["research proposal", "budget", "bio"],
            page_limit=3,
            info_url="https://research.google/outreach/research-scholar-program/",
            is_active=True
        )

    def find_matching_grants(
        self,
        research_gap: str,
        keywords: List[str],
        min_match_score: float = 0.3,
        max_results: int = 10
    ) -> List[GrantMatch]:
        """
        Find grants matching research gap

        Args:
            research_gap: Description of research gap
            keywords: Research keywords
            min_match_score: Minimum match score threshold
            max_results: Maximum results to return

        Returns:
            List of GrantMatch objects
        """
        matches = []

        for grant in self.grants.values():
            if not grant.is_active:
                continue

            # Skip if deadline passed
            if grant.deadline < datetime.now():
                continue

            # Calculate match score
            match_score, matched_kw, matched_topics = self._calculate_match_score(
                research_gap, keywords, grant
            )

            if match_score >= min_match_score:
                # Calculate urgency
                days_left = (grant.deadline - datetime.now()).days
                if days_left < 30:
                    urgency = "urgent"
                elif days_left < 60:
                    urgency = "high"
                elif days_left < 90:
                    urgency = "medium"
                else:
                    urgency = "low"

                # Generate strengths/weaknesses
                strengths, weaknesses, recommendations = self._analyze_fit(
                    research_gap, keywords, grant, match_score
                )

                match = GrantMatch(
                    grant=grant,
                    match_score=match_score,
                    matched_keywords=matched_kw,
                    matched_topics=matched_topics,
                    strengths=strengths,
                    weaknesses=weaknesses,
                    recommendations=recommendations,
                    days_until_deadline=days_left,
                    urgency=urgency
                )
                matches.append(match)

        # Sort by match score
        matches.sort(key=lambda x: x.match_score, reverse=True)

        return matches[:max_results]

    def _calculate_match_score(
        self,
        research_gap: str,
        keywords: List[str],
        grant: Grant
    ) -> tuple:
        """Calculate how well research matches grant"""

        research_text = (research_gap + " " + " ".join(keywords)).lower()

        # Keyword matching (40%)
        matched_keywords = []
        for gkw in grant.keywords:
            if gkw.lower() in research_text:
                matched_keywords.append(gkw)

        keyword_score = len(matched_keywords) / max(1, len(grant.keywords)) if grant.keywords else 0

        # Topic matching (30%)
        matched_topics = []
        for topic in grant.topics:
            if topic.lower() in research_text:
                matched_topics.append(topic)

        topic_score = len(matched_topics) / max(1, len(grant.topics)) if grant.topics else 0

        # Discipline matching (20%)
        discipline_score = 0
        for disc in grant.disciplines:
            if disc.lower() in research_text:
                discipline_score += 1
        discipline_score = discipline_score / max(1, len(grant.disciplines)) if grant.disciplines else 0

        # Funding amount feasibility (10%)
        # Assume research needs $1-5M, prefer grants in that range
        funding_score = 0.5  # Default moderate fit
        if grant.funding_amount_min <= 5_000_000:
            funding_score = 0.8

        # Combined score
        total_score = (
            keyword_score * 0.4 +
            topic_score * 0.3 +
            discipline_score * 0.2 +
            funding_score * 0.1
        )

        return total_score, matched_keywords, matched_topics

    def _analyze_fit(
        self,
        research_gap: str,
        keywords: List[str],
        grant: Grant,
        match_score: float
    ) -> tuple:
        """Analyze strengths, weaknesses, recommendations"""

        strengths = []
        weaknesses = []
        recommendations = []

        # Strengths
        if match_score > 0.7:
            strengths.append("Excellent keyword alignment with grant focus areas")
        if grant.funding_amount_max >= 5_000_000:
            strengths.append("Substantial funding available for comprehensive research")
        if grant.duration_years >= 3:
            strengths.append("Multi-year funding enables longitudinal studies")

        # Weaknesses
        days_left = (grant.deadline - datetime.now()).days
        if days_left < 45:
            weaknesses.append(f"Limited time to deadline ({days_left} days)")
        if grant.page_limit and grant.page_limit < 10:
            weaknesses.append("Strict page limit requires concise writing")
        if "collaborative" in grant.grant_type.value:
            weaknesses.append("Requires multi-institutional collaboration")

        # Recommendations
        if match_score < 0.7:
            recommendations.append("Emphasize alignment with grant focus areas in proposal")
        if "broader impacts" in [s.lower() for s in grant.required_sections]:
            recommendations.append("Develop strong broader impacts section addressing societal benefits")
        if grant.agency == GrantAgency.DARPA:
            recommendations.append("Focus on practical applications and national security relevance")
        if grant.agency == GrantAgency.NSF:
            recommendations.append("Highlight intellectual merit and broader impacts")
        recommendations.append(f"Begin drafting proposal at least {max(30, days_left // 2)} days before deadline")

        return strengths, weaknesses, recommendations

    def generate_proposal_outline(
        self,
        grant_id: str,
        research_question: str,
        research_gap: str,
        methodology: Optional[str] = None
    ) -> ProposalOutline:
        """
        Generate proposal outline tailored to grant requirements

        Args:
            grant_id: Grant ID
            research_question: Main research question
            research_gap: Description of gap being addressed
            methodology: Proposed methodology

        Returns:
            ProposalOutline
        """

        if grant_id not in self.grants:
            raise ValueError(f"Grant not found: {grant_id}")

        grant = self.grants[grant_id]

        # Project summary (1 page)
        project_summary = f"""
PROJECT SUMMARY

Title: {research_question}

Overview:
This project addresses a critical gap in {research_gap}. {research_question}

Intellectual Merit:
This research advances the state-of-the-art by combining novel methodologies with established frameworks. Expected outcomes include new algorithms, datasets, and theoretical insights.

Broader Impacts:
Results will benefit society through improved {self._extract_application_domain(research_gap)}, educational materials for students, and open-source software for the research community.

Keywords: {', '.join(grant.keywords[:5])}
"""

        # Project description sections
        project_description = {
            "Background and Significance": f"""
## Background and Significance

{research_question} This question is motivated by the following gap: {research_gap}

Recent advances in {grant.keywords[0] if grant.keywords else 'the field'} have enabled new approaches, but significant challenges remain. This project builds on prior work while introducing innovative techniques.

The significance of this research includes:
1. Advancing theoretical understanding of {grant.topics[0] if grant.topics else 'the domain'}
2. Developing practical tools and methods
3. Training the next generation of researchers
4. Contributing to societal benefit through {self._extract_application_domain(research_gap)}
""",

            "Research Objectives": f"""
## Research Objectives

This project has three primary objectives:

**Objective 1:** Develop novel {grant.keywords[0] if grant.keywords else 'methods'} for {research_question.split('?')[0].lower()}

**Objective 2:** Validate approaches through comprehensive experiments on benchmark datasets

**Objective 3:** Disseminate findings through publications, open-source software, and educational materials

Each objective is achievable within the proposed timeline and budget.
""",

            "Methodology": f"""
## Methodology

{methodology if methodology else f"The project will employ a mixed-methods approach combining theoretical analysis, algorithm development, and empirical validation."}

**Phase 1 (Months 1-12):** Develop core algorithms and frameworks
**Phase 2 (Months 13-24):** Conduct experiments and collect data
**Phase 3 (Months 25-36):** Analyze results and disseminate findings

Validation will be performed using established benchmarks and newly collected datasets. Statistical rigor will be ensured through proper experimental design.
""",

            "Expected Outcomes": f"""
## Expected Outcomes

This project will produce:

1. **Publications:** 6-8 peer-reviewed papers in top-tier venues
2. **Software:** Open-source implementations released on GitHub
3. **Datasets:** New benchmark datasets for community use
4. **Education:** Course materials and tutorials for students
5. **Broader Impacts:** Applications to {self._extract_application_domain(research_gap)}

All results will be made publicly available to maximize impact.
"""
        }

        # Timeline
        timeline = [
            {"phase": "Year 1", "milestones": ["Develop core framework", "Preliminary experiments", "First publications"]},
            {"phase": "Year 2", "milestones": ["Scale experiments", "Collect datasets", "Software release v1.0"]},
            {"phase": "Year 3", "milestones": ["Final validation", "Complete publications", "Dissemination"]}
        ]

        # Budget (sample)
        budget = {
            "Personnel": grant.funding_amount_max * 0.6,
            "Equipment": grant.funding_amount_max * 0.15,
            "Travel": grant.funding_amount_max * 0.05,
            "Other Direct Costs": grant.funding_amount_max * 0.1,
            "Indirect Costs": grant.funding_amount_max * 0.1
        }

        # Personnel
        personnel = [
            {"role": "PI", "effort": "2 months/year", "justification": "Overall project leadership"},
            {"role": "Co-PI", "effort": "1 month/year", "justification": "Domain expertise"},
            {"role": "Postdoc", "effort": "12 months/year", "justification": "Lead algorithm development"},
            {"role": "PhD Student", "effort": "12 months/year", "justification": "Conduct experiments"},
            {"role": "Research Programmer", "effort": "6 months/year", "justification": "Software development"}
        ]

        # Calculate word count
        word_count = len(project_summary.split()) + sum(len(v.split()) for v in project_description.values())

        return ProposalOutline(
            grant_id=grant_id,
            grant_title=grant.title,
            research_question=research_question,
            project_summary=project_summary.strip(),
            project_description=project_description,
            timeline=timeline,
            budget=budget,
            personnel=personnel,
            word_count=word_count
        )

    def _extract_application_domain(self, research_gap: str) -> str:
        """Extract application domain from research gap"""
        gap_lower = research_gap.lower()

        if "health" in gap_lower or "medical" in gap_lower or "protein" in gap_lower:
            return "healthcare and medicine"
        elif "climate" in gap_lower or "environment" in gap_lower:
            return "environmental sustainability"
        elif "education" in gap_lower:
            return "education and learning"
        elif "security" in gap_lower or "defense" in gap_lower:
            return "national security"
        else:
            return "science and technology"

    def get_grant(self, grant_id: str) -> Optional[Grant]:
        """Get grant by ID"""
        return self.grants.get(grant_id)

    def list_active_grants(
        self,
        agency: Optional[GrantAgency] = None,
        min_amount: Optional[int] = None,
        max_amount: Optional[int] = None
    ) -> List[Grant]:
        """List active grants with optional filters"""

        grants = [g for g in self.grants.values() if g.is_active and g.deadline >= datetime.now()]

        if agency:
            grants = [g for g in grants if g.agency == agency]

        if min_amount:
            grants = [g for g in grants if g.funding_amount_max >= min_amount]

        if max_amount:
            grants = [g for g in grants if g.funding_amount_min <= max_amount]

        # Sort by deadline (closest first)
        grants.sort(key=lambda g: g.deadline)

        return grants


if __name__ == "__main__":
    # Example usage
    print("Grant Matcher - Example")

    matcher = GrantMatcher()

    # Find matching grants
    matches = matcher.find_matching_grants(
        research_gap="Apply transformer architectures to protein structure prediction",
        keywords=["AI", "machine learning", "transformers", "protein folding", "bioinformatics"],
        min_match_score=0.3
    )

    print(f"\nFound {len(matches)} matching grants:")
    for i, match in enumerate(matches, 1):
        print(f"\n{i}. {match.grant.title}")
        print(f"   Agency: {match.grant.agency.value.upper()}")
        print(f"   Match Score: {match.match_score:.2f}")
        print(f"   Funding: ${match.grant.funding_amount_min:,} - ${match.grant.funding_amount_max:,}")
        print(f"   Deadline: {match.grant.deadline.strftime('%Y-%m-%d')} ({match.days_until_deadline} days)")
        print(f"   Urgency: {match.urgency}")
        print(f"   Matched Keywords: {', '.join(match.matched_keywords[:3])}")

    # Generate proposal outline
    if matches:
        print(f"\nGenerating proposal outline for top match...")
        outline = matcher.generate_proposal_outline(
            grant_id=matches[0].grant.grant_id,
            research_question="Can transformer architectures improve protein structure prediction accuracy?",
            research_gap="protein folding using deep learning",
            methodology="We will develop a novel transformer-based architecture with geometric attention mechanisms"
        )

        print(f"\nProposal Outline Generated:")
        print(f"- Word Count: {outline.word_count}")
        print(f"- Sections: {len(outline.project_description)}")
        print(f"- Timeline: {len(outline.timeline)} phases")
        print(f"- Budget: ${sum(outline.budget.values()):,.0f}")
