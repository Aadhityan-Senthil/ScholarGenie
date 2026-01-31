"""CrewAI multi-agent orchestration for ScholarGenie."""

from backend.crews.research_crew import ResearchCrew
from backend.crews.analysis_crew import AnalysisCrew
from backend.crews.discovery_crew import DiscoveryCrew

__all__ = ["ResearchCrew", "AnalysisCrew", "DiscoveryCrew"]
