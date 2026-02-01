"""Paper metadata models and utilities."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class Reference(BaseModel):
    """Reference/citation information."""
    title: str
    authors: Optional[List[str]] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None


class Figure(BaseModel):
    """Figure metadata."""
    id: str
    caption: str
    page: Optional[int] = None
    path: Optional[str] = None


class Section(BaseModel):
    """Document section."""
    title: str
    content: str
    level: int = 1
    subsections: List['Section'] = Field(default_factory=list)


Section.model_rebuild()


class PaperMetadata(BaseModel):
    """Complete paper metadata and content."""

    # Identifiers
    paper_id: str = Field(..., description="Unique paper identifier")
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None

    # Basic info
    title: str
    authors: List[Author] = Field(default_factory=list)
    abstract: Optional[str] = None

    # Publication info
    year: Optional[int] = None
    venue: Optional[str] = None
    publication_date: Optional[datetime] = None

    # Content
    sections: List[Section] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)

    # PDF info
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    is_open_access: bool = False

    # Metrics
    citation_count: Optional[int] = 0

    # Processing metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    source: Optional[str] = None  # arxiv, semantic_scholar, etc.

    # Additional fields
    keywords: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get full text of the paper."""
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}\n")

        if self.authors:
            author_names = [a.name for a in self.authors]
            parts.append(f"Authors: {', '.join(author_names)}\n")

        if self.abstract:
            parts.append(f"Abstract:\n{self.abstract}\n")

        for section in self.sections:
            parts.append(f"\n{section.title}\n{section.content}")

        return "\n".join(parts)

    def get_section_by_title(self, title: str) -> Optional[Section]:
        """Find a section by title (case-insensitive)."""
        title_lower = title.lower()
        for section in self.sections:
            if section.title.lower() == title_lower:
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        """Create from dictionary."""
        return cls(**data)


class Summary(BaseModel):
    """Paper summary at multiple granularities."""

    paper_id: str

    # Multi-level summaries
    tldr: str = Field(..., description="One-sentence summary")
    short_summary: str = Field(..., description="2-3 sentence summary")
    full_summary: str = Field(..., description="Detailed summary (200-400 words)")

    # Section summaries
    section_summaries: Dict[str, str] = Field(default_factory=dict)

    # Key information
    keypoints: List[str] = Field(default_factory=list, description="Bullet point key insights")
    methods: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        md = f"# Summary\n\n"
        md += f"## TL;DR\n{self.tldr}\n\n"
        md += f"## Overview\n{self.short_summary}\n\n"
        md += f"## Detailed Summary\n{self.full_summary}\n\n"

        if self.keypoints:
            md += "## Key Points\n"
            for point in self.keypoints:
                md += f"- {point}\n"
            md += "\n"

        if self.section_summaries:
            md += "## Section Summaries\n"
            for section, summary in self.section_summaries.items():
                md += f"### {section}\n{summary}\n\n"

        return md


class ExtractedData(BaseModel):
    """Extracted structured data from paper."""

    paper_id: str

    # Research elements
    research_question: Optional[str] = None
    hypothesis: Optional[str] = None

    # Methodology
    methods: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)

    # Results
    key_findings: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    # Additional
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)

    # Metadata
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
