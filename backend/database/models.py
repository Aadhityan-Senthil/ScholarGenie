"""SQLAlchemy database models for ScholarGenie Enterprise."""

import uuid
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, JSON, ARRAY, Enum as SQLEnum, Index
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class UserRole(str, enum.Enum):
    """User roles."""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class SubscriptionTier(str, enum.Enum):
    """Subscription tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TaskStatus(str, enum.Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# USER & AUTHENTICATION MODELS
# ============================================================================

class User(Base):
    """User model."""
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    organization: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), default=UserRole.RESEARCHER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    subscription_tier: Mapped[SubscriptionTier] = mapped_column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    papers = relationship("Paper", back_populates="ingested_by_user")
    tasks = relationship("Task", back_populates="user")
    analytics = relationship("UsageAnalytics", back_populates="user")

    def __repr__(self):
        return f"<User {self.username}>"


class APIKey(Base):
    """API Key model."""
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    scopes: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="api_keys")


# ============================================================================
# PAPER & KNOWLEDGE MODELS
# ============================================================================

class Paper(Base):
    """Paper model."""
    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[Optional[dict]] = mapped_column(JSONB)
    year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    venue: Mapped[Optional[str]] = mapped_column(String(255))
    doi: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    arxiv_id: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(Text)
    pdf_s3_key: Mapped[Optional[str]] = mapped_column(String(500))
    source: Mapped[Optional[str]] = mapped_column(String(50))
    citation_count: Mapped[int] = mapped_column(Integer, default=0)
    is_open_access: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    ingested_by: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    ingested_by_user = relationship("User", back_populates="papers")
    summary = relationship("PaperSummary", back_populates="paper", uselist=False, cascade="all, delete-orphan")
    extracted_data = relationship("ExtractedData", back_populates="paper", uselist=False, cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_paper_year_title', 'year', 'title'),
        Index('idx_paper_source_year', 'source', 'year'),
    )


class PaperSummary(Base):
    """Paper summary model."""
    __tablename__ = "paper_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), unique=True)
    tldr: Mapped[Optional[str]] = mapped_column(Text)
    short_summary: Mapped[Optional[str]] = mapped_column(Text)
    full_summary: Mapped[Optional[str]] = mapped_column(Text)
    keypoints: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    section_summaries: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper = relationship("Paper", back_populates="summary")


class ExtractedData(Base):
    """Extracted data model."""
    __tablename__ = "extracted_data"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), unique=True)
    research_questions: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    hypotheses: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    methods: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    datasets: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    models: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    metrics: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    key_findings: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    limitations: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    future_work: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    paper = relationship("Paper", back_populates="extracted_data")


# ============================================================================
# KNOWLEDGE GRAPH MODELS
# ============================================================================

class KGEntity(Base):
    """Knowledge graph entity model."""
    __tablename__ = "kg_entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    properties: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    outgoing_relations = relationship("KGRelationship", foreign_keys="KGRelationship.source_id", back_populates="source")
    incoming_relations = relationship("KGRelationship", foreign_keys="KGRelationship.target_id", back_populates="target")


class KGRelationship(Base):
    """Knowledge graph relationship model."""
    __tablename__ = "kg_relationships"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[str] = mapped_column(String(255), ForeignKey("kg_entities.entity_id"), index=True)
    target_id: Mapped[str] = mapped_column(String(255), ForeignKey("kg_entities.entity_id"), index=True)
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    properties: Mapped[Optional[dict]] = mapped_column(JSONB)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    source = relationship("KGEntity", foreign_keys=[source_id], back_populates="outgoing_relations")
    target = relationship("KGEntity", foreign_keys=[target_id], back_populates="incoming_relations")

    # Indexes
    __table_args__ = (
        Index('idx_kg_rel_source_target', 'source_id', 'target_id'),
        Index('idx_kg_rel_type', 'relation_type'),
    )


# ============================================================================
# RESEARCH GAP & DISCOVERY MODELS
# ============================================================================

class ResearchGap(Base):
    """Research gap model."""
    __tablename__ = "research_gaps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gap_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    gap_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    entities: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    potential_impact: Mapped[Optional[str]] = mapped_column(String(50))
    supporting_evidence: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    related_papers: Mapped[Optional[List[uuid.UUID]]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    discovered_by: Mapped[Optional[str]] = mapped_column(String(50))

    # Relationships
    validation = relationship("GapValidation", back_populates="gap", uselist=False, cascade="all, delete-orphan")


class GapValidation(Base):
    """Gap validation model."""
    __tablename__ = "gap_validations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gap_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("research_gaps.id", ondelete="CASCADE"), unique=True)
    is_valid: Mapped[Optional[bool]] = mapped_column(Boolean)
    novelty_score: Mapped[Optional[float]] = mapped_column(Float)
    impact_score: Mapped[Optional[float]] = mapped_column(Float)
    feasibility_score: Mapped[Optional[float]] = mapped_column(Float)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    recommendations: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    validated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    gap = relationship("ResearchGap", back_populates="validation")


class BreakthroughPrediction(Base):
    """Breakthrough prediction model."""
    __tablename__ = "breakthrough_predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    causal_chain: Mapped[Optional[dict]] = mapped_column(JSONB)
    required_elements: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    expected_impact: Mapped[Optional[float]] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    prerequisites: Mapped[Optional[List[str]]] = mapped_column(ARRAY(Text))
    timeline: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# TASK & JOB MODELS
# ============================================================================

class Task(Base):
    """Task model."""
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    task_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[TaskStatus] = mapped_column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)
    input_params: Mapped[Optional[dict]] = mapped_column(JSONB)
    result: Mapped[Optional[dict]] = mapped_column(JSONB)
    error: Mapped[Optional[str]] = mapped_column(Text)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="tasks")
    logs = relationship("TaskLog", back_populates="task", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_task_user_status', 'user_id', 'status'),
        Index('idx_task_type_status', 'task_type', 'status'),
    )


class TaskLog(Base):
    """Task log model."""
    __tablename__ = "task_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    level: Mapped[Optional[str]] = mapped_column(String(20))
    message: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    task = relationship("Task", back_populates="logs")


# ============================================================================
# ANALYTICS & USAGE MODELS
# ============================================================================

class UsageAnalytics(Base):
    """Usage analytics model."""
    __tablename__ = "usage_analytics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50))
    resource_id: Mapped[Optional[str]] = mapped_column(String(255))
    metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="analytics")

    # Indexes
    __table_args__ = (
        Index('idx_analytics_action_date', 'action', 'created_at'),
    )


class APIRequest(Base):
    """API request log model."""
    __tablename__ = "api_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    method: Mapped[Optional[str]] = mapped_column(String(10))
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    ip_address: Mapped[Optional[str]] = mapped_column(INET)
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Indexes
    __table_args__ = (
        Index('idx_api_req_endpoint_date', 'endpoint', 'created_at'),
        Index('idx_api_req_user_date', 'user_id', 'created_at'),
    )
