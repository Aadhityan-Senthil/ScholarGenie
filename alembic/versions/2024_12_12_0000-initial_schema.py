"""Initial schema

Revision ID: initial_schema
Revises:
Create Date: 2024-12-12 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE userrole AS ENUM ('admin', 'researcher', 'viewer')")
    op.execute("CREATE TYPE subscriptiontier AS ENUM ('free', 'pro', 'enterprise')")
    op.execute("CREATE TYPE taskstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled')")

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False, unique=True),
        sa.Column('username', sa.String(100), nullable=False, unique=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255)),
        sa.Column('organization', sa.String(255)),
        sa.Column('role', sa.Enum('admin', 'researcher', 'viewer', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('is_verified', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('subscription_tier', sa.Enum('free', 'pro', 'enterprise', name='subscriptiontier'), nullable=False),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_username', 'users', ['username'])

    # API Keys table
    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(100)),
        sa.Column('scopes', postgresql.ARRAY(sa.String)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('last_used', sa.DateTime(timezone=True)),
    )
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'])

    # Papers table
    op.create_table(
        'papers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('paper_id', sa.String(255), nullable=False, unique=True),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('abstract', sa.Text),
        sa.Column('authors', postgresql.JSONB),
        sa.Column('year', sa.Integer),
        sa.Column('venue', sa.String(255)),
        sa.Column('doi', sa.String(255)),
        sa.Column('arxiv_id', sa.String(50)),
        sa.Column('pdf_url', sa.Text),
        sa.Column('pdf_s3_key', sa.String(500)),
        sa.Column('source', sa.String(50)),
        sa.Column('citation_count', sa.Integer, default=0),
        sa.Column('is_open_access', sa.Boolean, default=False),
        sa.Column('metadata', postgresql.JSONB),
        sa.Column('ingested_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_papers_paper_id', 'papers', ['paper_id'])
    op.create_index('ix_papers_year', 'papers', ['year'])
    op.create_index('ix_papers_doi', 'papers', ['doi'])
    op.create_index('ix_papers_arxiv_id', 'papers', ['arxiv_id'])
    op.create_index('ix_papers_year_title', 'papers', ['year', 'title'])
    op.create_index('ix_papers_source_year', 'papers', ['source', 'year'])

    # Paper Summaries table
    op.create_table(
        'paper_summaries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('tldr', sa.Text),
        sa.Column('short_summary', sa.Text),
        sa.Column('full_summary', sa.Text),
        sa.Column('keypoints', postgresql.ARRAY(sa.Text)),
        sa.Column('section_summaries', postgresql.JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Extracted Data table
    op.create_table(
        'extracted_data',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('research_questions', postgresql.ARRAY(sa.Text)),
        sa.Column('hypotheses', postgresql.ARRAY(sa.Text)),
        sa.Column('methods', postgresql.ARRAY(sa.Text)),
        sa.Column('datasets', postgresql.ARRAY(sa.Text)),
        sa.Column('models', postgresql.ARRAY(sa.Text)),
        sa.Column('metrics', postgresql.ARRAY(sa.Text)),
        sa.Column('key_findings', postgresql.ARRAY(sa.Text)),
        sa.Column('limitations', postgresql.ARRAY(sa.Text)),
        sa.Column('future_work', postgresql.ARRAY(sa.Text)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # KG Entities table
    op.create_table(
        'kg_entities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('entity_id', sa.String(255), nullable=False, unique=True),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('properties', postgresql.JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_kg_entities_entity_id', 'kg_entities', ['entity_id'])
    op.create_index('ix_kg_entities_entity_type', 'kg_entities', ['entity_type'])

    # KG Relationships table
    op.create_table(
        'kg_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_id', sa.String(255), sa.ForeignKey('kg_entities.entity_id'), nullable=False),
        sa.Column('target_id', sa.String(255), sa.ForeignKey('kg_entities.entity_id'), nullable=False),
        sa.Column('relation_type', sa.String(50), nullable=False),
        sa.Column('properties', postgresql.JSONB),
        sa.Column('confidence', sa.Float, default=1.0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_kg_rel_source_id', 'kg_relationships', ['source_id'])
    op.create_index('ix_kg_rel_target_id', 'kg_relationships', ['target_id'])
    op.create_index('ix_kg_rel_source_target', 'kg_relationships', ['source_id', 'target_id'])
    op.create_index('ix_kg_rel_type', 'kg_relationships', ['relation_type'])

    # Research Gaps table
    op.create_table(
        'research_gaps',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('gap_id', sa.String(255), nullable=False, unique=True),
        sa.Column('gap_type', sa.String(50), nullable=False),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('entities', postgresql.ARRAY(sa.Text)),
        sa.Column('confidence', sa.Float),
        sa.Column('potential_impact', sa.String(50)),
        sa.Column('supporting_evidence', postgresql.ARRAY(sa.Text)),
        sa.Column('related_papers', postgresql.ARRAY(postgresql.UUID(as_uuid=True))),
        sa.Column('discovered_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('discovered_by', sa.String(50)),
    )
    op.create_index('ix_research_gaps_gap_id', 'research_gaps', ['gap_id'])
    op.create_index('ix_research_gaps_gap_type', 'research_gaps', ['gap_type'])

    # Gap Validations table
    op.create_table(
        'gap_validations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('gap_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('research_gaps.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('is_valid', sa.Boolean),
        sa.Column('novelty_score', sa.Float),
        sa.Column('impact_score', sa.Float),
        sa.Column('feasibility_score', sa.Float),
        sa.Column('explanation', sa.Text),
        sa.Column('recommendations', postgresql.ARRAY(sa.Text)),
        sa.Column('validated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # Breakthrough Predictions table
    op.create_table(
        'breakthrough_predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('prediction_id', sa.String(255), nullable=False, unique=True),
        sa.Column('title', sa.Text, nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('causal_chain', postgresql.JSONB),
        sa.Column('required_elements', postgresql.ARRAY(sa.Text)),
        sa.Column('expected_impact', sa.Float),
        sa.Column('confidence', sa.Float),
        sa.Column('reasoning', sa.Text),
        sa.Column('prerequisites', postgresql.ARRAY(sa.Text)),
        sa.Column('timeline', sa.String(50)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_breakthrough_pred_id', 'breakthrough_predictions', ['prediction_id'])

    # Tasks table
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('task_id', sa.String(255), nullable=False, unique=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('status', sa.Enum('pending', 'running', 'completed', 'failed', 'cancelled', name='taskstatus'), nullable=False),
        sa.Column('input_params', postgresql.JSONB),
        sa.Column('result', postgresql.JSONB),
        sa.Column('error', sa.Text),
        sa.Column('progress', sa.Integer, default=0),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_tasks_task_id', 'tasks', ['task_id'])
    op.create_index('ix_tasks_task_type', 'tasks', ['task_type'])
    op.create_index('ix_tasks_status', 'tasks', ['status'])
    op.create_index('ix_tasks_created_at', 'tasks', ['created_at'])
    op.create_index('ix_tasks_user_status', 'tasks', ['user_id', 'status'])
    op.create_index('ix_tasks_type_status', 'tasks', ['task_type', 'status'])

    # Task Logs table
    op.create_table(
        'task_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('level', sa.String(20)),
        sa.Column('message', sa.Text),
        sa.Column('metadata', postgresql.JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_task_logs_task_id', 'task_logs', ['task_id'])

    # Usage Analytics table
    op.create_table(
        'usage_analytics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50)),
        sa.Column('resource_id', sa.String(255)),
        sa.Column('metadata', postgresql.JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_usage_analytics_user_id', 'usage_analytics', ['user_id'])
    op.create_index('ix_usage_analytics_action', 'usage_analytics', ['action'])
    op.create_index('ix_usage_analytics_created_at', 'usage_analytics', ['created_at'])
    op.create_index('ix_usage_analytics_action_date', 'usage_analytics', ['action', 'created_at'])

    # API Requests table
    op.create_table(
        'api_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('endpoint', sa.String(255), nullable=False),
        sa.Column('method', sa.String(10)),
        sa.Column('status_code', sa.Integer),
        sa.Column('response_time_ms', sa.Integer),
        sa.Column('ip_address', postgresql.INET),
        sa.Column('user_agent', sa.Text),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index('ix_api_requests_user_id', 'api_requests', ['user_id'])
    op.create_index('ix_api_requests_endpoint', 'api_requests', ['endpoint'])
    op.create_index('ix_api_requests_created_at', 'api_requests', ['created_at'])
    op.create_index('ix_api_requests_endpoint_date', 'api_requests', ['endpoint', 'created_at'])
    op.create_index('ix_api_requests_user_date', 'api_requests', ['user_id', 'created_at'])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('api_requests')
    op.drop_table('usage_analytics')
    op.drop_table('task_logs')
    op.drop_table('tasks')
    op.drop_table('breakthrough_predictions')
    op.drop_table('gap_validations')
    op.drop_table('research_gaps')
    op.drop_table('kg_relationships')
    op.drop_table('kg_entities')
    op.drop_table('extracted_data')
    op.drop_table('paper_summaries')
    op.drop_table('papers')
    op.drop_table('api_keys')
    op.drop_table('users')

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS taskstatus")
    op.execute("DROP TYPE IF EXISTS subscriptiontier")
    op.execute("DROP TYPE IF EXISTS userrole")
