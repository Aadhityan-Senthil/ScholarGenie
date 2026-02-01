"""Celery background tasks for ScholarGenie."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from celery import Task
from sqlalchemy.orm import Session

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.database.models import Paper, Task as DBTask, TaskStatus
from backend.agents.paper_finder import PaperFinderAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.knowledge_graph import KnowledgeGraphAgent
from backend.utils.cache import get_cache

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """Base task with database session management."""

    _db: Session = None

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        if self._db is not None:
            self._db.close()
            self._db = None


# =====================================================
# Paper Processing Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, bind=True, name="backend.tasks.process_paper")
def process_paper(self, paper_id: str, user_id: str = None) -> Dict[str, Any]:
    """
    Process a paper: parse PDF, extract metadata, generate summary.

    Args:
        paper_id: Paper ID to process
        user_id: User ID who initiated the task

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing paper {paper_id}")

    try:
        # Update task status
        task = self.db.query(DBTask).filter(DBTask.task_id == self.request.id).first()
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self.db.commit()

        # Get paper from database
        paper = self.db.query(Paper).filter(Paper.paper_id == paper_id).first()
        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        # Initialize agents
        summarizer = SummarizerAgent()

        # Generate summary (this would normally fetch full text)
        summary = summarizer.summarize_abstract(
            paper_id=paper_id,
            abstract=paper.abstract or "",
            title=paper.title
        )

        # Update task progress
        if task:
            task.progress = 100
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = {"summary": summary}
            self.db.commit()

        return {
            "status": "success",
            "paper_id": paper_id,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error processing paper {paper_id}: {e}")

        # Update task status
        if task:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            self.db.commit()

        raise


@celery_app.task(base=DatabaseTask, bind=True, name="backend.tasks.batch_process_papers")
def batch_process_papers(self, paper_ids: List[str]) -> Dict[str, Any]:
    """
    Process multiple papers in batch.

    Args:
        paper_ids: List of paper IDs to process

    Returns:
        Dictionary with batch processing results
    """
    logger.info(f"Batch processing {len(paper_ids)} papers")

    results = []
    for idx, paper_id in enumerate(paper_ids):
        try:
            result = process_paper.delay(paper_id)
            results.append({"paper_id": paper_id, "task_id": result.id, "status": "queued"})

            # Update progress
            progress = int((idx + 1) / len(paper_ids) * 100)
            self.update_state(state="PROGRESS", meta={"current": idx + 1, "total": len(paper_ids)})

        except Exception as e:
            logger.error(f"Error queuing paper {paper_id}: {e}")
            results.append({"paper_id": paper_id, "status": "error", "error": str(e)})

    return {
        "status": "success",
        "total": len(paper_ids),
        "results": results
    }


# =====================================================
# Research Monitoring Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, name="backend.tasks.monitor_new_papers")
def monitor_new_papers() -> Dict[str, Any]:
    """
    Periodic task to monitor for new papers matching user alerts.

    Runs hourly to check for new papers.
    """
    logger.info("Monitoring for new papers")

    try:
        # This would integrate with research_monitor agent
        # For now, just a placeholder

        return {
            "status": "success",
            "checked_at": datetime.utcnow().isoformat(),
            "new_papers": 0
        }

    except Exception as e:
        logger.error(f"Error monitoring papers: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(base=DatabaseTask, name="backend.tasks.update_citation_counts")
def update_citation_counts() -> Dict[str, Any]:
    """
    Periodic task to update citation counts for papers.

    Runs daily.
    """
    logger.info("Updating citation counts")

    try:
        db = SessionLocal()
        papers = db.query(Paper).all()

        updated = 0
        for paper in papers:
            # This would fetch real citation count from external API
            # For now, just increment as placeholder
            paper.citation_count = paper.citation_count or 0
            updated += 1

        db.commit()
        db.close()

        return {
            "status": "success",
            "updated": updated,
            "updated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error updating citation counts: {e}")
        return {"status": "error", "error": str(e)}


# =====================================================
# Knowledge Graph Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, bind=True, name="backend.tasks.refresh_knowledge_graph")
def refresh_knowledge_graph(self) -> Dict[str, Any]:
    """
    Periodic task to refresh the knowledge graph.

    Runs weekly to rebuild graph from latest papers.
    """
    logger.info("Refreshing knowledge graph")

    try:
        # This would rebuild the entire knowledge graph
        # Resource-intensive operation suitable for background processing

        self.update_state(state="PROGRESS", meta={"stage": "extracting entities"})
        # Extract entities from all papers

        self.update_state(state="PROGRESS", meta={"stage": "building relationships"})
        # Build relationships between entities

        self.update_state(state="PROGRESS", meta={"stage": "computing metrics"})
        # Compute graph metrics

        return {
            "status": "success",
            "refreshed_at": datetime.utcnow().isoformat(),
            "entities": 0,
            "relationships": 0
        }

    except Exception as e:
        logger.error(f"Error refreshing knowledge graph: {e}")
        return {"status": "error", "error": str(e)}


# =====================================================
# Maintenance Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, name="backend.tasks.cleanup_old_tasks")
def cleanup_old_tasks(days: int = 30) -> Dict[str, Any]:
    """
    Clean up old completed/failed tasks from database.

    Args:
        days: Delete tasks older than this many days

    Returns:
        Dictionary with cleanup results
    """
    logger.info(f"Cleaning up tasks older than {days} days")

    try:
        db = SessionLocal()
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Delete old completed/failed tasks
        deleted = db.query(DBTask).filter(
            DBTask.created_at < cutoff_date,
            DBTask.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED])
        ).delete()

        db.commit()
        db.close()

        return {
            "status": "success",
            "deleted": deleted,
            "cutoff_date": cutoff_date.isoformat()
        }

    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name="backend.tasks.cleanup_cache")
def cleanup_cache(pattern: str = None) -> Dict[str, Any]:
    """
    Clean up Redis cache entries.

    Args:
        pattern: Optional pattern to match keys (e.g., "paper:*")

    Returns:
        Dictionary with cleanup results
    """
    logger.info(f"Cleaning up cache with pattern: {pattern or 'all'}")

    try:
        cache = get_cache()

        if pattern:
            deleted = cache.clear_pattern(pattern)
        else:
            cache.flush_all()
            deleted = -1  # Unknown count

        return {
            "status": "success",
            "deleted": deleted,
            "pattern": pattern
        }

    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")
        return {"status": "error", "error": str(e)}


# =====================================================
# Literature Review Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, bind=True, name="backend.tasks.generate_literature_review")
def generate_literature_review(
    self,
    paper_ids: List[str],
    research_question: str,
    style: str = "narrative",
    user_id: str = None
) -> Dict[str, Any]:
    """
    Generate literature review in background.

    This is resource-intensive and suitable for async processing.

    Args:
        paper_ids: List of paper IDs to include
        research_question: Research question for review
        style: Review style
        user_id: User ID who initiated the task

    Returns:
        Dictionary with review generation results
    """
    logger.info(f"Generating literature review with {len(paper_ids)} papers")

    try:
        from backend.agents.lit_review_generator import LiteratureReviewGenerator, ReviewStyle

        # Update progress
        self.update_state(state="PROGRESS", meta={"stage": "fetching papers"})

        # Get papers from database
        db = SessionLocal()
        papers = db.query(Paper).filter(Paper.paper_id.in_(paper_ids)).all()

        # Convert to format expected by generator
        paper_data = [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors or [],
                "year": p.year,
                "abstract": p.abstract,
                "keywords": []  # Would extract from metadata
            }
            for p in papers
        ]

        self.update_state(state="PROGRESS", meta={"stage": "generating review"})

        # Generate review
        generator = LiteratureReviewGenerator()
        review = generator.generate_review(
            papers=paper_data,
            research_question=research_question,
            style=ReviewStyle(style)
        )

        db.close()

        return {
            "status": "success",
            "review_id": review.review_id,
            "paper_count": len(paper_data),
            "generated_at": review.generated_at.isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating literature review: {e}")
        raise


# =====================================================
# Grant Matching Tasks
# =====================================================

@celery_app.task(base=DatabaseTask, name="backend.tasks.scan_new_grants")
def scan_new_grants() -> Dict[str, Any]:
    """
    Scan for new grant opportunities and notify relevant users.

    This would integrate with external grant databases.
    """
    logger.info("Scanning for new grants")

    try:
        # This would:
        # 1. Fetch new grants from external APIs
        # 2. Match against user research interests
        # 3. Send notifications

        return {
            "status": "success",
            "new_grants": 0,
            "scanned_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error scanning grants: {e}")
        return {"status": "error", "error": str(e)}
