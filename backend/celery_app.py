"""Celery application for ScholarGenie background tasks."""

import os
from celery import Celery
from celery.schedules import crontab

# Redis connection for Celery
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("CELERY_REDIS_DB", "1")  # Use different DB than cache
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Create Celery app
celery_app = Celery(
    "scholargenie",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard timeout
    task_soft_time_limit=3000,  # 50 minutes soft timeout
    task_acks_late=True,  # Acknowledge tasks after they complete
    task_reject_on_worker_lost=True,

    # Worker settings
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks

    # Result backend
    result_expires=86400,  # Results expire after 24 hours
    result_extended=True,

    # Beat schedule for periodic tasks
    beat_schedule={
        # Monitor new papers every hour
        "monitor-new-papers": {
            "task": "backend.tasks.monitor_new_papers",
            "schedule": crontab(minute=0),  # Every hour
        },

        # Update citation counts daily
        "update-citation-counts": {
            "task": "backend.tasks.update_citation_counts",
            "schedule": crontab(hour=2, minute=0),  # 2 AM daily
        },

        # Refresh knowledge graph weekly
        "refresh-knowledge-graph": {
            "task": "backend.tasks.refresh_knowledge_graph",
            "schedule": crontab(day_of_week=1, hour=3, minute=0),  # Monday 3 AM
        },

        # Clean up old tasks weekly
        "cleanup-old-tasks": {
            "task": "backend.tasks.cleanup_old_tasks",
            "schedule": crontab(day_of_week=0, hour=4, minute=0),  # Sunday 4 AM
        },
    },

    # Logging
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s",
)

if __name__ == "__main__":
    celery_app.start()
