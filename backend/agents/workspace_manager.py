"""
Collaborative Research Workspaces

This agent enables team collaboration on research projects:
1. Create and manage shared workspaces
2. Real-time collaboration on paper analysis
3. Task assignment and tracking
4. Shared annotations and notes
5. Discussion threads on papers
6. Version control for analyses
7. Team activity feed

Usage:
    manager = WorkspaceManager()
    workspace = manager.create_workspace(
        name="Transformer Research",
        owner_id="user1",
        members=["user2", "user3"]
    )
    manager.add_paper_to_workspace(workspace_id, paper_id)
    manager.add_annotation(workspace_id, paper_id, "Important finding!", user_id)
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid


class WorkspaceRole(Enum):
    """User roles in workspace"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class TaskStatus(Enum):
    """Task completion status"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ActivityType(Enum):
    """Types of workspace activities"""
    WORKSPACE_CREATED = "workspace_created"
    MEMBER_ADDED = "member_added"
    MEMBER_REMOVED = "member_removed"
    PAPER_ADDED = "paper_added"
    PAPER_REMOVED = "paper_removed"
    ANNOTATION_ADDED = "annotation_added"
    COMMENT_ADDED = "comment_added"
    TASK_CREATED = "task_created"
    TASK_UPDATED = "task_updated"
    TASK_COMPLETED = "task_completed"
    ANALYSIS_SHARED = "analysis_shared"


@dataclass
class WorkspaceMember:
    """Workspace member information"""
    user_id: str
    username: str
    email: str
    role: WorkspaceRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    contributions: int = 0  # Number of annotations, comments, etc.


@dataclass
class Workspace:
    """Collaborative workspace"""
    workspace_id: str
    name: str
    description: Optional[str] = None
    owner_id: str

    # Members
    members: Dict[str, WorkspaceMember] = field(default_factory=dict)

    # Content
    paper_ids: List[str] = field(default_factory=list)
    shared_analyses: Dict[str, Dict] = field(default_factory=dict)  # analysis_id -> analysis data

    # Settings
    is_public: bool = False
    allow_external_sharing: bool = True
    require_approval_for_members: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    # Stats
    total_papers: int = 0
    total_annotations: int = 0
    total_tasks: int = 0


@dataclass
class PaperAnnotation:
    """Annotation on a paper"""
    annotation_id: str
    workspace_id: str
    paper_id: str
    user_id: str
    username: str

    # Content
    text: str
    highlight_text: Optional[str] = None  # Text being highlighted
    page_number: Optional[int] = None
    position: Optional[Dict] = None  # {x, y, width, height} for PDF coordinates

    # Type
    annotation_type: str = "note"  # "note", "question", "insight", "critique", "todo"
    tags: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False

    # Engagement
    replies: List[str] = field(default_factory=list)  # Comment IDs
    upvotes: int = 0
    upvoters: Set[str] = field(default_factory=set)


@dataclass
class Comment:
    """Comment on annotation or paper"""
    comment_id: str
    workspace_id: str
    parent_id: str  # annotation_id or paper_id
    parent_type: str  # "annotation" or "paper"
    user_id: str
    username: str

    # Content
    text: str
    mentions: List[str] = field(default_factory=list)  # @username mentions

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_edited: bool = False

    # Engagement
    upvotes: int = 0
    upvoters: Set[str] = field(default_factory=set)


@dataclass
class Task:
    """Research task"""
    task_id: str
    workspace_id: str
    title: str
    description: Optional[str] = None

    # Assignment
    created_by: str
    assigned_to: Optional[str] = None
    assigned_to_username: Optional[str] = None

    # Status
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM

    # Related items
    related_papers: List[str] = field(default_factory=list)
    related_annotations: List[str] = field(default_factory=list)

    # Deadlines
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    # Progress
    progress_percentage: int = 0
    subtasks: List[Dict] = field(default_factory=list)


@dataclass
class Activity:
    """Activity log entry"""
    activity_id: str
    workspace_id: str
    user_id: str
    username: str
    activity_type: ActivityType

    # Content
    description: str
    metadata: Dict = field(default_factory=dict)

    # Timestamp
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SharedAnalysis:
    """Shared analysis result"""
    analysis_id: str
    workspace_id: str
    paper_id: str
    shared_by: str
    shared_by_username: str

    # Analysis content
    analysis_type: str  # "summary", "extraction", "gap_analysis", etc.
    content: Dict

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    previous_versions: List[Dict] = field(default_factory=list)

    # Engagement
    views: int = 0
    viewers: Set[str] = field(default_factory=set)


class WorkspaceManager:
    """
    Manages collaborative research workspaces
    """

    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}
        self.user_workspaces: Dict[str, List[str]] = {}  # user_id -> workspace_ids

        self.annotations: Dict[str, List[PaperAnnotation]] = {}  # workspace_id -> annotations
        self.comments: Dict[str, List[Comment]] = {}  # workspace_id -> comments
        self.tasks: Dict[str, List[Task]] = {}  # workspace_id -> tasks
        self.activities: Dict[str, List[Activity]] = {}  # workspace_id -> activities

    def create_workspace(
        self,
        name: str,
        owner_id: str,
        owner_username: str,
        owner_email: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = False
    ) -> Workspace:
        """
        Create a new workspace

        Args:
            name: Workspace name
            owner_id: Owner user ID
            owner_username: Owner username
            owner_email: Owner email
            description: Optional description
            tags: Optional tags
            is_public: Whether workspace is public

        Returns:
            Created workspace
        """
        workspace_id = str(uuid.uuid4())

        # Create owner member
        owner_member = WorkspaceMember(
            user_id=owner_id,
            username=owner_username,
            email=owner_email,
            role=WorkspaceRole.OWNER
        )

        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            owner_id=owner_id,
            members={owner_id: owner_member},
            is_public=is_public,
            tags=tags or []
        )

        self.workspaces[workspace_id] = workspace

        # Track user workspaces
        if owner_id not in self.user_workspaces:
            self.user_workspaces[owner_id] = []
        self.user_workspaces[owner_id].append(workspace_id)

        # Initialize collections
        self.annotations[workspace_id] = []
        self.comments[workspace_id] = []
        self.tasks[workspace_id] = []
        self.activities[workspace_id] = []

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=owner_id,
            username=owner_username,
            activity_type=ActivityType.WORKSPACE_CREATED,
            description=f"Created workspace '{name}'"
        )

        return workspace

    def add_member(
        self,
        workspace_id: str,
        user_id: str,
        username: str,
        email: str,
        role: WorkspaceRole = WorkspaceRole.MEMBER,
        added_by: str = None
    ) -> bool:
        """Add a member to workspace"""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]

        if user_id in workspace.members:
            return False  # Already a member

        member = WorkspaceMember(
            user_id=user_id,
            username=username,
            email=email,
            role=role
        )

        workspace.members[user_id] = member
        workspace.updated_at = datetime.now()

        # Track user workspaces
        if user_id not in self.user_workspaces:
            self.user_workspaces[user_id] = []
        self.user_workspaces[user_id].append(workspace_id)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=added_by or user_id,
            username=username,
            activity_type=ActivityType.MEMBER_ADDED,
            description=f"{username} joined the workspace",
            metadata={"new_member_id": user_id, "role": role.value}
        )

        return True

    def remove_member(self, workspace_id: str, user_id: str, removed_by: str) -> bool:
        """Remove a member from workspace"""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]

        if user_id not in workspace.members:
            return False

        if user_id == workspace.owner_id:
            return False  # Cannot remove owner

        member = workspace.members[user_id]
        del workspace.members[user_id]
        workspace.updated_at = datetime.now()

        # Update user workspaces
        if user_id in self.user_workspaces:
            self.user_workspaces[user_id].remove(workspace_id)

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=removed_by,
            username="System",
            activity_type=ActivityType.MEMBER_REMOVED,
            description=f"{member.username} left the workspace"
        )

        return True

    def add_paper_to_workspace(
        self,
        workspace_id: str,
        paper_id: str,
        added_by: str,
        username: str
    ) -> bool:
        """Add a paper to workspace"""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]

        if paper_id in workspace.paper_ids:
            return False  # Already added

        workspace.paper_ids.append(paper_id)
        workspace.total_papers += 1
        workspace.updated_at = datetime.now()

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=added_by,
            username=username,
            activity_type=ActivityType.PAPER_ADDED,
            description=f"Added paper to workspace",
            metadata={"paper_id": paper_id}
        )

        return True

    def add_annotation(
        self,
        workspace_id: str,
        paper_id: str,
        user_id: str,
        username: str,
        text: str,
        annotation_type: str = "note",
        highlight_text: Optional[str] = None,
        page_number: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> PaperAnnotation:
        """Add annotation to a paper"""
        annotation = PaperAnnotation(
            annotation_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            paper_id=paper_id,
            user_id=user_id,
            username=username,
            text=text,
            annotation_type=annotation_type,
            highlight_text=highlight_text,
            page_number=page_number,
            tags=tags or []
        )

        self.annotations[workspace_id].append(annotation)

        # Update workspace stats
        workspace = self.workspaces[workspace_id]
        workspace.total_annotations += 1
        workspace.members[user_id].contributions += 1

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=user_id,
            username=username,
            activity_type=ActivityType.ANNOTATION_ADDED,
            description=f"Added annotation on paper",
            metadata={"paper_id": paper_id, "annotation_type": annotation_type}
        )

        return annotation

    def add_comment(
        self,
        workspace_id: str,
        parent_id: str,
        parent_type: str,
        user_id: str,
        username: str,
        text: str,
        mentions: Optional[List[str]] = None
    ) -> Comment:
        """Add comment to annotation or paper"""
        comment = Comment(
            comment_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            parent_id=parent_id,
            parent_type=parent_type,
            user_id=user_id,
            username=username,
            text=text,
            mentions=mentions or []
        )

        self.comments[workspace_id].append(comment)

        # Update member contributions
        workspace = self.workspaces[workspace_id]
        workspace.members[user_id].contributions += 1

        # If it's a reply to annotation, add to annotation's replies
        if parent_type == "annotation":
            for annotation in self.annotations[workspace_id]:
                if annotation.annotation_id == parent_id:
                    annotation.replies.append(comment.comment_id)
                    break

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=user_id,
            username=username,
            activity_type=ActivityType.COMMENT_ADDED,
            description=f"Commented on {parent_type}"
        )

        return comment

    def create_task(
        self,
        workspace_id: str,
        title: str,
        created_by: str,
        creator_username: str,
        description: Optional[str] = None,
        assigned_to: Optional[str] = None,
        assigned_to_username: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: Optional[datetime] = None,
        related_papers: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Task:
        """Create a research task"""
        task = Task(
            task_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            title=title,
            description=description,
            created_by=created_by,
            assigned_to=assigned_to,
            assigned_to_username=assigned_to_username,
            priority=priority,
            due_date=due_date,
            related_papers=related_papers or [],
            tags=tags or []
        )

        self.tasks[workspace_id].append(task)

        # Update workspace stats
        workspace = self.workspaces[workspace_id]
        workspace.total_tasks += 1

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=created_by,
            username=creator_username,
            activity_type=ActivityType.TASK_CREATED,
            description=f"Created task: {title}",
            metadata={
                "task_id": task.task_id,
                "assigned_to": assigned_to,
                "priority": priority.value
            }
        )

        return task

    def update_task(
        self,
        workspace_id: str,
        task_id: str,
        updates: Dict,
        updated_by: str,
        username: str
    ) -> bool:
        """Update task fields"""
        if workspace_id not in self.tasks:
            return False

        for task in self.tasks[workspace_id]:
            if task.task_id == task_id:
                # Update fields
                for key, value in updates.items():
                    if hasattr(task, key):
                        if key == "status" and isinstance(value, str):
                            setattr(task, key, TaskStatus(value))
                        elif key == "priority" and isinstance(value, str):
                            setattr(task, key, TaskPriority(value))
                        else:
                            setattr(task, key, value)

                task.updated_at = datetime.now()

                # If status changed to completed
                if "status" in updates and updates["status"] == TaskStatus.COMPLETED.value:
                    task.completed_at = datetime.now()
                    task.progress_percentage = 100

                    # Log completion
                    self._log_activity(
                        workspace_id=workspace_id,
                        user_id=updated_by,
                        username=username,
                        activity_type=ActivityType.TASK_COMPLETED,
                        description=f"Completed task: {task.title}"
                    )
                else:
                    # Log update
                    self._log_activity(
                        workspace_id=workspace_id,
                        user_id=updated_by,
                        username=username,
                        activity_type=ActivityType.TASK_UPDATED,
                        description=f"Updated task: {task.title}"
                    )

                return True

        return False

    def share_analysis(
        self,
        workspace_id: str,
        paper_id: str,
        shared_by: str,
        username: str,
        analysis_type: str,
        content: Dict
    ) -> SharedAnalysis:
        """Share an analysis in workspace"""
        analysis = SharedAnalysis(
            analysis_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            paper_id=paper_id,
            shared_by=shared_by,
            shared_by_username=username,
            analysis_type=analysis_type,
            content=content
        )

        workspace = self.workspaces[workspace_id]
        workspace.shared_analyses[analysis.analysis_id] = analysis

        # Update member contributions
        workspace.members[shared_by].contributions += 1

        # Log activity
        self._log_activity(
            workspace_id=workspace_id,
            user_id=shared_by,
            username=username,
            activity_type=ActivityType.ANALYSIS_SHARED,
            description=f"Shared {analysis_type} analysis",
            metadata={"paper_id": paper_id, "analysis_type": analysis_type}
        )

        return analysis

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID"""
        return self.workspaces.get(workspace_id)

    def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get all workspaces for a user"""
        workspace_ids = self.user_workspaces.get(user_id, [])
        return [self.workspaces[wid] for wid in workspace_ids if wid in self.workspaces]

    def get_workspace_papers(self, workspace_id: str) -> List[str]:
        """Get all paper IDs in workspace"""
        workspace = self.workspaces.get(workspace_id)
        return workspace.paper_ids if workspace else []

    def get_annotations(
        self,
        workspace_id: str,
        paper_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[PaperAnnotation]:
        """Get annotations (optionally filtered by paper or user)"""
        annotations = self.annotations.get(workspace_id, [])

        if paper_id:
            annotations = [a for a in annotations if a.paper_id == paper_id]

        if user_id:
            annotations = [a for a in annotations if a.user_id == user_id]

        return sorted(annotations, key=lambda x: x.created_at, reverse=True)

    def get_comments(
        self,
        workspace_id: str,
        parent_id: Optional[str] = None
    ) -> List[Comment]:
        """Get comments (optionally filtered by parent)"""
        comments = self.comments.get(workspace_id, [])

        if parent_id:
            comments = [c for c in comments if c.parent_id == parent_id]

        return sorted(comments, key=lambda x: x.created_at)

    def get_tasks(
        self,
        workspace_id: str,
        assigned_to: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> List[Task]:
        """Get tasks (optionally filtered)"""
        tasks = self.tasks.get(workspace_id, [])

        if assigned_to:
            tasks = [t for t in tasks if t.assigned_to == assigned_to]

        if status:
            tasks = [t for t in tasks if t.status == status]

        return sorted(tasks, key=lambda x: (x.priority.value, x.created_at), reverse=True)

    def get_activities(
        self,
        workspace_id: str,
        limit: int = 50
    ) -> List[Activity]:
        """Get recent activities"""
        activities = self.activities.get(workspace_id, [])
        return sorted(activities, key=lambda x: x.created_at, reverse=True)[:limit]

    def get_workspace_stats(self, workspace_id: str) -> Dict:
        """Get workspace statistics"""
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return {}

        return {
            "total_members": len(workspace.members),
            "total_papers": workspace.total_papers,
            "total_annotations": workspace.total_annotations,
            "total_tasks": workspace.total_tasks,
            "tasks_completed": len([t for t in self.tasks.get(workspace_id, []) if t.status == TaskStatus.COMPLETED]),
            "total_comments": len(self.comments.get(workspace_id, [])),
            "total_activities": len(self.activities.get(workspace_id, [])),
            "most_active_members": self._get_most_active_members(workspace_id, top_k=5)
        }

    def _get_most_active_members(self, workspace_id: str, top_k: int = 5) -> List[Dict]:
        """Get most active members"""
        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return []

        members = sorted(
            workspace.members.values(),
            key=lambda m: m.contributions,
            reverse=True
        )[:top_k]

        return [
            {
                "user_id": m.user_id,
                "username": m.username,
                "contributions": m.contributions,
                "role": m.role.value
            }
            for m in members
        ]

    def _log_activity(
        self,
        workspace_id: str,
        user_id: str,
        username: str,
        activity_type: ActivityType,
        description: str,
        metadata: Optional[Dict] = None
    ):
        """Log an activity"""
        activity = Activity(
            activity_id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            user_id=user_id,
            username=username,
            activity_type=activity_type,
            description=description,
            metadata=metadata or {}
        )

        self.activities[workspace_id].append(activity)

        # Update member last active
        workspace = self.workspaces.get(workspace_id)
        if workspace and user_id in workspace.members:
            workspace.members[user_id].last_active = datetime.now()


if __name__ == "__main__":
    # Example usage
    print("Workspace Manager - Example")

    manager = WorkspaceManager()

    # Create workspace
    workspace = manager.create_workspace(
        name="Transformer Research Team",
        owner_id="user1",
        owner_username="Alice",
        owner_email="alice@university.edu",
        description="Collaborative research on transformer architectures",
        tags=["NLP", "Deep Learning"]
    )

    print(f"\nCreated workspace: {workspace.name} (ID: {workspace.workspace_id})")

    # Add members
    manager.add_member(workspace.workspace_id, "user2", "Bob", "bob@university.edu", WorkspaceRole.MEMBER)
    manager.add_member(workspace.workspace_id, "user3", "Carol", "carol@university.edu", WorkspaceRole.MEMBER)

    print(f"Members: {len(workspace.members)}")

    # Add papers
    manager.add_paper_to_workspace(workspace.workspace_id, "paper_attention", "user1", "Alice")

    # Add annotation
    annotation = manager.add_annotation(
        workspace_id=workspace.workspace_id,
        paper_id="paper_attention",
        user_id="user1",
        username="Alice",
        text="This is a groundbreaking approach to sequence modeling",
        annotation_type="insight",
        page_number=5
    )

    print(f"\nAdded annotation: {annotation.text}")

    # Create task
    task = manager.create_task(
        workspace_id=workspace.workspace_id,
        title="Review attention mechanism section",
        created_by="user1",
        creator_username="Alice",
        assigned_to="user2",
        assigned_to_username="Bob",
        priority=TaskPriority.HIGH,
        related_papers=["paper_attention"]
    )

    print(f"Created task: {task.title}")

    # Get stats
    stats = manager.get_workspace_stats(workspace.workspace_id)
    print(f"\nWorkspace stats: {stats}")
