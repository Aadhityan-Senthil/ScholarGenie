"""
Live Research Alerts & Monitoring System

This agent provides real-time monitoring of scientific literature:
1. Daily monitoring of arXiv, PubMed, and other sources
2. Automatic analysis of new papers matching user interests
3. Alerts on competitor publications
4. Trend detection in research areas
5. Custom alert rules and filters
6. Email/webhook notifications

Usage:
    monitor = ResearchMonitor()
    alert_config = AlertConfig(
        keywords=['transformers', 'attention'],
        authors=['Vaswani'],
        min_relevance_score=0.8
    )
    monitor.add_alert(user_id='user1', config=alert_config)
    new_papers = await monitor.check_for_updates()
"""

import asyncio
import arxiv
import httpx
from typing import List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AlertSource(Enum):
    """Sources for monitoring"""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    GOOGLE_SCHOLAR = "google_scholar"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    KEYWORD_MATCH = "keyword_match"
    AUTHOR_MATCH = "author_match"
    CITATION_ALERT = "citation_alert"  # Your paper was cited
    COMPETITOR_ALERT = "competitor_alert"
    TREND_ALERT = "trend_alert"
    BREAKTHROUGH_ALERT = "breakthrough_alert"


@dataclass
class AlertConfig:
    """Configuration for a research alert"""
    alert_id: str
    user_id: str
    name: str
    description: Optional[str] = None

    # Matching criteria
    keywords: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    affiliations: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)  # arXiv categories

    # Advanced filters
    min_relevance_score: float = 0.7
    exclude_keywords: List[str] = field(default_factory=list)
    min_citation_count: int = 0
    publication_date_after: Optional[datetime] = None

    # Alert settings
    sources: List[AlertSource] = field(default_factory=lambda: [AlertSource.ARXIV])
    check_frequency: str = "daily"  # "hourly", "daily", "weekly"
    priority: AlertPriority = AlertPriority.MEDIUM
    enabled: bool = True

    # Notification settings
    email_notification: bool = True
    webhook_url: Optional[str] = None
    max_alerts_per_day: int = 50

    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_checked: Optional[datetime] = None
    total_alerts_sent: int = 0


@dataclass
class ResearchAlert:
    """An individual research alert notification"""
    alert_id: str
    paper_id: str
    alert_config_id: str
    user_id: str

    # Paper details
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    source: AlertSource
    pdf_url: Optional[str] = None
    arxiv_id: Optional[str] = None

    # Matching details
    match_type: AlertType
    relevance_score: float
    matched_keywords: List[str] = field(default_factory=list)
    matched_authors: List[str] = field(default_factory=list)
    reason: str = ""

    # Alert metadata
    priority: AlertPriority = AlertPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    read: bool = False
    dismissed: bool = False

    # Analysis
    auto_summary: Optional[str] = None
    key_findings: List[str] = field(default_factory=list)
    relevance_explanation: Optional[str] = None


@dataclass
class TrendAlert:
    """Alert about emerging research trends"""
    trend_id: str
    trend_name: str
    description: str

    # Trend metrics
    paper_count: int
    growth_rate: float  # Papers per week
    key_papers: List[str]
    top_authors: List[str]
    top_venues: List[str]

    # Temporal data
    first_observed: datetime
    peak_activity: Optional[datetime] = None

    # User relevance
    relevance_to_user: float = 0.0
    user_interests_matched: List[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompetitorAlert:
    """Alert about competitor activity"""
    competitor_name: str  # Author or institution
    paper_id: str
    title: str
    published_date: datetime

    # Competitive analysis
    overlapping_keywords: List[str]
    competitive_advantage: str  # What they did differently
    threat_level: str  # "low", "medium", "high"

    # Recommendations
    action_items: List[str]

    created_at: datetime = field(default_factory=datetime.now)


class ResearchMonitor:
    """
    Monitors scientific literature sources for new papers matching user interests
    """

    def __init__(self, embedding_service=None, summarizer=None):
        self.alert_configs: Dict[str, AlertConfig] = {}
        self.user_alerts: Dict[str, List[str]] = defaultdict(list)  # user_id -> alert_config_ids
        self.sent_alerts: List[ResearchAlert] = []
        self.trend_alerts: List[TrendAlert] = []
        self.competitor_alerts: List[CompetitorAlert] = []

        # Services
        self.embedding_service = embedding_service
        self.summarizer = summarizer

        # Tracking
        self.last_check_time: Dict[AlertSource, datetime] = {}
        self.seen_papers: Set[str] = set()

        # Stats
        self.total_papers_checked = 0
        self.total_alerts_sent = 0

    def add_alert(self, config: AlertConfig) -> str:
        """
        Add a new alert configuration

        Args:
            config: AlertConfig object

        Returns:
            alert_id
        """
        self.alert_configs[config.alert_id] = config
        self.user_alerts[config.user_id].append(config.alert_id)

        logger.info(f"Alert added: {config.name} (ID: {config.alert_id}) for user {config.user_id}")
        return config.alert_id

    def remove_alert(self, alert_id: str):
        """Remove an alert configuration"""
        if alert_id in self.alert_configs:
            config = self.alert_configs[alert_id]
            self.user_alerts[config.user_id].remove(alert_id)
            del self.alert_configs[alert_id]
            logger.info(f"Alert removed: {alert_id}")

    def update_alert(self, alert_id: str, updates: Dict):
        """Update an existing alert configuration"""
        if alert_id in self.alert_configs:
            config = self.alert_configs[alert_id]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"Alert updated: {alert_id}")

    async def check_for_updates(self, source: Optional[AlertSource] = None) -> List[ResearchAlert]:
        """
        Check all sources for new papers and generate alerts

        Args:
            source: Optional specific source to check (otherwise check all)

        Returns:
            List of new alerts generated
        """
        new_alerts = []

        # Determine sources to check
        sources_to_check = [source] if source else list(AlertSource)

        for src in sources_to_check:
            if src == AlertSource.ARXIV:
                alerts = await self._check_arxiv()
                new_alerts.extend(alerts)
            elif src == AlertSource.PUBMED:
                alerts = await self._check_pubmed()
                new_alerts.extend(alerts)
            # Add more sources as needed

        # Update tracking
        self.sent_alerts.extend(new_alerts)
        self.total_alerts_sent += len(new_alerts)

        logger.info(f"Check complete: {len(new_alerts)} new alerts generated")
        return new_alerts

    async def _check_arxiv(self) -> List[ResearchAlert]:
        """
        Check arXiv for new papers matching alert criteria

        Returns:
            List of alerts
        """
        logger.info("Checking arXiv for updates...")
        alerts = []

        # Get active arxiv alert configs
        arxiv_configs = [
            config for config in self.alert_configs.values()
            if config.enabled and AlertSource.ARXIV in config.sources
        ]

        if not arxiv_configs:
            return alerts

        # Fetch recent papers from arXiv (last 7 days)
        date_threshold = datetime.now() - timedelta(days=7)

        for config in arxiv_configs:
            # Build arXiv query
            query_parts = []

            if config.keywords:
                keyword_query = " OR ".join([f'all:"{kw}"' for kw in config.keywords])
                query_parts.append(f"({keyword_query})")

            if config.authors:
                author_query = " OR ".join([f'au:"{author}"' for author in config.authors])
                query_parts.append(f"({author_query})")

            if config.categories:
                cat_query = " OR ".join([f'cat:{cat}' for cat in config.categories])
                query_parts.append(f"({cat_query})")

            if not query_parts:
                continue

            query = " AND ".join(query_parts)

            try:
                # Query arXiv
                search = arxiv.Search(
                    query=query,
                    max_results=50,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )

                for paper in search.results():
                    # Check if already seen
                    paper_id = paper.entry_id
                    if paper_id in self.seen_papers:
                        continue

                    # Check publication date
                    if paper.published < date_threshold:
                        continue

                    # Calculate relevance
                    relevance = self._calculate_relevance(paper, config)

                    if relevance >= config.min_relevance_score:
                        # Create alert
                        alert = ResearchAlert(
                            alert_id=f"alert_{len(self.sent_alerts)}",
                            paper_id=paper_id,
                            alert_config_id=config.alert_id,
                            user_id=config.user_id,
                            title=paper.title,
                            authors=[author.name for author in paper.authors],
                            abstract=paper.summary,
                            published_date=paper.published,
                            source=AlertSource.ARXIV,
                            pdf_url=paper.pdf_url,
                            arxiv_id=paper.entry_id.split('/')[-1],
                            match_type=AlertType.KEYWORD_MATCH,
                            relevance_score=relevance,
                            matched_keywords=self._find_matched_keywords(paper, config.keywords),
                            matched_authors=self._find_matched_authors(paper, config.authors),
                            priority=config.priority,
                            reason=self._generate_match_reason(paper, config)
                        )

                        # Generate auto-summary if summarizer available
                        if self.summarizer:
                            try:
                                alert.auto_summary = await self._generate_summary(paper.summary)
                            except:
                                pass

                        alerts.append(alert)
                        self.seen_papers.add(paper_id)
                        self.total_papers_checked += 1

                        # Update config stats
                        config.total_alerts_sent += 1

                        # Respect max alerts per day
                        if len(alerts) >= config.max_alerts_per_day:
                            break

            except Exception as e:
                logger.error(f"Error checking arXiv for config {config.alert_id}: {e}")
                continue

            # Update last checked time
            config.last_checked = datetime.now()

        self.last_check_time[AlertSource.ARXIV] = datetime.now()
        logger.info(f"arXiv check complete: {len(alerts)} alerts generated")
        return alerts

    async def _check_pubmed(self) -> List[ResearchAlert]:
        """
        Check PubMed for new papers

        Returns:
            List of alerts
        """
        # TODO: Implement PubMed checking
        logger.info("PubMed checking not yet implemented")
        return []

    def _calculate_relevance(self, paper, config: AlertConfig) -> float:
        """
        Calculate how relevant a paper is to the alert config

        Returns:
            Relevance score 0-1
        """
        score = 0.0
        weights = []

        # Keyword matching in title and abstract
        text = (paper.title + " " + paper.summary).lower()

        if config.keywords:
            keyword_matches = sum(1 for kw in config.keywords if kw.lower() in text)
            keyword_score = min(1.0, keyword_matches / max(1, len(config.keywords)))
            score += keyword_score * 0.4
            weights.append(0.4)

        # Author matching
        if config.authors:
            paper_authors = [author.name.lower() for author in paper.authors]
            author_matches = sum(
                1 for target_author in config.authors
                if any(target_author.lower() in pa for pa in paper_authors)
            )
            author_score = author_matches / len(config.authors) if config.authors else 0
            score += author_score * 0.3
            weights.append(0.3)

        # Category matching
        if config.categories:
            paper_cats = [cat.lower() for cat in paper.categories]
            cat_matches = sum(1 for cat in config.categories if cat.lower() in paper_cats)
            cat_score = cat_matches / len(config.categories) if config.categories else 0
            score += cat_score * 0.2
            weights.append(0.2)

        # Venue matching (if applicable)
        if config.venues:
            venue_score = 0.1  # Placeholder
            score += venue_score * 0.1
            weights.append(0.1)

        # Normalize by total weight
        total_weight = sum(weights) if weights else 1.0
        return score / total_weight if total_weight > 0 else 0.0

    def _find_matched_keywords(self, paper, keywords: List[str]) -> List[str]:
        """Find which keywords matched in the paper"""
        text = (paper.title + " " + paper.summary).lower()
        return [kw for kw in keywords if kw.lower() in text]

    def _find_matched_authors(self, paper, target_authors: List[str]) -> List[str]:
        """Find which authors matched"""
        paper_authors = [author.name.lower() for author in paper.authors]
        matched = []
        for target in target_authors:
            target_lower = target.lower()
            for pa in paper_authors:
                if target_lower in pa:
                    matched.append(target)
                    break
        return matched

    def _generate_match_reason(self, paper, config: AlertConfig) -> str:
        """Generate human-readable reason for the match"""
        reasons = []

        matched_keywords = self._find_matched_keywords(paper, config.keywords)
        if matched_keywords:
            reasons.append(f"Keywords: {', '.join(matched_keywords[:3])}")

        matched_authors = self._find_matched_authors(paper, config.authors)
        if matched_authors:
            reasons.append(f"Authors: {', '.join(matched_authors[:3])}")

        if not reasons:
            reasons.append("General relevance to your interests")

        return " | ".join(reasons)

    async def _generate_summary(self, abstract: str) -> str:
        """Generate a brief summary of the abstract"""
        # Simple truncation for now
        # In production, use actual summarizer
        return abstract[:200] + "..." if len(abstract) > 200 else abstract

    def detect_trends(self, time_window_days: int = 30) -> List[TrendAlert]:
        """
        Detect emerging trends in recent alerts

        Args:
            time_window_days: Look at alerts from last N days

        Returns:
            List of detected trends
        """
        cutoff_date = datetime.now() - timedelta(days=time_window_days)

        # Get recent alerts
        recent_alerts = [a for a in self.sent_alerts if a.created_at >= cutoff_date]

        if len(recent_alerts) < 10:
            return []

        # Extract keywords from alerts
        keyword_counts = defaultdict(int)
        keyword_papers = defaultdict(list)

        for alert in recent_alerts:
            for keyword in alert.matched_keywords:
                keyword_counts[keyword] += 1
                keyword_papers[keyword].append(alert.paper_id)

        # Find rapidly growing keywords
        trends = []
        for keyword, count in keyword_counts.items():
            if count >= 5:  # Minimum threshold
                growth_rate = count / time_window_days * 7  # Papers per week

                trend = TrendAlert(
                    trend_id=f"trend_{len(trends)}",
                    trend_name=keyword,
                    description=f"Emerging trend: '{keyword}' appearing in {count} papers",
                    paper_count=count,
                    growth_rate=growth_rate,
                    key_papers=keyword_papers[keyword][:10],
                    top_authors=[],  # TODO: Extract from papers
                    top_venues=[],
                    first_observed=min([a.created_at for a in recent_alerts if keyword in a.matched_keywords])
                )

                trends.append(trend)

        # Sort by growth rate
        trends.sort(key=lambda x: x.growth_rate, reverse=True)

        self.trend_alerts.extend(trends[:20])  # Keep top 20
        logger.info(f"Detected {len(trends)} trends")
        return trends

    def detect_competitor_activity(self, competitors: List[str]) -> List[CompetitorAlert]:
        """
        Detect when competitors publish papers

        Args:
            competitors: List of competitor names (authors or institutions)

        Returns:
            List of competitor alerts
        """
        competitor_alerts = []

        for alert in self.sent_alerts:
            for competitor in competitors:
                # Check if competitor is in authors
                if any(competitor.lower() in author.lower() for author in alert.authors):
                    comp_alert = CompetitorAlert(
                        competitor_name=competitor,
                        paper_id=alert.paper_id,
                        title=alert.title,
                        published_date=alert.published_date,
                        overlapping_keywords=alert.matched_keywords,
                        competitive_advantage="Analysis pending",
                        threat_level="medium",
                        action_items=[
                            "Review paper in detail",
                            "Compare with our approach",
                            "Consider collaboration or differentiation"
                        ]
                    )
                    competitor_alerts.append(comp_alert)

        self.competitor_alerts.extend(competitor_alerts)
        logger.info(f"Detected {len(competitor_alerts)} competitor publications")
        return competitor_alerts

    def get_user_alerts(self, user_id: str, unread_only: bool = False) -> List[ResearchAlert]:
        """Get all alerts for a user"""
        alerts = [a for a in self.sent_alerts if a.user_id == user_id]

        if unread_only:
            alerts = [a for a in alerts if not a.read]

        return sorted(alerts, key=lambda x: x.created_at, reverse=True)

    def mark_alert_read(self, alert_id: str):
        """Mark an alert as read"""
        for alert in self.sent_alerts:
            if alert.alert_id == alert_id:
                alert.read = True
                break

    def dismiss_alert(self, alert_id: str):
        """Dismiss an alert"""
        for alert in self.sent_alerts:
            if alert.alert_id == alert_id:
                alert.dismissed = True
                break

    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            'total_alert_configs': len(self.alert_configs),
            'active_configs': sum(1 for c in self.alert_configs.values() if c.enabled),
            'total_papers_checked': self.total_papers_checked,
            'total_alerts_sent': self.total_alerts_sent,
            'unique_papers_seen': len(self.seen_papers),
            'trend_alerts': len(self.trend_alerts),
            'competitor_alerts': len(self.competitor_alerts),
            'last_check_times': {
                source.value: time.isoformat() if time else None
                for source, time in self.last_check_time.items()
            }
        }

    def export_alerts(self, user_id: str, format: str = "json") -> str:
        """Export user's alerts to JSON/CSV"""
        alerts = self.get_user_alerts(user_id)

        if format == "json":
            return json.dumps([{
                'alert_id': a.alert_id,
                'title': a.title,
                'authors': a.authors,
                'published_date': a.published_date.isoformat(),
                'relevance_score': a.relevance_score,
                'matched_keywords': a.matched_keywords,
                'reason': a.reason,
                'pdf_url': a.pdf_url
            } for a in alerts], indent=2)

        elif format == "csv":
            # TODO: Implement CSV export
            return "CSV export not yet implemented"

        return ""


if __name__ == "__main__":
    # Example usage
    print("Research Monitor - Example")

    monitor = ResearchMonitor()

    # Create alert config
    alert_config = AlertConfig(
        alert_id="alert_1",
        user_id="user1",
        name="Transformer Research",
        description="Track new papers on transformers and attention mechanisms",
        keywords=["transformer", "attention", "BERT", "GPT"],
        authors=["Vaswani", "Devlin"],
        categories=["cs.CL", "cs.LG"],
        min_relevance_score=0.7,
        sources=[AlertSource.ARXIV],
        check_frequency="daily"
    )

    monitor.add_alert(alert_config)

    # Check for updates (async)
    async def run_check():
        alerts = await monitor.check_for_updates()
        print(f"\nFound {len(alerts)} matching papers")

        for alert in alerts[:5]:
            print(f"\n- {alert.title}")
            print(f"  Authors: {', '.join(alert.authors[:3])}")
            print(f"  Relevance: {alert.relevance_score:.2f}")
            print(f"  Reason: {alert.reason}")

    # Run
    asyncio.run(run_check())

    # Get stats
    stats = monitor.get_statistics()
    print(f"\nMonitor Stats: {stats}")
