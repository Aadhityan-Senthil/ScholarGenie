"""
Citation Network Analysis Agent

This agent builds and analyzes citation networks to:
1. Identify foundational papers using PageRank
2. Discover "hidden gem" papers with high betweenness centrality
3. Map research lineages and evolution
4. Predict future citation impact
5. Find influential authors and research communities
6. Detect citation patterns and anomalies

Usage:
    agent = CitationNetworkAgent()
    agent.build_network_from_papers(papers)
    foundational = agent.find_foundational_papers(top_k=10)
    prediction = agent.predict_citation_impact(paper_id)
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from enum import Enum


class CitationType(Enum):
    """Types of citation relationships"""
    DIRECT = "direct"  # Paper A cites Paper B
    CO_CITATION = "co_citation"  # Papers A and B both cite Paper C
    BIBLIOGRAPHIC_COUPLING = "bibliographic_coupling"  # Papers A and B cite common papers
    TEMPORAL = "temporal"  # Citation across time periods


class PaperCategory(Enum):
    """Paper importance categories"""
    FOUNDATIONAL = "foundational"  # High PageRank, many citations
    HIDDEN_GEM = "hidden_gem"  # High betweenness, low citations
    EMERGING_STAR = "emerging_star"  # Rapid citation growth
    BRIDGE = "bridge"  # Connects different communities
    RECENT = "recent"  # Published recently, still accumulating citations
    STANDARD = "standard"  # Normal citation patterns


@dataclass
class CitationNode:
    """Represents a paper in the citation network"""
    paper_id: str
    title: str
    authors: List[str]
    publication_date: datetime
    venue: Optional[str] = None

    # Citation metrics
    in_citations: int = 0  # Number of papers citing this
    out_citations: int = 0  # Number of papers this cites

    # Network metrics (computed)
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0

    # Derived metrics
    h_index: int = 0
    citation_velocity: float = 0.0  # Citations per month
    category: PaperCategory = PaperCategory.STANDARD

    # Community
    community_id: Optional[int] = None

    # Additional data
    abstract: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class CitationEdge:
    """Represents a citation relationship"""
    source_id: str  # Paper that cites
    target_id: str  # Paper being cited
    citation_type: CitationType
    context: Optional[str] = None  # Context of citation in text
    year_cited: Optional[int] = None
    weight: float = 1.0


@dataclass
class CitationPrediction:
    """Prediction of future citation impact"""
    paper_id: str
    predicted_citations_1year: int
    predicted_citations_3year: int
    predicted_citations_5year: int
    confidence_score: float  # 0-1
    factors: Dict[str, float]  # Feature importance
    comparable_papers: List[str]  # Similar papers for comparison


@dataclass
class ResearchLineage:
    """Traces the evolution of a research idea"""
    root_paper_id: str
    generations: List[List[str]]  # Papers grouped by generation
    key_papers: List[str]  # Most influential in lineage
    branches: List[Dict]  # Divergent research directions
    total_descendants: int


@dataclass
class CitationAnomaly:
    """Detected unusual citation pattern"""
    paper_id: str
    anomaly_type: str  # "citation_burst", "delayed_recognition", "self_citation_cluster"
    severity: float  # 0-1
    description: str
    evidence: Dict


class CitationNetworkAgent:
    """
    Builds and analyzes scientific citation networks
    """

    def __init__(self):
        self.citation_graph = nx.DiGraph()
        self.co_citation_graph = nx.Graph()
        self.coupling_graph = nx.Graph()

        self.nodes: Dict[str, CitationNode] = {}
        self.edges: List[CitationEdge] = []

        # Analysis results
        self.foundational_papers: List[Tuple[str, float]] = []
        self.hidden_gems: List[Tuple[str, float]] = []
        self.communities: Dict[int, List[str]] = {}
        self.anomalies: List[CitationAnomaly] = []

    def build_network_from_papers(self, papers: List[Dict]) -> Dict[str, any]:
        """
        Build citation network from a list of papers

        Args:
            papers: List of paper dicts with keys:
                - paper_id, title, authors, publication_date, references, citations

        Returns:
            Stats about the built network
        """
        print(f"Building citation network from {len(papers)} papers...")

        # Add all papers as nodes
        for paper in papers:
            node = CitationNode(
                paper_id=paper['paper_id'],
                title=paper.get('title', 'Unknown'),
                authors=paper.get('authors', []),
                publication_date=paper.get('publication_date', datetime.now()),
                venue=paper.get('venue'),
                abstract=paper.get('abstract'),
                topics=paper.get('topics', [])
            )
            self.nodes[paper['paper_id']] = node
            self.citation_graph.add_node(paper['paper_id'], data=node)

        # Add citation edges
        edge_count = 0
        for paper in papers:
            paper_id = paper['paper_id']

            # Outgoing citations (papers this paper cites)
            references = paper.get('references', [])
            for ref_id in references:
                if ref_id in self.nodes:
                    edge = CitationEdge(
                        source_id=paper_id,
                        target_id=ref_id,
                        citation_type=CitationType.DIRECT,
                        year_cited=paper.get('publication_date', datetime.now()).year
                    )
                    self.edges.append(edge)
                    self.citation_graph.add_edge(paper_id, ref_id, data=edge)
                    edge_count += 1

                    # Update citation counts
                    self.nodes[paper_id].out_citations += 1
                    self.nodes[ref_id].in_citations += 1

        # Build co-citation and coupling graphs
        self._build_cocitation_graph()
        self._build_coupling_graph()

        stats = {
            'total_papers': len(self.nodes),
            'total_citations': edge_count,
            'avg_citations_per_paper': edge_count / len(self.nodes) if self.nodes else 0,
            'connected_components': nx.number_weakly_connected_components(self.citation_graph),
            'density': nx.density(self.citation_graph),
        }

        print(f"Citation network built: {stats['total_papers']} papers, {stats['total_citations']} citations")
        return stats

    def _build_cocitation_graph(self):
        """Build co-citation network (papers cited together)"""
        paper_ids = list(self.nodes.keys())

        for i, paper1 in enumerate(paper_ids):
            for paper2 in paper_ids[i+1:]:
                # Find papers that cite both paper1 and paper2
                citing_paper1 = set(self.citation_graph.predecessors(paper1))
                citing_paper2 = set(self.citation_graph.predecessors(paper2))
                common_citers = citing_paper1 & citing_paper2

                if len(common_citers) >= 2:  # Threshold
                    weight = len(common_citers)
                    self.co_citation_graph.add_edge(paper1, paper2, weight=weight)

    def _build_coupling_graph(self):
        """Build bibliographic coupling network (papers that cite common references)"""
        paper_ids = list(self.nodes.keys())

        for i, paper1 in enumerate(paper_ids):
            for paper2 in paper_ids[i+1:]:
                # Find common references
                refs1 = set(self.citation_graph.successors(paper1))
                refs2 = set(self.citation_graph.successors(paper2))
                common_refs = refs1 & refs2

                if len(common_refs) >= 2:  # Threshold
                    weight = len(common_refs)
                    self.coupling_graph.add_edge(paper1, paper2, weight=weight)

    def compute_network_metrics(self):
        """Compute all network centrality metrics"""
        print("Computing network metrics...")

        if len(self.citation_graph.nodes()) == 0:
            print("Warning: Empty citation graph")
            return

        # PageRank (measures importance based on citation structure)
        pagerank_scores = nx.pagerank(self.citation_graph, alpha=0.85)
        for paper_id, score in pagerank_scores.items():
            self.nodes[paper_id].pagerank = score

        # Betweenness centrality (measures bridge papers)
        if len(self.citation_graph.nodes()) > 2:
            betweenness = nx.betweenness_centrality(self.citation_graph)
            for paper_id, score in betweenness.items():
                self.nodes[paper_id].betweenness_centrality = score

        # Closeness centrality (measures how central a paper is)
        if nx.is_weakly_connected(self.citation_graph):
            closeness = nx.closeness_centrality(self.citation_graph)
            for paper_id, score in closeness.items():
                self.nodes[paper_id].closeness_centrality = score

        # Eigenvector centrality (measures connection to important papers)
        try:
            eigenvector = nx.eigenvector_centrality(self.citation_graph.to_undirected(), max_iter=1000)
            for paper_id, score in eigenvector.items():
                self.nodes[paper_id].eigenvector_centrality = score
        except:
            print("Warning: Could not compute eigenvector centrality")

        # Citation velocity (citations per month since publication)
        current_date = datetime.now()
        for paper_id, node in self.nodes.items():
            months_since_pub = max(1, (current_date - node.publication_date).days / 30)
            node.citation_velocity = node.in_citations / months_since_pub

        # Categorize papers
        self._categorize_papers()

        print("Network metrics computed successfully")

    def _categorize_papers(self):
        """Categorize papers based on metrics"""
        # Calculate thresholds
        pagerank_values = [n.pagerank for n in self.nodes.values()]
        betweenness_values = [n.betweenness_centrality for n in self.nodes.values()]
        citation_values = [n.in_citations for n in self.nodes.values()]

        if not pagerank_values:
            return

        pr_threshold = np.percentile(pagerank_values, 90)
        bt_threshold = np.percentile(betweenness_values, 90)
        citation_threshold = np.percentile(citation_values, 75)

        for node in self.nodes.values():
            # Recent papers (< 1 year old)
            age_months = (datetime.now() - node.publication_date).days / 30
            if age_months < 12:
                node.category = PaperCategory.RECENT
            # Foundational (high PageRank)
            elif node.pagerank >= pr_threshold:
                node.category = PaperCategory.FOUNDATIONAL
            # Hidden gem (high betweenness, low citations)
            elif node.betweenness_centrality >= bt_threshold and node.in_citations < citation_threshold:
                node.category = PaperCategory.HIDDEN_GEM
            # Bridge (high betweenness)
            elif node.betweenness_centrality >= bt_threshold:
                node.category = PaperCategory.BRIDGE
            # Emerging star (high citation velocity)
            elif node.citation_velocity > np.percentile([n.citation_velocity for n in self.nodes.values()], 85):
                node.category = PaperCategory.EMERGING_STAR
            else:
                node.category = PaperCategory.STANDARD

    def find_foundational_papers(self, top_k: int = 10) -> List[Dict]:
        """
        Identify foundational papers using PageRank

        Returns:
            List of paper dicts with metrics
        """
        # Sort by PageRank
        sorted_papers = sorted(
            self.nodes.values(),
            key=lambda x: x.pagerank,
            reverse=True
        )[:top_k]

        self.foundational_papers = [(p.paper_id, p.pagerank) for p in sorted_papers]

        return [{
            'paper_id': p.paper_id,
            'title': p.title,
            'authors': p.authors,
            'pagerank': p.pagerank,
            'citations': p.in_citations,
            'publication_date': p.publication_date.isoformat(),
            'category': p.category.value
        } for p in sorted_papers]

    def find_hidden_gems(self, top_k: int = 10) -> List[Dict]:
        """
        Find papers with high impact potential but low current citations
        (high betweenness, low citations)

        Returns:
            List of hidden gem papers
        """
        # Filter for hidden gems
        gems = [n for n in self.nodes.values()
                if n.category == PaperCategory.HIDDEN_GEM]

        # Sort by betweenness
        sorted_gems = sorted(
            gems,
            key=lambda x: x.betweenness_centrality,
            reverse=True
        )[:top_k]

        self.hidden_gems = [(p.paper_id, p.betweenness_centrality) for p in sorted_gems]

        return [{
            'paper_id': p.paper_id,
            'title': p.title,
            'authors': p.authors,
            'betweenness': p.betweenness_centrality,
            'citations': p.in_citations,
            'publication_date': p.publication_date.isoformat(),
            'reason': 'High bridge potential, undervalued by current citations'
        } for p in sorted_gems]

    def detect_communities(self, method: str = "louvain") -> Dict[int, List[str]]:
        """
        Detect research communities in citation network

        Args:
            method: "louvain", "label_propagation", or "greedy_modularity"

        Returns:
            Dict mapping community_id to list of paper_ids
        """
        print(f"Detecting communities using {method}...")

        # Convert to undirected for community detection
        G = self.citation_graph.to_undirected()

        if method == "louvain":
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                print("Warning: python-louvain not available, using label propagation")
                method = "label_propagation"

        if method == "label_propagation":
            communities_generator = nx.algorithms.community.label_propagation_communities(G)
            communities_list = list(communities_generator)
            partition = {}
            for idx, community in enumerate(communities_list):
                for node in community:
                    partition[node] = idx

        elif method == "greedy_modularity":
            communities_generator = nx.algorithms.community.greedy_modularity_communities(G)
            communities_list = list(communities_generator)
            partition = {}
            for idx, community in enumerate(communities_list):
                for node in community:
                    partition[node] = idx

        # Update nodes with community assignment
        for paper_id, community_id in partition.items():
            if paper_id in self.nodes:
                self.nodes[paper_id].community_id = community_id

        # Group by community
        self.communities = defaultdict(list)
        for paper_id, community_id in partition.items():
            self.communities[community_id].append(paper_id)

        print(f"Found {len(self.communities)} communities")
        return dict(self.communities)

    def trace_research_lineage(self, root_paper_id: str, max_generations: int = 5) -> ResearchLineage:
        """
        Trace the evolution of research from a root paper

        Args:
            root_paper_id: Starting paper
            max_generations: Maximum generations to trace

        Returns:
            ResearchLineage object
        """
        if root_paper_id not in self.nodes:
            raise ValueError(f"Paper {root_paper_id} not found")

        generations = [[root_paper_id]]
        visited = {root_paper_id}

        # BFS to find descendants
        for gen in range(max_generations):
            current_gen = generations[-1]
            next_gen = []

            for paper_id in current_gen:
                # Find papers that cite this paper
                citing_papers = list(self.citation_graph.predecessors(paper_id))
                for citing_id in citing_papers:
                    if citing_id not in visited:
                        next_gen.append(citing_id)
                        visited.add(citing_id)

            if not next_gen:
                break

            generations.append(next_gen)

        # Find key papers (high PageRank within lineage)
        lineage_papers = [p for gen in generations for p in gen]
        key_papers = sorted(
            lineage_papers,
            key=lambda x: self.nodes[x].pagerank,
            reverse=True
        )[:min(10, len(lineage_papers))]

        # Detect branches (papers with multiple descendants in different directions)
        branches = []
        for paper_id in lineage_papers:
            descendants = list(self.citation_graph.predecessors(paper_id))
            if len(descendants) >= 3:
                # Check if descendants diverge (different communities)
                communities = [self.nodes[d].community_id for d in descendants if d in self.nodes]
                if len(set(communities)) >= 2:
                    branches.append({
                        'branch_point': paper_id,
                        'title': self.nodes[paper_id].title,
                        'descendants': len(descendants),
                        'directions': len(set(communities))
                    })

        return ResearchLineage(
            root_paper_id=root_paper_id,
            generations=generations,
            key_papers=key_papers,
            branches=branches,
            total_descendants=len(visited) - 1
        )

    def predict_citation_impact(self, paper_id: str) -> CitationPrediction:
        """
        Predict future citation impact using features:
        - Current citations
        - Citation velocity
        - PageRank
        - Author h-index (average)
        - Venue prestige
        - Age of paper
        - Network position

        Returns:
            CitationPrediction with 1-year, 3-year, 5-year predictions
        """
        if paper_id not in self.nodes:
            raise ValueError(f"Paper {paper_id} not found")

        node = self.nodes[paper_id]

        # Feature extraction
        age_months = (datetime.now() - node.publication_date).days / 30

        features = {
            'current_citations': node.in_citations,
            'citation_velocity': node.citation_velocity,
            'pagerank': node.pagerank,
            'betweenness': node.betweenness_centrality,
            'age_months': age_months,
            'num_references': node.out_citations,
        }

        # Simple prediction model (linear extrapolation with decay)
        # In production, use trained ML model

        # Citation velocity decays over time
        decay_factor = 0.8

        # 1-year prediction
        pred_1year = int(node.in_citations + node.citation_velocity * 12)

        # 3-year prediction (with decay)
        pred_3year = int(node.in_citations +
                        node.citation_velocity * 12 +
                        node.citation_velocity * 12 * decay_factor +
                        node.citation_velocity * 12 * (decay_factor ** 2))

        # 5-year prediction
        pred_5year = int(node.in_citations +
                        sum(node.citation_velocity * 12 * (decay_factor ** i)
                            for i in range(5)))

        # Adjust based on PageRank (high PageRank = sustained citations)
        pagerank_multiplier = 1.0 + node.pagerank * 10
        pred_1year = int(pred_1year * pagerank_multiplier)
        pred_3year = int(pred_3year * pagerank_multiplier)
        pred_5year = int(pred_5year * pagerank_multiplier)

        # Confidence based on age (more confident for older papers)
        confidence = min(0.95, age_months / 24) if age_months > 0 else 0.3

        # Find comparable papers (same community, similar metrics)
        comparable_papers = []
        if node.community_id is not None:
            community_papers = self.communities.get(node.community_id, [])
            for comp_id in community_papers[:5]:
                if comp_id != paper_id and comp_id in self.nodes:
                    comparable_papers.append(comp_id)

        # Feature importance
        importance = {
            'citation_velocity': 0.35,
            'current_citations': 0.25,
            'pagerank': 0.20,
            'network_position': 0.15,
            'age': 0.05
        }

        return CitationPrediction(
            paper_id=paper_id,
            predicted_citations_1year=pred_1year,
            predicted_citations_3year=pred_3year,
            predicted_citations_5year=pred_5year,
            confidence_score=confidence,
            factors=importance,
            comparable_papers=comparable_papers
        )

    def detect_citation_anomalies(self) -> List[CitationAnomaly]:
        """
        Detect unusual citation patterns:
        - Citation bursts (sudden spike)
        - Delayed recognition (old paper suddenly cited)
        - Self-citation clusters
        - Citation cartels

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Citation burst detection
        for node in self.nodes.values():
            # If citation velocity is very high compared to age
            age_years = (datetime.now() - node.publication_date).days / 365
            if age_years > 2 and node.citation_velocity > 5:
                # Check if this is unusual (> 2 std dev from mean)
                avg_velocity = np.mean([n.citation_velocity for n in self.nodes.values()])
                std_velocity = np.std([n.citation_velocity for n in self.nodes.values()])

                if node.citation_velocity > avg_velocity + 2 * std_velocity:
                    anomalies.append(CitationAnomaly(
                        paper_id=node.paper_id,
                        anomaly_type="citation_burst",
                        severity=0.8,
                        description=f"Unusual citation spike for paper published {age_years:.1f} years ago",
                        evidence={
                            'citation_velocity': node.citation_velocity,
                            'expected_velocity': avg_velocity,
                            'age_years': age_years
                        }
                    ))

        # Delayed recognition detection
        for node in self.nodes.values():
            age_years = (datetime.now() - node.publication_date).days / 365
            # Old paper with recent high citations
            if age_years > 5 and node.citation_velocity > 2:
                anomalies.append(CitationAnomaly(
                    paper_id=node.paper_id,
                    anomaly_type="delayed_recognition",
                    severity=0.7,
                    description=f"Paper from {node.publication_date.year} experiencing renewed interest",
                    evidence={
                        'age_years': age_years,
                        'citation_velocity': node.citation_velocity
                    }
                ))

        self.anomalies = anomalies
        return anomalies

    def get_influential_authors(self, top_k: int = 20) -> List[Dict]:
        """
        Rank authors by their influence in the network

        Returns:
            List of author dicts with metrics
        """
        author_metrics = defaultdict(lambda: {
            'papers': [],
            'total_citations': 0,
            'avg_pagerank': 0,
            'h_index': 0
        })

        # Aggregate metrics per author
        for node in self.nodes.values():
            for author in node.authors:
                author_metrics[author]['papers'].append(node.paper_id)
                author_metrics[author]['total_citations'] += node.in_citations

        # Calculate averages and h-index
        for author, metrics in author_metrics.items():
            num_papers = len(metrics['papers'])

            # Average PageRank
            pageranks = [self.nodes[pid].pagerank for pid in metrics['papers']]
            metrics['avg_pagerank'] = np.mean(pageranks)

            # H-index: author has h papers with at least h citations each
            citations = sorted([self.nodes[pid].in_citations for pid in metrics['papers']], reverse=True)
            h = 0
            for i, cites in enumerate(citations, 1):
                if cites >= i:
                    h = i
                else:
                    break
            metrics['h_index'] = h
            metrics['num_papers'] = num_papers

        # Sort by composite score
        author_scores = []
        for author, metrics in author_metrics.items():
            score = (metrics['avg_pagerank'] * 100 +
                    metrics['h_index'] * 10 +
                    metrics['total_citations'] * 0.1)
            author_scores.append((author, score, metrics))

        author_scores.sort(key=lambda x: x[1], reverse=True)

        return [{
            'author': author,
            'score': score,
            'papers': metrics['num_papers'],
            'citations': metrics['total_citations'],
            'h_index': metrics['h_index'],
            'avg_pagerank': metrics['avg_pagerank']
        } for author, score, metrics in author_scores[:top_k]]

    def export_network(self, output_path: str, format: str = "graphml"):
        """
        Export citation network for visualization

        Args:
            output_path: Path to output file
            format: "graphml", "gexf", or "json"
        """
        if format == "graphml":
            # Add node attributes
            for paper_id in self.citation_graph.nodes():
                node = self.nodes[paper_id]
                self.citation_graph.nodes[paper_id]['title'] = node.title
                self.citation_graph.nodes[paper_id]['pagerank'] = node.pagerank
                self.citation_graph.nodes[paper_id]['category'] = node.category.value
                self.citation_graph.nodes[paper_id]['citations'] = node.in_citations

            nx.write_graphml(self.citation_graph, output_path)

        elif format == "gexf":
            nx.write_gexf(self.citation_graph, output_path)

        elif format == "json":
            # Export as JSON for web visualization
            data = {
                'nodes': [
                    {
                        'id': node.paper_id,
                        'label': node.title,
                        'size': node.pagerank * 100,
                        'category': node.category.value,
                        'citations': node.in_citations,
                        'community': node.community_id
                    }
                    for node in self.nodes.values()
                ],
                'edges': [
                    {
                        'source': edge.source_id,
                        'target': edge.target_id,
                        'type': edge.citation_type.value
                    }
                    for edge in self.edges
                ]
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Network exported to {output_path} ({format} format)")

    def get_network_summary(self) -> Dict:
        """Get comprehensive network statistics"""
        return {
            'total_papers': len(self.nodes),
            'total_citations': len(self.edges),
            'avg_citations_per_paper': np.mean([n.in_citations for n in self.nodes.values()]),
            'median_citations': np.median([n.in_citations for n in self.nodes.values()]),
            'max_citations': max([n.in_citations for n in self.nodes.values()]) if self.nodes else 0,
            'density': nx.density(self.citation_graph),
            'num_communities': len(self.communities),
            'foundational_papers': len([n for n in self.nodes.values() if n.category == PaperCategory.FOUNDATIONAL]),
            'hidden_gems': len([n for n in self.nodes.values() if n.category == PaperCategory.HIDDEN_GEM]),
            'emerging_stars': len([n for n in self.nodes.values() if n.category == PaperCategory.EMERGING_STAR]),
            'anomalies_detected': len(self.anomalies)
        }


if __name__ == "__main__":
    # Example usage
    print("Citation Network Analysis Agent - Example")

    # Sample papers
    papers = [
        {
            'paper_id': 'paper1',
            'title': 'Attention Is All You Need',
            'authors': ['Vaswani', 'Shazeer'],
            'publication_date': datetime(2017, 6, 1),
            'venue': 'NeurIPS',
            'references': []
        },
        {
            'paper_id': 'paper2',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'authors': ['Devlin', 'Chang'],
            'publication_date': datetime(2018, 10, 1),
            'venue': 'NAACL',
            'references': ['paper1']
        },
        {
            'paper_id': 'paper3',
            'title': 'GPT-3: Language Models are Few-Shot Learners',
            'authors': ['Brown', 'Mann'],
            'publication_date': datetime(2020, 5, 1),
            'venue': 'NeurIPS',
            'references': ['paper1', 'paper2']
        }
    ]

    agent = CitationNetworkAgent()

    # Build network
    stats = agent.build_network_from_papers(papers)
    print("\nNetwork Stats:", stats)

    # Compute metrics
    agent.compute_network_metrics()

    # Find foundational papers
    foundational = agent.find_foundational_papers(top_k=3)
    print("\nFoundational Papers:")
    for p in foundational:
        print(f"  - {p['title']}: PageRank={p['pagerank']:.4f}")

    # Get network summary
    summary = agent.get_network_summary()
    print("\nNetwork Summary:", summary)
