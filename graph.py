#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScholarGenie — Neo4j Knowledge Graph
=====================================
Optional Neo4j integration for persistent graph storage.
Falls back gracefully to the in-memory builder if Neo4j is not configured.

Setup (one-time):
  1. Download Neo4j Desktop: https://neo4j.com/download/
  2. Create a new Local DBMS, set any password, then Start it.
  3. Add these lines to your .env:
       NEO4J_URI=bolt://localhost:7687
       NEO4J_USER=neo4j
       NEO4J_PASSWORD=your_password_here
  4. pip install neo4j

Node types:
  (:Paper  {id, title, year, source, citations, url})
  (:Concept {name})

Relationships:
  (:Paper)-[:HAS_CONCEPT]->(:Concept)
  (:Paper)-[:RELATED_TO {weight}]-(:Paper)   # weight = shared concept count
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── Driver availability ────────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase
    NEO4J_DRIVER_AVAILABLE = True
except ImportError:
    NEO4J_DRIVER_AVAILABLE = False

# ── Concept extraction ─────────────────────────────────────────────────────────
_STOP = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "we", "our",
    "their", "its", "it", "as", "but", "not", "also", "can", "using",
    "based", "via", "paper", "study", "show", "results", "method", "approach",
    "propose", "present", "demonstrate", "evaluate", "analysis", "work",
    "new", "novel", "effective", "efficient", "large", "small", "high", "low",
    "two", "three", "well", "which", "both", "each", "such", "more", "most",
    "than", "then", "when", "where", "here", "there", "all", "any", "one",
    "used", "use", "used", "has", "have", "show", "shows", "shown",
}

def _extract_concepts(paper: dict, max_n: int = 8) -> List[str]:
    """Extract key technical concepts from title + abstract."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '') or ''}".lower()
    text = re.sub(r"[^\w\s]", " ", text)
    seen: set = set()
    result: List[str] = []
    for w in text.split():
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= max_n:
                break
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j Graph class
# ══════════════════════════════════════════════════════════════════════════════

class Neo4jGraph:
    """
    Wraps a Neo4j Bolt connection. Exposes build_graph() and get_graph()
    that return the same dict shape as the in-memory knowledge-graph builder,
    so the existing frontend works without changes.
    """

    def __init__(self):
        self._driver = None
        self._available = False

        if not NEO4J_DRIVER_AVAILABLE:
            return

        uri      = os.getenv("NEO4J_URI", "").strip()
        user     = os.getenv("NEO4J_USER", "neo4j").strip()
        password = os.getenv("NEO4J_PASSWORD", "").strip()

        if not uri or not password:
            return

        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connection
            with self._driver.session() as s:
                s.run("RETURN 1")
            self._available = True
            self._init_schema()
            print(f"[OK] Neo4j connected: {uri}")
        except Exception as e:
            print(f"[!] Neo4j connection failed: {e}")
            self._driver = None
            self._available = False

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _init_schema(self):
        with self._driver.session() as s:
            try:
                s.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
                s.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            except Exception:
                pass  # constraints may already exist

    # ── Build ───────────────────────────────────────────────────────────────────

    def build_graph(self, papers: List[dict]) -> Dict:
        """
        Store papers + concept relationships. Clears previous graph first.
        Returns a graph dict (same shape as in-memory builder).
        """
        if not self._available:
            return {"error": "Neo4j not available", "neo4j": False}

        with self._driver.session() as s:
            # Wipe existing graph
            s.run("MATCH (n) DETACH DELETE n")

            for paper in papers:
                concepts = _extract_concepts(paper)

                # Create/update Paper node
                s.run(
                    """
                    MERGE (p:Paper {id: $id})
                    SET p.title     = $title,
                        p.year      = $year,
                        p.source    = $source,
                        p.citations = $citations,
                        p.url       = $url
                    """,
                    id=paper.get("id", ""),
                    title=(paper.get("title") or "")[:200],
                    year=paper.get("year"),
                    source=paper.get("source", ""),
                    citations=paper.get("citations", 0),
                    url=paper.get("url", ""),
                )

                # Create Concept nodes and HAS_CONCEPT edges
                for concept in concepts:
                    s.run(
                        """
                        MERGE (c:Concept {name: $concept})
                        WITH c
                        MATCH (p:Paper {id: $pid})
                        MERGE (p)-[:HAS_CONCEPT]->(c)
                        """,
                        concept=concept,
                        pid=paper.get("id", ""),
                    )

            # Build RELATED_TO edges based on shared concepts
            s.run(
                """
                MATCH (p1:Paper)-[:HAS_CONCEPT]->(c:Concept)<-[:HAS_CONCEPT]-(p2:Paper)
                WHERE id(p1) < id(p2)
                MERGE (p1)-[r:RELATED_TO]-(p2)
                ON CREATE SET r.weight = 1
                ON MATCH  SET r.weight = r.weight + 1
                """
            )

        return self.get_graph()

    # ── Query ───────────────────────────────────────────────────────────────────

    def get_graph(self) -> Dict:
        """Return current graph data in the standard ScholarGenie dict format."""
        if not self._available:
            return {"nodes": [], "edges": [], "shared_concepts": [], "neo4j": False}

        with self._driver.session() as s:
            # Paper nodes with concepts
            paper_rows = s.run(
                """
                MATCH (p:Paper)
                OPTIONAL MATCH (p)-[:HAS_CONCEPT]->(c:Concept)
                WITH p, collect(c.name) AS concepts
                RETURN p.id         AS id,
                       p.title      AS title,
                       p.year       AS year,
                       p.source     AS source,
                       p.citations  AS citations,
                       p.url        AS url,
                       concepts
                ORDER BY p.citations DESC
                """
            ).data()

            # Edges (only show meaningful connections)
            edge_rows = s.run(
                """
                MATCH (p1:Paper)-[r:RELATED_TO]-(p2:Paper)
                WHERE id(p1) < id(p2) AND r.weight >= 2
                RETURN p1.id AS source, p2.id AS target, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT 300
                """
            ).data()

            # Top shared concepts
            concept_rows = s.run(
                """
                MATCH (c:Concept)<-[:HAS_CONCEPT]-(p:Paper)
                WITH c.name AS concept, collect(p.title) AS paper_titles, count(p) AS cnt
                WHERE cnt > 1
                RETURN concept, paper_titles, cnt
                ORDER BY cnt DESC
                LIMIT 20
                """
            ).data()

            # Stats
            stats = s.run(
                """
                MATCH (p:Paper) WITH count(p) AS papers
                OPTIONAL MATCH ()-[r:RELATED_TO]-()
                WITH papers, count(r)/2 AS connections
                OPTIONAL MATCH (c:Concept)<-[:HAS_CONCEPT]-()
                WITH papers, connections, count(DISTINCT c) AS concepts
                RETURN papers, connections, concepts
                """
            ).data()

        nodes = [
            {
                "id":        row["id"],
                "label":     (row["title"] or "")[:60],
                "type":      "paper",
                "year":      row.get("year"),
                "concepts":  row.get("concepts", []),
                "source":    row.get("source", ""),
                "citations": row.get("citations", 0),
            }
            for row in paper_rows
        ]

        edges = [
            {
                "source": row["source"],
                "target": row["target"],
                "label":  f"{row['weight']} shared concepts",
            }
            for row in edge_rows
        ]

        shared_concepts = [
            {
                "concept": row["concept"],
                "papers":  row.get("paper_titles", [])[:5],
                "count":   row["cnt"],
            }
            for row in concept_rows
        ]

        st = stats[0] if stats else {}
        return {
            "nodes":           nodes,
            "edges":           edges,
            "shared_concepts": shared_concepts,
            "neo4j":           True,
            "stats": {
                "papers":      st.get("papers", len(nodes)),
                "connections": st.get("connections", len(edges)),
                "concepts":    st.get("concepts", len(shared_concepts)),
            },
        }

    def node_count(self) -> int:
        if not self._available:
            return 0
        with self._driver.session() as s:
            r = s.run("MATCH (p:Paper) RETURN count(p) AS n").single()
            return r["n"] if r else 0

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            self._available = False


# ── Singleton ──────────────────────────────────────────────────────────────────
_instance: Optional[Neo4jGraph] = None


def get_neo4j() -> Neo4jGraph:
    global _instance
    if _instance is None:
        _instance = Neo4jGraph()
    return _instance


def neo4j_available() -> bool:
    return get_neo4j().available
