#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScholarGenie — CrewAI Agent Pipeline
=====================================
4 specialized agents working sequentially to automate the full research workflow:

  1. ResearchAgent     — finds papers from arXiv + Semantic Scholar
  2. AnalysisAgent     — generates AI summaries for each paper
  3. GapFinderAgent    — identifies research gaps using hybrid AI + rule-based method
  4. PresentationAgent — generates a PPTX presentation from findings

Runs alongside the existing api.py — existing endpoints are untouched.
Triggered via POST /api/pipeline from the frontend.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ── Check CrewAI availability ──────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import tool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("! CrewAI not installed. Run: pip install crewai")


# ── Build Groq LLM for CrewAI ─────────────────────────────────────────────────
def _build_llm() -> Optional[object]:
    """
    Returns a crewai.LLM instance backed by Groq (free).
    Uses crewai.LLM with groq/ model prefix — no OPENAI_API_KEY required.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_groq_api_key_here":
        return None
    try:
        return LLM(
            model="groq/llama-3.1-8b-instant",
            api_key=key,
            temperature=0.3,
        )
    except Exception as e:
        print(f"! LLM build error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS  (each wraps an existing api.py function)
# ══════════════════════════════════════════════════════════════════════════════

@tool("Search Research Papers")
def search_papers_tool(query: str) -> str:
    """
    Search arXiv and Semantic Scholar for academic research papers on a topic.
    Automatically saves all found papers to the library.
    Input: research query string (e.g. 'semantic search transformer models').
    """
    try:
        from api import find_papers, db
        papers = find_papers(query, 10)
        if not papers:
            return "No papers found. Try a different query."
        for p in papers:
            db.save_paper(p)
        lines = [f"Found and saved {len(papers)} papers:"]
        for p in papers:
            lines.append(
                f"  [{p.get('year', 'N/A')}] {p['title']} "
                f"| Source: {p.get('source', '?')} "
                f"| Citations: {p.get('citations', 0)}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


@tool("Summarize Saved Papers")
def summarize_papers_tool(instruction: str) -> str:
    """
    Generate AI-powered summaries for all papers currently saved in the library.
    Extracts TL;DR and key contributions from each paper.
    Input: any string (tool operates on the saved library, input is ignored).
    """
    try:
        from api import db, summarize_paper
        papers = db.get_papers()[:8]
        if not papers:
            return "No papers in library. Run search first."
        results = []
        for p in papers:
            s = summarize_paper(p)
            db.save_summary(p["id"], s)
            kp = " | ".join(s.get("keypoints", [])[:2])
            results.append(
                f"[{p['title'][:55]}]\n"
                f"  TLDR: {s.get('tldr', 'N/A')}\n"
                f"  Key Points: {kp or 'See full summary'}"
            )
        return f"Summarized {len(results)} papers:\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Summarization error: {e}"


@tool("Discover Research Gaps")
def discover_gaps_tool(instruction: str) -> str:
    """
    Identify research gaps and underexplored areas from the saved paper collection.
    Uses rule-based detectors to find: Underexplored Areas, Temporal Gaps,
    Methodological Gaps, and Cross-Domain Opportunities.
    Input: any string (tool operates on the saved library, input is ignored).
    """
    try:
        from api import db, discover_gaps
        papers = db.get_papers()
        if not papers:
            return "No papers in library. Run search first."
        gaps = discover_gaps(papers)
        if not gaps:
            return "No significant gaps identified from current paper collection."
        lines = [f"Identified {len(gaps)} research gaps:"]
        for g in gaps:
            lines.append(
                f"  [{g.get('type', 'Gap')}] {g.get('title', 'Untitled')}\n"
                f"    {g.get('description', '')[:150]}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"Gap discovery error: {e}"


@tool("Generate Research Presentation")
def generate_presentation_tool(topic: str) -> str:
    """
    Generate a 14-slide PPTX research presentation from the saved papers.
    Input: presentation topic or title string.
    """
    try:
        from api import db, summarize_paper, generate_pptx
        papers = db.get_papers()
        if not papers:
            return "No papers in library. Run search first."
        # Use the top paper for the main presentation
        paper = papers[0]
        summary = db.get_summary(paper["id"]) or summarize_paper(paper)
        path = generate_pptx(paper, summary)
        if path:
            return (
                f"Presentation generated successfully!\n"
                f"  File: {Path(path).name}\n"
                f"  Paper: {paper['title'][:60]}\n"
                f"  Slides: 14\n"
                f"  Download: /api/present/download/{paper['id']}"
            )
        return "Presentation generation failed. Check paper data."
    except Exception as e:
        return f"Presentation error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def _build_agents(llm):
    """Construct and return the 4 ScholarGenie CrewAI agents."""

    research_agent = Agent(
        role="Senior Research Discovery Specialist",
        goal=(
            "Find relevant academic papers on the given topic using ONLY the provided "
            "Search Research Papers tool. Never attempt to use web search tools."
        ),
        backstory=(
            "You are an expert research librarian. You call the Search Research Papers "
            "tool exactly once, accept whatever papers it returns, and report them. "
            "You never attempt to validate results with external search engines."
        ),
        tools=[search_papers_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    analysis_agent = Agent(
        role="Academic Research Analyst",
        goal=(
            "Generate AI summaries for all papers in the library using the provided tool."
        ),
        backstory=(
            "You call the Summarize Saved Papers tool once and report the results. "
            "You do not attempt any additional processing beyond what the tool returns."
        ),
        tools=[summarize_papers_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    gap_finder_agent = Agent(
        role="Research Gap Identification Expert",
        goal=(
            "Identify research gaps from the paper collection using the provided tool."
        ),
        backstory=(
            "You call the Discover Research Gaps tool once and report all gaps found. "
            "You do not attempt any additional analysis beyond what the tool returns."
        ),
        tools=[discover_gaps_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    presentation_agent = Agent(
        role="Academic Presentation Designer",
        goal=(
            "Generate a PPTX presentation from the saved papers using the provided tool."
        ),
        backstory=(
            "You call the Generate Research Presentation tool once with the topic "
            "and report the download link. No additional steps needed."
        ),
        tools=[generate_presentation_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    return research_agent, analysis_agent, gap_finder_agent, presentation_agent


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str) -> dict:
    """
    Run the full ScholarGenie agent pipeline for a given research query.

    Sequence:
      ResearchAgent → AnalysisAgent → GapFinderAgent → PresentationAgent

    Returns a dict with status, counts, and agent output.
    """
    if not CREWAI_AVAILABLE:
        return {
            "status": "error",
            "message": "CrewAI is not installed. Run: pip install crewai",
        }

    llm = _build_llm()
    if llm is None:
        return {
            "status": "error",
            "message": (
                "GROQ_API_KEY is not set. "
                "Open your .env file and add: GROQ_API_KEY=your_actual_key_here"
            ),
        }

    research_agent, analysis_agent, gap_finder_agent, presentation_agent = (
        _build_agents(llm)
    )

    # ── Task 1: Paper Discovery ───────────────────────────────────────────────
    research_task = Task(
        description=(
            f"Search for 10 academic research papers on the topic: '{query}'.\n"
            "Use the search tool to find papers from arXiv and Semantic Scholar.\n"
            "All papers will be automatically saved to the library.\n"
            "Report how many papers were found and provide a brief list."
        ),
        expected_output=(
            "A numbered list of 10 papers with titles, publication years, and sources."
        ),
        agent=research_agent,
    )

    # ── Task 2: Summarization ─────────────────────────────────────────────────
    analysis_task = Task(
        description=(
            "Summarize all papers currently saved in the library.\n"
            "Use the summarize tool to generate AI summaries for each paper.\n"
            "Extract TL;DR and key contributions. Report the summary results."
        ),
        expected_output=(
            "AI-generated TL;DR and key points for each paper in the library."
        ),
        agent=analysis_agent,
        context=[research_task],
    )

    # ── Task 3: Gap Discovery ─────────────────────────────────────────────────
    gap_task = Task(
        description=(
            "Analyze the saved papers and identify research gaps.\n"
            "Use the gap discovery tool to find underexplored areas, temporal gaps,\n"
            "methodological gaps, and cross-domain opportunities.\n"
            "Report all gaps found with their types and descriptions."
        ),
        expected_output=(
            "A categorized list of research gaps with types, titles, and descriptions."
        ),
        agent=gap_finder_agent,
        context=[research_task, analysis_task],
    )

    # ── Task 4: Presentation ──────────────────────────────────────────────────
    presentation_task = Task(
        description=(
            f"Generate a professional research presentation on the topic: '{query}'.\n"
            "Use the presentation tool to create a 14-slide PPTX file.\n"
            "Report the filename and download link."
        ),
        expected_output=(
            "Confirmation of PPTX file generated with filename and download link."
        ),
        agent=presentation_agent,
        context=[research_task, analysis_task, gap_task],
    )

    # ── Assemble and run Crew ─────────────────────────────────────────────────
    crew = Crew(
        agents=[research_agent, analysis_agent, gap_finder_agent, presentation_agent],
        tasks=[research_task, analysis_task, gap_task, presentation_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        agent_output = str(result)
    except Exception as e:
        agent_output = f"Crew error: {e}"

    # ── Collect rich data from DB for frontend display ────────────────────────
    try:
        from api import db, discover_gaps
        papers = db.get_papers()
        gaps = discover_gaps(papers) if papers else []
        paper_count = len(papers)
        gap_count = len(gaps)

        # Trimmed papers list for frontend cards
        papers_data = [
            {
                "id": p.get("id", ""),
                "title": p.get("title", ""),
                "year": p.get("year", "N/A"),
                "source": p.get("source", ""),
                "citations": p.get("citations", 0),
                "url": p.get("url", ""),
                "authors": (p.get("authors") or [])[:3],
            }
            for p in papers[:10]
        ]

        # TLDR + keypoints per paper (for inline display)
        summaries_data: dict = {}
        for p in papers[:10]:
            s = db.get_summary(p["id"])
            if s:
                summaries_data[p["id"]] = {
                    "tldr": s.get("tldr", ""),
                    "keypoints": (s.get("keypoints") or [])[:3],
                }

        # Gaps list for frontend cards
        gaps_data = [
            {
                "type": g.get("type", "Gap"),
                "title": g.get("title", ""),
                "description": g.get("description", ""),
                "confidence": g.get("confidence", 0.5),
            }
            for g in gaps[:10]
        ]

        # Always generate the PPTX ourselves — don't rely on the agent task
        presentation_id = None
        if papers:
            try:
                from api import summarize_paper, generate_pptx
                paper0 = papers[0]
                summary0 = db.get_summary(paper0["id"]) or summarize_paper(paper0)
                pptx_path = generate_pptx(paper0, summary0)
                if pptx_path:
                    presentation_id = paper0["id"]
            except Exception as pptx_err:
                print(f"! PPTX generation in pipeline failed: {pptx_err}")

    except Exception:
        paper_count = 0
        gap_count = 0
        papers_data = []
        summaries_data = {}
        gaps_data = []
        presentation_id = None

    return {
        "status": "success",
        "query": query,
        "papers_found": paper_count,
        "gaps_identified": gap_count,
        "agent_output": agent_output,
        "papers": papers_data,
        "summaries": summaries_data,
        "gaps": gaps_data,
        "presentation_id": presentation_id,
        "message": (
            f"Pipeline complete. {paper_count} papers analyzed, "
            f"{gap_count} research gaps identified."
        ),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "transformer models NLP"
    print(f"\n ScholarGenie Agent Pipeline\n Query: {q}\n{'='*50}")
    out = run_pipeline(q)
    print(f"\n Status : {out['status']}")
    print(f" Message: {out.get('message', '')}")
    if out.get("agent_output"):
        print(f"\n Agent Output:\n{out['agent_output'][:800]}")
