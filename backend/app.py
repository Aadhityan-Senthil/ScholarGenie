"""FastAPI backend service for ScholarGenie."""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

from backend.agents.paper_finder import PaperFinderAgent
from backend.agents.pdf_parser import PDFParserAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.extractor import ExtractorAgent
from backend.agents.presenter import PresenterAgent
from backend.agents.evaluator import EvaluatorAgent
from backend.agents.knowledge_graph import KnowledgeGraphAgent
from backend.agents.gap_discovery import GapDiscoveryAgent
from backend.agents.graph_rag import GraphRAG
from backend.agents.link_prediction import LinkPredictor
from backend.agents.llm_reasoner import LLMReasoner
from backend.agents.causal_reasoning import CausalGraphReasoner
from backend.agents.hypothesis_tree import HypothesisTreeGenerator
from backend.agents.gap_reporter import GapReportGenerator
from backend.agents.citation_network import CitationNetworkAgent
from backend.agents.research_monitor import ResearchMonitor, AlertConfig, AlertSource, AlertPriority
from backend.agents.workspace_manager import WorkspaceManager, WorkspaceRole, TaskStatus, TaskPriority
from backend.agents.lit_review_generator import LiteratureReviewGenerator, ReviewStyle, CitationStyle
from backend.agents.grant_matcher import GrantMatcher, GrantAgency
from backend.agents.domain_transfer import DomainTransferAgent, Domain
from backend.crews.research_crew import ResearchCrew
from backend.crews.analysis_crew import AnalysisCrew
from backend.crews.discovery_crew import DiscoveryCrew
from backend.utils.storage import VectorStore
from backend.utils.embeddings import EmbeddingService
from backend.utils.metadata import PaperMetadata, Summary, ExtractedData

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ScholarGenie API",
    description="Multi-agent system for scientific paper analysis",
    version="1.0.0"
)

# Include authentication routes
from backend.auth.routes import router as auth_router
app.include_router(auth_router)

# Add security and rate limiting middleware
from backend.middleware.security import SecurityMiddleware, RequestValidationMiddleware
from backend.middleware.rate_limit import RateLimitMiddleware

app.add_middleware(SecurityMiddleware)
app.add_middleware(RequestValidationMiddleware, max_content_length=100 * 1024 * 1024)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, requests_per_hour=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents and services
logger.info("Initializing agents and services...")
paper_finder = PaperFinderAgent()
pdf_parser = PDFParserAgent()
summarizer = SummarizerAgent()
extractor = ExtractorAgent()
presenter = PresenterAgent()
evaluator = EvaluatorAgent()
vector_store = VectorStore()
embedding_service = EmbeddingService()

# Initialize advanced features
knowledge_graph = KnowledgeGraphAgent()
gap_discovery = GapDiscoveryAgent(knowledge_graph, embedding_service)
graph_rag = GraphRAG(knowledge_graph, vector_store, embedding_service)
link_predictor = LinkPredictor(knowledge_graph)
llm_reasoner = LLMReasoner(knowledge_graph)
causal_reasoner = CausalGraphReasoner(knowledge_graph)
hypothesis_generator = HypothesisTreeGenerator(knowledge_graph, causal_reasoner)
gap_reporter = GapReportGenerator(knowledge_graph, gap_discovery, llm_reasoner, causal_reasoner)
citation_network = CitationNetworkAgent()
research_monitor = ResearchMonitor(embedding_service=embedding_service, summarizer=summarizer)
workspace_manager = WorkspaceManager()
lit_review_generator = LiteratureReviewGenerator()
grant_matcher = GrantMatcher()
domain_transfer = DomainTransferAgent()

# Initialize CrewAI multi-agent crews
research_crew = ResearchCrew(verbose=False)
analysis_crew = AnalysisCrew(verbose=False)
discovery_crew = DiscoveryCrew(kg_agent=knowledge_graph, verbose=False)

logger.info("Advanced features and CrewAI crews initialized")

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# In-memory cache for processed papers (use Redis in production)
paper_cache: Dict[str, PaperMetadata] = {}
summary_cache: Dict[str, Summary] = {}
extracted_cache: Dict[str, ExtractedData] = {}


# =====================================================
# Health & Status Endpoints
# =====================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns system status and service availability.
    """
    from backend.utils.cache import get_cache
    from backend.database.session import engine

    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check database connection
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        status["services"]["database"] = "connected"
    except Exception as e:
        status["services"]["database"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Check Redis connection
    try:
        cache = get_cache()
        if cache.enabled:
            status["services"]["redis"] = "connected"
        else:
            status["services"]["redis"] = "disconnected"
            status["status"] = "degraded"
    except Exception as e:
        status["services"]["redis"] = f"error: {str(e)}"
        status["status"] = "degraded"

    # Add agent status
    status["services"]["agents"] = {
        "paper_finder": "initialized",
        "summarizer": "initialized",
        "knowledge_graph": "initialized",
        "lit_review_generator": "initialized",
        "grant_matcher": "initialized",
        "domain_transfer": "initialized"
    }

    return status


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "ScholarGenie API",
        "version": "1.0.0",
        "description": "Multi-agent system for scientific paper analysis",
        "documentation": "/docs",
        "health": "/health"
    }


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    sources: Optional[List[str]] = None


class SearchResult(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: Optional[str]
    year: Optional[int]
    venue: Optional[str]
    pdf_url: Optional[str]
    is_open_access: bool
    source: str


class IngestRequest(BaseModel):
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pdf_url: Optional[str] = None
    paper_id: Optional[str] = None


class IngestResponse(BaseModel):
    paper_id: str
    title: str
    status: str
    message: str


class SummarizeRequest(BaseModel):
    paper_id: str


class SummarizeResponse(BaseModel):
    paper_id: str
    summary: Dict[str, Any]


class GeneratePPTXRequest(BaseModel):
    paper_id: str


class GenerateReportRequest(BaseModel):
    paper_id: str
    format: str = "markdown"


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Paper discovery endpoints
@app.post("/api/search", response_model=List[SearchResult])
async def search_papers(request: SearchRequest):
    """Search for papers across multiple sources.

    Args:
        request: Search parameters

    Returns:
        List of paper results
    """
    try:
        logger.info(f"Searching for: {request.query}")

        papers = paper_finder.search(
            query=request.query,
            max_results=request.max_results,
            sources=request.sources
        )

        # Convert to response format
        results = []
        for paper in papers:
            author_names = [a["name"] if isinstance(a, dict) else a for a in paper.get("authors", [])]

            results.append(SearchResult(
                paper_id=paper["paper_id"],
                title=paper["title"],
                authors=author_names,
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                venue=paper.get("venue"),
                pdf_url=paper.get("pdf_url"),
                is_open_access=paper.get("is_open_access", False),
                source=paper.get("source", "unknown")
            ))

        return results

    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_paper(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a paper by DOI, arXiv ID, or PDF URL.

    Args:
        request: Ingest parameters
        background_tasks: FastAPI background tasks

    Returns:
        Ingest status
    """
    try:
        paper_metadata = None

        # Get paper by DOI
        if request.doi:
            logger.info(f"Ingesting paper by DOI: {request.doi}")
            paper_data = paper_finder.get_paper_by_doi(request.doi)

            if not paper_data:
                raise HTTPException(status_code=404, detail=f"Paper not found for DOI: {request.doi}")

            if not paper_data.get("pdf_url"):
                raise HTTPException(status_code=404, detail=f"No PDF available for DOI: {request.doi}")

            # Download and parse PDF
            paper_metadata = pdf_parser.download_and_parse(
                pdf_url=paper_data["pdf_url"],
                paper_id=paper_data["paper_id"]
            )

        # Get paper by arXiv ID
        elif request.arxiv_id:
            logger.info(f"Ingesting paper by arXiv ID: {request.arxiv_id}")
            paper_data = paper_finder.get_paper_by_arxiv_id(request.arxiv_id)

            if not paper_data:
                raise HTTPException(status_code=404, detail=f"Paper not found for arXiv ID: {request.arxiv_id}")

            paper_metadata = pdf_parser.download_and_parse(
                pdf_url=paper_data["pdf_url"],
                paper_id=paper_data["paper_id"]
            )

        # Direct PDF URL
        elif request.pdf_url:
            logger.info(f"Ingesting paper from PDF URL: {request.pdf_url}")
            paper_metadata = pdf_parser.download_and_parse(
                pdf_url=request.pdf_url,
                paper_id=request.paper_id
            )

        else:
            raise HTTPException(status_code=400, detail="Must provide doi, arxiv_id, or pdf_url")

        # Store in cache
        paper_cache[paper_metadata.paper_id] = paper_metadata

        # Generate embeddings and store in vector DB (background task)
        background_tasks.add_task(
            _index_paper,
            paper_metadata
        )

        return IngestResponse(
            paper_id=paper_metadata.paper_id,
            title=paper_metadata.title,
            status="success",
            message="Paper ingested successfully. Embedding generation in progress."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _index_paper(paper: PaperMetadata):
    """Index paper in vector store (background task).

    Args:
        paper: Paper metadata
    """
    try:
        logger.info(f"Indexing paper: {paper.paper_id}")

        # Get full text
        full_text = paper.get_full_text()

        # Generate embeddings
        chunks, embeddings = embedding_service.chunk_and_embed(full_text)

        # Store in vector DB
        metadata = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "year": paper.year,
            "venue": paper.venue
        }

        vector_store.add_paper(
            paper_id=paper.paper_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )

        logger.info(f"Paper indexed: {paper.paper_id}")

    except Exception as e:
        logger.error(f"Error indexing paper {paper.paper_id}: {e}")


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_paper(request: SummarizeRequest):
    """Generate multi-granularity summary of a paper.

    Args:
        request: Summarization parameters

    Returns:
        Summary data
    """
    try:
        # Check cache
        if request.paper_id in summary_cache:
            logger.info(f"Returning cached summary for: {request.paper_id}")
            return SummarizeResponse(
                paper_id=request.paper_id,
                summary=summary_cache[request.paper_id].model_dump()
            )

        # Get paper metadata
        if request.paper_id not in paper_cache:
            raise HTTPException(status_code=404, detail=f"Paper not found: {request.paper_id}")

        paper = paper_cache[request.paper_id]

        logger.info(f"Generating summary for: {paper.title}")

        # Generate summary
        summary = summarizer.summarize_paper(paper)

        # Cache
        summary_cache[request.paper_id] = summary

        return SummarizeResponse(
            paper_id=request.paper_id,
            summary=summary.model_dump()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract")
async def extract_data(request: SummarizeRequest):
    """Extract structured data from a paper.

    Args:
        request: Extraction parameters

    Returns:
        Extracted data
    """
    try:
        # Check cache
        if request.paper_id in extracted_cache:
            logger.info(f"Returning cached extracted data for: {request.paper_id}")
            return extracted_cache[request.paper_id].model_dump()

        # Get paper metadata
        if request.paper_id not in paper_cache:
            raise HTTPException(status_code=404, detail=f"Paper not found: {request.paper_id}")

        paper = paper_cache[request.paper_id]

        logger.info(f"Extracting data from: {paper.title}")

        # Extract
        extracted_data = extractor.extract(paper)

        # Cache
        extracted_cache[request.paper_id] = extracted_data

        return extracted_data.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-pptx")
async def generate_presentation(request: GeneratePPTXRequest):
    """Generate PowerPoint presentation for a paper.

    Args:
        request: Generation parameters

    Returns:
        File download
    """
    try:
        # Get paper, summary, and extracted data
        if request.paper_id not in paper_cache:
            raise HTTPException(status_code=404, detail=f"Paper not found: {request.paper_id}")

        paper = paper_cache[request.paper_id]

        # Get or generate summary
        if request.paper_id not in summary_cache:
            summary = summarizer.summarize_paper(paper)
            summary_cache[request.paper_id] = summary
        else:
            summary = summary_cache[request.paper_id]

        # Get or generate extracted data
        if request.paper_id not in extracted_cache:
            extracted_data = extractor.extract(paper)
            extracted_cache[request.paper_id] = extracted_data
        else:
            extracted_data = extracted_cache[request.paper_id]

        logger.info(f"Generating presentation for: {paper.title}")

        # Generate PPTX
        output_path = presenter.generate_pptx(paper, summary, extracted_data)

        return FileResponse(
            path=output_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=os.path.basename(output_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating presentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-report")
async def generate_report(request: GenerateReportRequest):
    """Generate report for a paper.

    Args:
        request: Generation parameters

    Returns:
        File download
    """
    try:
        # Get paper and summary
        if request.paper_id not in paper_cache:
            raise HTTPException(status_code=404, detail=f"Paper not found: {request.paper_id}")

        paper = paper_cache[request.paper_id]

        if request.paper_id not in summary_cache:
            summary = summarizer.summarize_paper(paper)
            summary_cache[request.paper_id] = summary
        else:
            summary = summary_cache[request.paper_id]

        extracted_data = extracted_cache.get(request.paper_id)

        logger.info(f"Generating {request.format} report for: {paper.title}")

        # Generate Markdown
        output_path = presenter.generate_markdown_report(paper, summary, extracted_data)

        return FileResponse(
            path=output_path,
            media_type="text/markdown",
            filename=os.path.basename(output_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/semantic-search")
async def semantic_search(request: SemanticSearchRequest):
    """Perform semantic search across ingested papers.

    Args:
        request: Search parameters

    Returns:
        Search results
    """
    try:
        logger.info(f"Semantic search for: {request.query}")

        # Generate query embedding
        query_embedding = embedding_service.embed_text(request.query)

        # Search vector store
        results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_dict=request.filter
        )

        return {
            "query": request.query,
            "results": [
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "distance": results["distances"][i],
                    "metadata": results["metadatas"][i]
                }
                for i in range(len(results["ids"]))
            ]
        }

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers")
async def list_papers():
    """List all ingested papers.

    Returns:
        List of paper metadata
    """
    try:
        papers = []
        for paper_id, paper in paper_cache.items():
            papers.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "year": paper.year,
                "venue": paper.venue,
                "has_summary": paper_id in summary_cache,
                "has_extracted_data": paper_id in extracted_cache
            })

        return {"papers": papers, "count": len(papers)}

    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Get paper details.

    Args:
        paper_id: Paper identifier

    Returns:
        Paper metadata
    """
    if paper_id not in paper_cache:
        raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

    paper = paper_cache[paper_id]

    return {
        "paper": paper.model_dump(),
        "has_summary": paper_id in summary_cache,
        "has_extracted_data": paper_id in extracted_cache
    }


@app.post("/api/evaluate/{paper_id}")
async def evaluate_summary(paper_id: str):
    """Evaluate summary quality.

    Args:
        paper_id: Paper identifier

    Returns:
        Evaluation results
    """
    try:
        if paper_id not in paper_cache:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        if paper_id not in summary_cache:
            raise HTTPException(status_code=404, detail=f"Summary not found for: {paper_id}")

        paper = paper_cache[paper_id]
        summary = summary_cache[paper_id]

        logger.info(f"Evaluating summary for: {paper.title}")

        results = evaluator.evaluate_summary(paper, summary)

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get system statistics.

    Returns:
        Statistics
    """
    kg_stats = knowledge_graph.get_graph_statistics()

    return {
        "papers_ingested": len(paper_cache),
        "papers_summarized": len(summary_cache),
        "papers_extracted": len(extracted_cache),
        "vector_store_count": vector_store.count(),
        "knowledge_graph": kg_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


# Knowledge Graph endpoints
@app.post("/api/knowledge-graph/build")
async def build_knowledge_graph(background_tasks: BackgroundTasks):
    """Build knowledge graph from ingested papers.

    Returns:
        Build status
    """
    try:
        logger.info("Building knowledge graph...")

        # Build graph from all cached papers
        for paper_id, paper in paper_cache.items():
            extracted_data = extracted_cache.get(paper_id)
            if extracted_data:
                knowledge_graph.build_graph_from_paper(paper, extracted_data)

        stats = knowledge_graph.get_graph_statistics()

        return {
            "status": "success",
            "message": "Knowledge graph built successfully",
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-graph/export")
async def export_knowledge_graph():
    """Export knowledge graph to JSON.

    Returns:
        Graph data
    """
    try:
        return knowledge_graph.export_to_dict()

    except Exception as e:
        logger.error(f"Error exporting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge-graph/subgraph/{node_id}")
async def get_subgraph(node_id: str, depth: int = 2):
    """Get subgraph around a node.

    Args:
        node_id: Center node ID
        depth: Expansion depth

    Returns:
        Subgraph data
    """
    try:
        subgraph = knowledge_graph.get_subgraph(node_id, depth=depth)

        return {
            "center_node": node_id,
            "depth": depth,
            "nodes": list(subgraph.nodes()),
            "edges": list(subgraph.edges(data=True))
        }

    except Exception as e:
        logger.error(f"Error getting subgraph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Gap Discovery endpoints
@app.post("/api/gaps/discover")
async def discover_gaps():
    """Discover all research gaps.

    Returns:
        List of discovered gaps
    """
    try:
        logger.info("Discovering research gaps...")

        gaps = gap_discovery.discover_all_gaps()

        return {
            "total_gaps": len(gaps),
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "type": g.gap_type,
                    "title": g.title,
                    "description": g.description,
                    "confidence": g.confidence,
                    "impact": g.potential_impact,
                    "entities": g.entities[:10],  # Limit for response size
                    "evidence": g.supporting_evidence[:5]
                }
                for g in gaps
            ]
        }

    except Exception as e:
        logger.error(f"Error discovering gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gaps/{gap_type}")
async def get_gaps_by_type(gap_type: str):
    """Get gaps of a specific type.

    Args:
        gap_type: Gap type

    Returns:
        Filtered gaps
    """
    try:
        gaps = gap_discovery.get_gaps_by_type(gap_type)

        return {
            "gap_type": gap_type,
            "count": len(gaps),
            "gaps": [g.to_dict() for g in gaps]
        }

    except Exception as e:
        logger.error(f"Error getting gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gaps/validate")
async def validate_gaps():
    """Validate discovered gaps using LLM reasoning.

    Returns:
        Validation results
    """
    try:
        logger.info("Validating research gaps...")

        gaps = gap_discovery.gaps[:20]  # Validate top 20
        validations = []

        for gap in gaps:
            validation = llm_reasoner.validate_gap(gap)
            validations.append(llm_reasoner.export_validation(validation))

        valid_count = sum(1 for v in validations if v["is_valid"])

        return {
            "total_validated": len(validations),
            "valid_gaps": valid_count,
            "validations": validations
        }

    except Exception as e:
        logger.error(f"Error validating gaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gaps/report")
async def generate_gap_report(format: str = "markdown"):
    """Generate comprehensive research gap report.

    Args:
        format: Report format (markdown or html)

    Returns:
        File download
    """
    try:
        logger.info(f"Generating gap report in {format} format...")

        output_path = f"./outputs/gap_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format.split('/')[-1]}"

        report_path = gap_reporter.generate_full_report(
            output_path=output_path,
            format=format
        )

        media_type = "text/markdown" if format == "markdown" else "text/html"

        return FileResponse(
            path=report_path,
            media_type=media_type,
            filename=os.path.basename(report_path)
        )

    except Exception as e:
        logger.error(f"Error generating gap report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# GraphRAG endpoints
@app.post("/api/graph-rag/retrieve")
async def graph_rag_retrieve(query: str, top_k: int = 5, method: str = "hybrid"):
    """Retrieve using GraphRAG.

    Args:
        query: Search query
        top_k: Number of results
        method: Retrieval method (vector, graph, hybrid)

    Returns:
        Retrieval results
    """
    try:
        results = graph_rag.retrieve(query, top_k=top_k, method=method)

        return {
            "query": query,
            "method": method,
            "results": [
                {
                    "content": r.content,
                    "source_id": r.source_id,
                    "source_type": r.source_type,
                    "score": r.score,
                    "retrieval_method": r.retrieval_method
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Error in GraphRAG retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph-rag/reasoning-paths")
async def get_reasoning_paths(source: str, target: str):
    """Find reasoning paths between entities.

    Args:
        source: Source entity ID
        target: Target entity ID

    Returns:
        Paths and explanations
    """
    try:
        paths = graph_rag.multi_hop_query(source, target, max_hops=3)

        return {
            "source": source,
            "target": target,
            "paths_found": len(paths),
            "paths": [
                {
                    "nodes": path,
                    "length": len(path) - 1,
                    "explanation": graph_rag.explain_path(path)
                }
                for path in paths
            ]
        }

    except Exception as e:
        logger.error(f"Error finding reasoning paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Link Prediction endpoints
@app.post("/api/link-prediction/train")
async def train_link_predictor():
    """Train link prediction model.

    Returns:
        Training status
    """
    try:
        logger.info("Training link prediction model...")

        link_predictor.train(use_node2vec=True)

        return {
            "status": "success",
            "message": "Link prediction model trained successfully"
        }

    except Exception as e:
        logger.error(f"Error training link predictor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/link-prediction/predict")
async def predict_missing_links(top_k: int = 20):
    """Predict missing links in the graph.

    Args:
        top_k: Number of predictions

    Returns:
        Link predictions
    """
    try:
        predictions = link_predictor.predict_missing_links(top_k=top_k)

        return {
            "total_predictions": len(predictions),
            "predictions": link_predictor.export_predictions(predictions)
        }

    except Exception as e:
        logger.error(f"Error predicting links: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/link-prediction/for-node/{node_id}")
async def predict_for_node(node_id: str, top_k: int = 10):
    """Predict missing links for a specific node.

    Args:
        node_id: Node ID
        top_k: Number of predictions

    Returns:
        Link predictions
    """
    try:
        predictions = link_predictor.predict_for_node(node_id, top_k=top_k)

        return {
            "node_id": node_id,
            "predictions": link_predictor.export_predictions(predictions)
        }

    except Exception as e:
        logger.error(f"Error predicting links for node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Causal Reasoning endpoints
@app.post("/api/causal/build-graph")
async def build_causal_graph():
    """Build causal graph from knowledge graph.

    Returns:
        Causal graph statistics
    """
    try:
        logger.info("Building causal graph...")

        causal_reasoner.build_causal_graph()

        stats = causal_reasoner.get_causal_statistics()

        return {
            "status": "success",
            "message": "Causal graph built successfully",
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error building causal graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/causal/predict-breakthroughs")
async def predict_breakthroughs(top_k: int = 10):
    """Predict breakthrough opportunities using causal reasoning.

    Args:
        top_k: Number of predictions

    Returns:
        Breakthrough predictions
    """
    try:
        logger.info("Predicting breakthrough opportunities...")

        breakthroughs = causal_reasoner.predict_breakthroughs(top_k=top_k)

        return {
            "total_predictions": len(breakthroughs),
            "breakthroughs": causal_reasoner.export_predictions(breakthroughs)
        }

    except Exception as e:
        logger.error(f"Error predicting breakthroughs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/causal/explain-path")
async def explain_causal_path(source: str, target: str):
    """Explain causal path between two entities.

    Args:
        source: Source entity ID
        target: Target entity ID

    Returns:
        Path explanation
    """
    try:
        explanation = causal_reasoner.explain_causal_path(source, target)

        if not explanation:
            raise HTTPException(status_code=404, detail="No causal path found")

        return explanation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining causal path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Hypothesis Tree endpoints
@app.post("/api/hypothesis/generate-tree")
async def generate_hypothesis_tree(claim: str, max_depth: int = 3):
    """Generate hypothesis tree from a causal claim.

    Args:
        claim: Root hypothesis claim
        max_depth: Maximum tree depth

    Returns:
        Hypothesis tree
    """
    try:
        logger.info(f"Generating hypothesis tree for: {claim}")

        hypotheses = hypothesis_generator.generate_hypothesis_tree(
            root_claim=claim,
            max_depth=max_depth
        )

        tree_structure = hypothesis_generator.get_hypothesis_tree_structure()

        return {
            "hypotheses": hypothesis_generator.export_hypotheses(),
            "tree_structure": tree_structure
        }

    except Exception as e:
        logger.error(f"Error generating hypothesis tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hypothesis/from-breakthrough/{breakthrough_id}")
async def generate_hypotheses_from_breakthrough(breakthrough_id: str):
    """Generate hypothesis tree from a breakthrough prediction.

    Args:
        breakthrough_id: Breakthrough prediction ID

    Returns:
        Hypothesis tree
    """
    try:
        # Get breakthrough prediction
        breakthroughs = causal_reasoner.predict_breakthroughs(top_k=50)
        breakthrough = next((b for b in breakthroughs if b.prediction_id == breakthrough_id), None)

        if not breakthrough:
            raise HTTPException(status_code=404, detail=f"Breakthrough not found: {breakthrough_id}")

        hypotheses = hypothesis_generator.generate_from_breakthrough(breakthrough, max_depth=2)

        return {
            "breakthrough": breakthrough.to_dict(),
            "hypotheses": hypothesis_generator.export_hypotheses(),
            "research_plan": hypothesis_generator.generate_research_plan()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating hypotheses from breakthrough: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CITATION NETWORK ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/citations/build-network")
async def build_citation_network():
    """Build citation network from all ingested papers.

    Returns:
        Network statistics and metadata
    """
    try:
        logger.info("Building citation network...")

        # Convert cached papers to format expected by citation network
        papers_data = []
        for paper_id, paper in paper_cache.items():
            papers_data.append({
                'paper_id': paper_id,
                'title': paper.title,
                'authors': paper.authors,
                'publication_date': paper.publication_date or datetime.now(),
                'venue': paper.venue,
                'abstract': paper.abstract,
                'topics': [],
                'references': getattr(paper, 'references', []),
                'citations': getattr(paper, 'citations', [])
            })

        if len(papers_data) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 papers to build citation network"
            )

        # Build network
        stats = citation_network.build_network_from_papers(papers_data)

        # Compute metrics
        citation_network.compute_network_metrics()

        # Detect communities
        communities = citation_network.detect_communities(method="louvain")

        return {
            "status": "success",
            "stats": stats,
            "communities": len(communities),
            "message": f"Citation network built with {stats['total_papers']} papers"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building citation network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/foundational")
async def get_foundational_papers(top_k: int = 10):
    """Get foundational papers (highest PageRank).

    Args:
        top_k: Number of papers to return

    Returns:
        List of foundational papers with metrics
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Citation network not built yet. Call /api/citations/build-network first"
            )

        foundational = citation_network.find_foundational_papers(top_k=top_k)

        return {
            "count": len(foundational),
            "papers": foundational
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding foundational papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/hidden-gems")
async def get_hidden_gems(top_k: int = 10):
    """Get hidden gem papers (high betweenness, low citations).

    Args:
        top_k: Number of papers to return

    Returns:
        List of hidden gem papers
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Citation network not built yet"
            )

        gems = citation_network.find_hidden_gems(top_k=top_k)

        return {
            "count": len(gems),
            "papers": gems
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding hidden gems: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/analyze/{paper_id}")
async def analyze_paper_citations(paper_id: str):
    """Get detailed citation analysis for a specific paper.

    Args:
        paper_id: Paper ID

    Returns:
        Citation metrics and predictions
    """
    try:
        if paper_id not in citation_network.nodes:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        node = citation_network.nodes[paper_id]

        # Get citation prediction
        prediction = citation_network.predict_citation_impact(paper_id)

        # Get research lineage
        lineage = citation_network.trace_research_lineage(paper_id, max_generations=3)

        return {
            "paper_id": paper_id,
            "title": node.title,
            "metrics": {
                "citations": node.in_citations,
                "references": node.out_citations,
                "pagerank": node.pagerank,
                "betweenness": node.betweenness_centrality,
                "category": node.category.value,
                "community": node.community_id,
                "citation_velocity": node.citation_velocity
            },
            "prediction": {
                "1_year": prediction.predicted_citations_1year,
                "3_year": prediction.predicted_citations_3year,
                "5_year": prediction.predicted_citations_5year,
                "confidence": prediction.confidence_score,
                "comparable_papers": prediction.comparable_papers
            },
            "lineage": {
                "generations": len(lineage.generations),
                "total_descendants": lineage.total_descendants,
                "key_papers": lineage.key_papers[:5],
                "branches": lineage.branches
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing paper citations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/predict-impact/{paper_id}")
async def predict_citation_impact(paper_id: str):
    """Predict future citation impact for a paper.

    Args:
        paper_id: Paper ID

    Returns:
        Citation predictions for 1, 3, and 5 years
    """
    try:
        if paper_id not in citation_network.nodes:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        prediction = citation_network.predict_citation_impact(paper_id)

        return {
            "paper_id": paper_id,
            "current_citations": citation_network.nodes[paper_id].in_citations,
            "predictions": {
                "1_year": prediction.predicted_citations_1year,
                "3_year": prediction.predicted_citations_3year,
                "5_year": prediction.predicted_citations_5year
            },
            "confidence": prediction.confidence_score,
            "factors": prediction.factors,
            "comparable_papers": [
                {
                    "paper_id": pid,
                    "title": citation_network.nodes[pid].title,
                    "citations": citation_network.nodes[pid].in_citations
                }
                for pid in prediction.comparable_papers[:5]
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting citation impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/lineage/{paper_id}")
async def trace_research_lineage(paper_id: str, max_generations: int = 5):
    """Trace the research lineage from a root paper.

    Args:
        paper_id: Root paper ID
        max_generations: Maximum generations to trace

    Returns:
        Research lineage with generations and branches
    """
    try:
        if paper_id not in citation_network.nodes:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")

        lineage = citation_network.trace_research_lineage(paper_id, max_generations=max_generations)

        # Enrich with paper details
        generations_detailed = []
        for gen in lineage.generations:
            gen_papers = []
            for pid in gen:
                node = citation_network.nodes[pid]
                gen_papers.append({
                    "paper_id": pid,
                    "title": node.title,
                    "authors": node.authors,
                    "citations": node.in_citations,
                    "pagerank": node.pagerank
                })
            generations_detailed.append(gen_papers)

        return {
            "root_paper_id": paper_id,
            "root_title": citation_network.nodes[paper_id].title,
            "generations": generations_detailed,
            "total_descendants": lineage.total_descendants,
            "key_papers": [
                {
                    "paper_id": pid,
                    "title": citation_network.nodes[pid].title,
                    "pagerank": citation_network.nodes[pid].pagerank
                }
                for pid in lineage.key_papers
            ],
            "branches": lineage.branches
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracing lineage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/communities")
async def get_research_communities():
    """Get research communities detected in citation network.

    Returns:
        List of communities with member papers
    """
    try:
        if len(citation_network.communities) == 0:
            # Detect communities if not already done
            if len(citation_network.nodes) > 0:
                citation_network.detect_communities(method="louvain")
            else:
                raise HTTPException(status_code=400, detail="Citation network not built yet")

        communities_detailed = []
        for community_id, paper_ids in citation_network.communities.items():
            papers = []
            for pid in paper_ids[:20]:  # Limit to 20 papers per community
                node = citation_network.nodes[pid]
                papers.append({
                    "paper_id": pid,
                    "title": node.title,
                    "citations": node.in_citations,
                    "pagerank": node.pagerank
                })

            # Find most cited papers in community
            papers.sort(key=lambda x: x['pagerank'], reverse=True)

            communities_detailed.append({
                "community_id": community_id,
                "size": len(paper_ids),
                "top_papers": papers[:10]
            })

        return {
            "total_communities": len(communities_detailed),
            "communities": sorted(communities_detailed, key=lambda x: x['size'], reverse=True)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting communities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/authors")
async def get_influential_authors(top_k: int = 20):
    """Get most influential authors in the network.

    Args:
        top_k: Number of authors to return

    Returns:
        List of authors with influence metrics
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(status_code=400, detail="Citation network not built yet")

        authors = citation_network.get_influential_authors(top_k=top_k)

        return {
            "count": len(authors),
            "authors": authors
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting influential authors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/anomalies")
async def detect_citation_anomalies():
    """Detect unusual citation patterns.

    Returns:
        List of citation anomalies
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(status_code=400, detail="Citation network not built yet")

        anomalies = citation_network.detect_citation_anomalies()

        anomalies_detailed = []
        for anomaly in anomalies:
            node = citation_network.nodes[anomaly.paper_id]
            anomalies_detailed.append({
                "paper_id": anomaly.paper_id,
                "title": node.title,
                "anomaly_type": anomaly.anomaly_type,
                "severity": anomaly.severity,
                "description": anomaly.description,
                "evidence": anomaly.evidence
            })

        return {
            "count": len(anomalies_detailed),
            "anomalies": anomalies_detailed
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/summary")
async def get_citation_network_summary():
    """Get comprehensive citation network statistics.

    Returns:
        Network summary with all key metrics
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(status_code=400, detail="Citation network not built yet")

        summary = citation_network.get_network_summary()

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting network summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/citations/export")
async def export_citation_network(format: str = "json"):
    """Export citation network for visualization.

    Args:
        format: Export format ("json", "graphml", "gexf")

    Returns:
        FileResponse with network data
    """
    try:
        if len(citation_network.nodes) == 0:
            raise HTTPException(status_code=400, detail="Citation network not built yet")

        # Create output directory if needed
        os.makedirs("./data/exports", exist_ok=True)

        output_path = f"./data/exports/citation_network.{format}"
        citation_network.export_network(output_path, format=format)

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"citation_network.{format}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LIVE RESEARCH ALERTS & MONITORING ENDPOINTS
# ============================================================================

class AlertConfigRequest(BaseModel):
    """Request model for creating/updating alert configs"""
    user_id: str
    name: str
    description: Optional[str] = None
    keywords: List[str] = []
    authors: List[str] = []
    affiliations: List[str] = []
    venues: List[str] = []
    categories: List[str] = []
    min_relevance_score: float = 0.7
    exclude_keywords: List[str] = []
    sources: List[str] = ["arxiv"]
    check_frequency: str = "daily"
    priority: str = "medium"
    enabled: bool = True
    email_notification: bool = True
    webhook_url: Optional[str] = None
    max_alerts_per_day: int = 50


@app.post("/api/alerts/create")
async def create_alert(request: AlertConfigRequest):
    """Create a new research alert configuration.

    Args:
        request: Alert configuration

    Returns:
        Created alert config with ID
    """
    try:
        import uuid

        # Convert sources strings to AlertSource enums
        sources = [AlertSource(s.lower()) for s in request.sources if s.lower() in [e.value for e in AlertSource]]

        # Create config
        alert_config = AlertConfig(
            alert_id=str(uuid.uuid4()),
            user_id=request.user_id,
            name=request.name,
            description=request.description,
            keywords=request.keywords,
            authors=request.authors,
            affiliations=request.affiliations,
            venues=request.venues,
            categories=request.categories,
            min_relevance_score=request.min_relevance_score,
            exclude_keywords=request.exclude_keywords,
            sources=sources,
            check_frequency=request.check_frequency,
            priority=AlertPriority(request.priority.lower()),
            enabled=request.enabled,
            email_notification=request.email_notification,
            webhook_url=request.webhook_url,
            max_alerts_per_day=request.max_alerts_per_day
        )

        alert_id = research_monitor.add_alert(alert_config)

        return {
            "status": "success",
            "alert_id": alert_id,
            "message": f"Alert '{request.name}' created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/list/{user_id}")
async def list_user_alerts(user_id: str):
    """Get all alert configurations for a user.

    Args:
        user_id: User ID

    Returns:
        List of alert configurations
    """
    try:
        alert_ids = research_monitor.user_alerts.get(user_id, [])
        configs = [research_monitor.alert_configs[aid] for aid in alert_ids]

        return {
            "user_id": user_id,
            "count": len(configs),
            "alerts": [
                {
                    "alert_id": c.alert_id,
                    "name": c.name,
                    "description": c.description,
                    "keywords": c.keywords,
                    "authors": c.authors,
                    "enabled": c.enabled,
                    "sources": [s.value for s in c.sources],
                    "check_frequency": c.check_frequency,
                    "priority": c.priority.value,
                    "created_at": c.created_at.isoformat(),
                    "last_checked": c.last_checked.isoformat() if c.last_checked else None,
                    "total_alerts_sent": c.total_alerts_sent
                }
                for c in configs
            ]
        }

    except Exception as e:
        logger.error(f"Error listing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/alerts/update/{alert_id}")
async def update_alert(alert_id: str, updates: Dict):
    """Update an existing alert configuration.

    Args:
        alert_id: Alert ID
        updates: Dictionary of fields to update

    Returns:
        Success message
    """
    try:
        research_monitor.update_alert(alert_id, updates)

        return {
            "status": "success",
            "message": f"Alert {alert_id} updated successfully"
        }

    except Exception as e:
        logger.error(f"Error updating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/alerts/delete/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert configuration.

    Args:
        alert_id: Alert ID

    Returns:
        Success message
    """
    try:
        research_monitor.remove_alert(alert_id)

        return {
            "status": "success",
            "message": f"Alert {alert_id} deleted successfully"
        }

    except Exception as e:
        logger.error(f"Error deleting alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/check")
async def check_alerts(source: Optional[str] = None):
    """Manually trigger alert checking.

    Args:
        source: Optional source to check (arxiv, pubmed, etc.)

    Returns:
        List of new alerts generated
    """
    try:
        logger.info(f"Manually checking alerts (source: {source or 'all'})")

        source_enum = AlertSource(source.lower()) if source else None
        new_alerts = await research_monitor.check_for_updates(source=source_enum)

        return {
            "status": "success",
            "alerts_generated": len(new_alerts),
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "paper_id": a.paper_id,
                    "title": a.title,
                    "authors": a.authors,
                    "published_date": a.published_date.isoformat(),
                    "relevance_score": a.relevance_score,
                    "matched_keywords": a.matched_keywords,
                    "matched_authors": a.matched_authors,
                    "reason": a.reason,
                    "priority": a.priority.value,
                    "pdf_url": a.pdf_url,
                    "auto_summary": a.auto_summary
                }
                for a in new_alerts[:50]  # Return max 50
            ]
        }

    except Exception as e:
        logger.error(f"Error checking alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/notifications/{user_id}")
async def get_user_notifications(user_id: str, unread_only: bool = False):
    """Get alert notifications for a user.

    Args:
        user_id: User ID
        unread_only: Only return unread alerts

    Returns:
        List of alert notifications
    """
    try:
        alerts = research_monitor.get_user_alerts(user_id, unread_only=unread_only)

        return {
            "user_id": user_id,
            "count": len(alerts),
            "unread_count": sum(1 for a in alerts if not a.read),
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "paper_id": a.paper_id,
                    "title": a.title,
                    "authors": a.authors[:5],
                    "abstract": a.abstract[:300] + "..." if len(a.abstract) > 300 else a.abstract,
                    "published_date": a.published_date.isoformat(),
                    "source": a.source.value,
                    "match_type": a.match_type.value,
                    "relevance_score": a.relevance_score,
                    "matched_keywords": a.matched_keywords,
                    "matched_authors": a.matched_authors,
                    "reason": a.reason,
                    "priority": a.priority.value,
                    "pdf_url": a.pdf_url,
                    "arxiv_id": a.arxiv_id,
                    "created_at": a.created_at.isoformat(),
                    "read": a.read,
                    "dismissed": a.dismissed,
                    "auto_summary": a.auto_summary
                }
                for a in alerts[:100]  # Return max 100
            ]
        }

    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/mark-read/{alert_id}")
async def mark_alert_read(alert_id: str):
    """Mark an alert as read.

    Args:
        alert_id: Alert ID

    Returns:
        Success message
    """
    try:
        research_monitor.mark_alert_read(alert_id)

        return {
            "status": "success",
            "message": f"Alert {alert_id} marked as read"
        }

    except Exception as e:
        logger.error(f"Error marking alert as read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/dismiss/{alert_id}")
async def dismiss_alert(alert_id: str):
    """Dismiss an alert.

    Args:
        alert_id: Alert ID

    Returns:
        Success message
    """
    try:
        research_monitor.dismiss_alert(alert_id)

        return {
            "status": "success",
            "message": f"Alert {alert_id} dismissed"
        }

    except Exception as e:
        logger.error(f"Error dismissing alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/trends")
async def get_trend_alerts(time_window_days: int = 30):
    """Get emerging research trends.

    Args:
        time_window_days: Look at alerts from last N days

    Returns:
        List of trend alerts
    """
    try:
        trends = research_monitor.detect_trends(time_window_days=time_window_days)

        return {
            "time_window_days": time_window_days,
            "trends_detected": len(trends),
            "trends": [
                {
                    "trend_id": t.trend_id,
                    "trend_name": t.trend_name,
                    "description": t.description,
                    "paper_count": t.paper_count,
                    "growth_rate": t.growth_rate,
                    "key_papers": t.key_papers[:10],
                    "first_observed": t.first_observed.isoformat()
                }
                for t in trends
            ]
        }

    except Exception as e:
        logger.error(f"Error detecting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/alerts/competitors")
async def detect_competitor_activity(competitors: List[str]):
    """Detect when competitors publish papers.

    Args:
        competitors: List of competitor names

    Returns:
        List of competitor alerts
    """
    try:
        comp_alerts = research_monitor.detect_competitor_activity(competitors)

        return {
            "competitors_monitored": competitors,
            "alerts_detected": len(comp_alerts),
            "alerts": [
                {
                    "competitor": ca.competitor_name,
                    "paper_id": ca.paper_id,
                    "title": ca.title,
                    "published_date": ca.published_date.isoformat(),
                    "overlapping_keywords": ca.overlapping_keywords,
                    "threat_level": ca.threat_level,
                    "action_items": ca.action_items
                }
                for ca in comp_alerts
            ]
        }

    except Exception as e:
        logger.error(f"Error detecting competitor activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/statistics")
async def get_monitoring_statistics():
    """Get research monitoring statistics.

    Returns:
        Statistics about monitoring activity
    """
    try:
        stats = research_monitor.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/export/{user_id}")
async def export_user_alerts(user_id: str, format: str = "json"):
    """Export user's alerts.

    Args:
        user_id: User ID
        format: Export format (json or csv)

    Returns:
        Exported alerts
    """
    try:
        exported = research_monitor.export_alerts(user_id, format=format)

        return {
            "user_id": user_id,
            "format": format,
            "data": exported
        }

    except Exception as e:
        logger.error(f"Error exporting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COLLABORATIVE WORKSPACE ENDPOINTS
# ============================================================================

class CreateWorkspaceRequest(BaseModel):
    """Request to create workspace"""
    name: str
    owner_id: str
    owner_username: str
    owner_email: str
    description: Optional[str] = None
    tags: List[str] = []
    is_public: bool = False


class AddMemberRequest(BaseModel):
    """Request to add member"""
    user_id: str
    username: str
    email: str
    role: str = "member"
    added_by: str


class CreateTaskRequest(BaseModel):
    """Request to create task"""
    title: str
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    assigned_to_username: Optional[str] = None
    priority: str = "medium"
    due_date: Optional[str] = None
    related_papers: List[str] = []
    tags: List[str] = []


@app.post("/api/workspaces/create")
async def create_workspace(request: CreateWorkspaceRequest):
    """Create a new collaborative workspace.

    Args:
        request: Workspace configuration

    Returns:
        Created workspace
    """
    try:
        workspace = workspace_manager.create_workspace(
            name=request.name,
            owner_id=request.owner_id,
            owner_username=request.owner_username,
            owner_email=request.owner_email,
            description=request.description,
            tags=request.tags,
            is_public=request.is_public
        )

        return {
            "status": "success",
            "workspace_id": workspace.workspace_id,
            "workspace": {
                "workspace_id": workspace.workspace_id,
                "name": workspace.name,
                "description": workspace.description,
                "owner_id": workspace.owner_id,
                "members": len(workspace.members),
                "created_at": workspace.created_at.isoformat(),
                "tags": workspace.tags
            }
        }

    except Exception as e:
        logger.error(f"Error creating workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/user/{user_id}")
async def get_user_workspaces(user_id: str):
    """Get all workspaces for a user.

    Args:
        user_id: User ID

    Returns:
        List of workspaces
    """
    try:
        workspaces = workspace_manager.get_user_workspaces(user_id)

        return {
            "user_id": user_id,
            "count": len(workspaces),
            "workspaces": [
                {
                    "workspace_id": w.workspace_id,
                    "name": w.name,
                    "description": w.description,
                    "owner_id": w.owner_id,
                    "member_count": len(w.members),
                    "paper_count": w.total_papers,
                    "task_count": w.total_tasks,
                    "is_public": w.is_public,
                    "created_at": w.created_at.isoformat(),
                    "updated_at": w.updated_at.isoformat(),
                    "tags": w.tags,
                    "user_role": w.members[user_id].role.value if user_id in w.members else None
                }
                for w in workspaces
            ]
        }

    except Exception as e:
        logger.error(f"Error getting user workspaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    """Get workspace details.

    Args:
        workspace_id: Workspace ID

    Returns:
        Workspace details
    """
    try:
        workspace = workspace_manager.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return {
            "workspace_id": workspace.workspace_id,
            "name": workspace.name,
            "description": workspace.description,
            "owner_id": workspace.owner_id,
            "is_public": workspace.is_public,
            "created_at": workspace.created_at.isoformat(),
            "updated_at": workspace.updated_at.isoformat(),
            "tags": workspace.tags,
            "members": [
                {
                    "user_id": m.user_id,
                    "username": m.username,
                    "email": m.email,
                    "role": m.role.value,
                    "joined_at": m.joined_at.isoformat(),
                    "contributions": m.contributions
                }
                for m in workspace.members.values()
            ],
            "stats": workspace_manager.get_workspace_stats(workspace_id)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspaces/{workspace_id}/members")
async def add_workspace_member(workspace_id: str, request: AddMemberRequest):
    """Add a member to workspace.

    Args:
        workspace_id: Workspace ID
        request: Member details

    Returns:
        Success message
    """
    try:
        role = WorkspaceRole(request.role.lower())

        success = workspace_manager.add_member(
            workspace_id=workspace_id,
            user_id=request.user_id,
            username=request.username,
            email=request.email,
            role=role,
            added_by=request.added_by
        )

        if not success:
            raise HTTPException(status_code=400, detail="Could not add member")

        return {
            "status": "success",
            "message": f"Member {request.username} added to workspace"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workspaces/{workspace_id}/members/{user_id}")
async def remove_workspace_member(workspace_id: str, user_id: str, removed_by: str):
    """Remove a member from workspace.

    Args:
        workspace_id: Workspace ID
        user_id: User ID to remove
        removed_by: User ID performing removal

    Returns:
        Success message
    """
    try:
        success = workspace_manager.remove_member(workspace_id, user_id, removed_by)

        if not success:
            raise HTTPException(status_code=400, detail="Could not remove member")

        return {
            "status": "success",
            "message": "Member removed from workspace"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspaces/{workspace_id}/papers/{paper_id}")
async def add_paper_to_workspace(workspace_id: str, paper_id: str, added_by: str, username: str):
    """Add a paper to workspace.

    Args:
        workspace_id: Workspace ID
        paper_id: Paper ID
        added_by: User ID adding paper
        username: Username

    Returns:
        Success message
    """
    try:
        success = workspace_manager.add_paper_to_workspace(
            workspace_id=workspace_id,
            paper_id=paper_id,
            added_by=added_by,
            username=username
        )

        if not success:
            raise HTTPException(status_code=400, detail="Could not add paper")

        return {
            "status": "success",
            "message": "Paper added to workspace"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/papers")
async def get_workspace_papers(workspace_id: str):
    """Get all papers in workspace.

    Args:
        workspace_id: Workspace ID

    Returns:
        List of paper IDs
    """
    try:
        paper_ids = workspace_manager.get_workspace_papers(workspace_id)

        return {
            "workspace_id": workspace_id,
            "count": len(paper_ids),
            "paper_ids": paper_ids
        }

    except Exception as e:
        logger.error(f"Error getting workspace papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspaces/{workspace_id}/annotations")
async def add_annotation(
    workspace_id: str,
    paper_id: str,
    user_id: str,
    username: str,
    text: str,
    annotation_type: str = "note",
    highlight_text: Optional[str] = None,
    page_number: Optional[int] = None,
    tags: List[str] = []
):
    """Add annotation to a paper.

    Args:
        workspace_id: Workspace ID
        paper_id: Paper ID
        user_id: User ID
        username: Username
        text: Annotation text
        annotation_type: Type of annotation
        highlight_text: Highlighted text
        page_number: Page number
        tags: Tags

    Returns:
        Created annotation
    """
    try:
        annotation = workspace_manager.add_annotation(
            workspace_id=workspace_id,
            paper_id=paper_id,
            user_id=user_id,
            username=username,
            text=text,
            annotation_type=annotation_type,
            highlight_text=highlight_text,
            page_number=page_number,
            tags=tags
        )

        return {
            "status": "success",
            "annotation": {
                "annotation_id": annotation.annotation_id,
                "paper_id": annotation.paper_id,
                "user_id": annotation.user_id,
                "username": annotation.username,
                "text": annotation.text,
                "annotation_type": annotation.annotation_type,
                "highlight_text": annotation.highlight_text,
                "page_number": annotation.page_number,
                "tags": annotation.tags,
                "created_at": annotation.created_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error adding annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/annotations")
async def get_annotations(
    workspace_id: str,
    paper_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Get annotations in workspace.

    Args:
        workspace_id: Workspace ID
        paper_id: Optional paper filter
        user_id: Optional user filter

    Returns:
        List of annotations
    """
    try:
        annotations = workspace_manager.get_annotations(workspace_id, paper_id, user_id)

        return {
            "workspace_id": workspace_id,
            "count": len(annotations),
            "annotations": [
                {
                    "annotation_id": a.annotation_id,
                    "paper_id": a.paper_id,
                    "user_id": a.user_id,
                    "username": a.username,
                    "text": a.text,
                    "annotation_type": a.annotation_type,
                    "highlight_text": a.highlight_text,
                    "page_number": a.page_number,
                    "tags": a.tags,
                    "created_at": a.created_at.isoformat(),
                    "replies_count": len(a.replies),
                    "upvotes": a.upvotes
                }
                for a in annotations
            ]
        }

    except Exception as e:
        logger.error(f"Error getting annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspaces/{workspace_id}/comments")
async def add_comment(
    workspace_id: str,
    parent_id: str,
    parent_type: str,
    user_id: str,
    username: str,
    text: str,
    mentions: List[str] = []
):
    """Add comment to annotation or paper.

    Args:
        workspace_id: Workspace ID
        parent_id: Parent annotation or paper ID
        parent_type: "annotation" or "paper"
        user_id: User ID
        username: Username
        text: Comment text
        mentions: List of mentioned usernames

    Returns:
        Created comment
    """
    try:
        comment = workspace_manager.add_comment(
            workspace_id=workspace_id,
            parent_id=parent_id,
            parent_type=parent_type,
            user_id=user_id,
            username=username,
            text=text,
            mentions=mentions
        )

        return {
            "status": "success",
            "comment": {
                "comment_id": comment.comment_id,
                "parent_id": comment.parent_id,
                "parent_type": comment.parent_type,
                "user_id": comment.user_id,
                "username": comment.username,
                "text": comment.text,
                "mentions": comment.mentions,
                "created_at": comment.created_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Error adding comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/comments")
async def get_comments(workspace_id: str, parent_id: Optional[str] = None):
    """Get comments in workspace.

    Args:
        workspace_id: Workspace ID
        parent_id: Optional parent filter

    Returns:
        List of comments
    """
    try:
        comments = workspace_manager.get_comments(workspace_id, parent_id)

        return {
            "workspace_id": workspace_id,
            "count": len(comments),
            "comments": [
                {
                    "comment_id": c.comment_id,
                    "parent_id": c.parent_id,
                    "parent_type": c.parent_type,
                    "user_id": c.user_id,
                    "username": c.username,
                    "text": c.text,
                    "mentions": c.mentions,
                    "created_at": c.created_at.isoformat(),
                    "upvotes": c.upvotes
                }
                for c in comments
            ]
        }

    except Exception as e:
        logger.error(f"Error getting comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/workspaces/{workspace_id}/tasks")
async def create_task(workspace_id: str, created_by: str, creator_username: str, request: CreateTaskRequest):
    """Create a research task.

    Args:
        workspace_id: Workspace ID
        created_by: Creator user ID
        creator_username: Creator username
        request: Task details

    Returns:
        Created task
    """
    try:
        priority = TaskPriority(request.priority.lower())
        due_date = datetime.fromisoformat(request.due_date) if request.due_date else None

        task = workspace_manager.create_task(
            workspace_id=workspace_id,
            title=request.title,
            created_by=created_by,
            creator_username=creator_username,
            description=request.description,
            assigned_to=request.assigned_to,
            assigned_to_username=request.assigned_to_username,
            priority=priority,
            due_date=due_date,
            related_papers=request.related_papers,
            tags=request.tags
        )

        return {
            "status": "success",
            "task": {
                "task_id": task.task_id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "assigned_to": task.assigned_to,
                "assigned_to_username": task.assigned_to_username,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "created_at": task.created_at.isoformat(),
                "related_papers": task.related_papers,
                "tags": task.tags
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {request.priority}")
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/tasks")
async def get_tasks(
    workspace_id: str,
    assigned_to: Optional[str] = None,
    status: Optional[str] = None
):
    """Get tasks in workspace.

    Args:
        workspace_id: Workspace ID
        assigned_to: Optional assignee filter
        status: Optional status filter

    Returns:
        List of tasks
    """
    try:
        status_enum = TaskStatus(status.lower()) if status else None
        tasks = workspace_manager.get_tasks(workspace_id, assigned_to, status_enum)

        return {
            "workspace_id": workspace_id,
            "count": len(tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "title": t.title,
                    "description": t.description,
                    "status": t.status.value,
                    "priority": t.priority.value,
                    "created_by": t.created_by,
                    "assigned_to": t.assigned_to,
                    "assigned_to_username": t.assigned_to_username,
                    "due_date": t.due_date.isoformat() if t.due_date else None,
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "progress_percentage": t.progress_percentage,
                    "created_at": t.created_at.isoformat(),
                    "related_papers": t.related_papers,
                    "tags": t.tags
                }
                for t in tasks
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/workspaces/{workspace_id}/tasks/{task_id}")
async def update_task(workspace_id: str, task_id: str, updates: Dict, updated_by: str, username: str):
    """Update a task.

    Args:
        workspace_id: Workspace ID
        task_id: Task ID
        updates: Fields to update
        updated_by: User ID performing update
        username: Username

    Returns:
        Success message
    """
    try:
        success = workspace_manager.update_task(workspace_id, task_id, updates, updated_by, username)

        if not success:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "status": "success",
            "message": "Task updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/activities")
async def get_activities(workspace_id: str, limit: int = 50):
    """Get recent workspace activities.

    Args:
        workspace_id: Workspace ID
        limit: Maximum number of activities

    Returns:
        List of activities
    """
    try:
        activities = workspace_manager.get_activities(workspace_id, limit)

        return {
            "workspace_id": workspace_id,
            "count": len(activities),
            "activities": [
                {
                    "activity_id": a.activity_id,
                    "user_id": a.user_id,
                    "username": a.username,
                    "activity_type": a.activity_type.value,
                    "description": a.description,
                    "metadata": a.metadata,
                    "created_at": a.created_at.isoformat()
                }
                for a in activities
            ]
        }

    except Exception as e:
        logger.error(f"Error getting activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspaces/{workspace_id}/stats")
async def get_workspace_stats(workspace_id: str):
    """Get workspace statistics.

    Args:
        workspace_id: Workspace ID

    Returns:
        Workspace stats
    """
    try:
        stats = workspace_manager.get_workspace_stats(workspace_id)

        return stats

    except Exception as e:
        logger.error(f"Error getting workspace stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# LITERATURE REVIEW GENERATOR ENDPOINTS
# ============================================================================

class GenerateReviewRequest(BaseModel):
    """Request to generate literature review"""
    paper_ids: List[str]
    research_question: str
    style: str = "narrative"
    citation_style: str = "apa"
    include_methodology: bool = True
    include_discussion: bool = True
    min_papers_per_theme: int = 3


@app.post("/api/lit-review/generate")
async def generate_literature_review(request: GenerateReviewRequest):
    """Generate a complete literature review.

    Args:
        request: Review generation parameters

    Returns:
        Generated literature review
    """
    try:
        logger.info(f"Generating literature review for {len(request.paper_ids)} papers")

        # Get papers from cache
        papers = []
        for paper_id in request.paper_ids:
            if paper_id in paper_cache:
                paper_meta = paper_cache[paper_id]
                papers.append({
                    'paper_id': paper_id,
                    'title': paper_meta.title,
                    'authors': paper_meta.authors,
                    'publication_date': paper_meta.publication_date or datetime.now(),
                    'venue': paper_meta.venue,
                    'abstract': paper_meta.abstract,
                    'keywords': getattr(paper_meta, 'keywords', []),
                    'topics': []
                })

        if len(papers) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 2 papers for literature review. Found {len(papers)}"
            )

        # Convert style and citation style
        review_style = ReviewStyle(request.style.lower())
        cite_style = CitationStyle(request.citation_style.lower())

        # Generate review
        review = lit_review_generator.generate_review(
            papers=papers,
            research_question=request.research_question,
            style=review_style,
            citation_style=cite_style,
            include_methodology=request.include_methodology,
            include_discussion=request.include_discussion,
            min_papers_per_theme=request.min_papers_per_theme
        )

        # Get review ID
        review_id = list(lit_review_generator.reviews.keys())[-1]

        return {
            "status": "success",
            "review_id": review_id,
            "review": {
                "title": review.title,
                "research_question": review.research_question,
                "total_papers": review.total_papers,
                "word_count": review.word_count,
                "citation_count": review.citation_count,
                "themes_identified": review.themes_identified,
                "date_range": f"{review.date_range[0]}-{review.date_range[1]}",
                "style": review.style.value,
                "citation_style": review.citation_style.value,
                "generated_at": review.generated_at.isoformat()
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid style: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating literature review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lit-review/{review_id}")
async def get_literature_review(review_id: str):
    """Get a generated literature review.

    Args:
        review_id: Review ID

    Returns:
        Complete literature review
    """
    try:
        review = lit_review_generator.get_review(review_id)

        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        return {
            "review_id": review_id,
            "title": review.title,
            "research_question": review.research_question,
            "abstract": review.abstract,
            "introduction": {
                "title": review.introduction.title,
                "content": review.introduction.content,
                "word_count": review.introduction.word_count
            },
            "methodology": {
                "title": review.methodology.title,
                "content": review.methodology.content,
                "word_count": review.methodology.word_count
            } if review.methodology else None,
            "body_sections": [
                {
                    "title": section.title,
                    "content": section.content,
                    "word_count": section.word_count,
                    "citations_count": len(section.citations)
                }
                for section in review.body_sections
            ],
            "discussion": {
                "title": review.discussion.title,
                "content": review.discussion.content,
                "word_count": review.discussion.word_count
            } if review.discussion else None,
            "conclusion": {
                "title": review.conclusion.title,
                "content": review.conclusion.content,
                "word_count": review.conclusion.word_count
            },
            "future_work": {
                "title": review.future_work.title,
                "content": review.future_work.content,
                "word_count": review.future_work.word_count
            },
            "bibliography": review.bibliography,
            "stats": {
                "total_papers": review.total_papers,
                "word_count": review.word_count,
                "citation_count": review.citation_count,
                "themes_identified": review.themes_identified,
                "date_range": review.date_range
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lit-review/{review_id}/export")
async def export_literature_review(review_id: str, format: str = "markdown"):
    """Export literature review in various formats.

    Args:
        review_id: Review ID
        format: Export format (markdown, latex, json)

    Returns:
        Exported review file
    """
    try:
        review = lit_review_generator.get_review(review_id)

        if not review:
            raise HTTPException(status_code=404, detail="Review not found")

        # Create exports directory
        os.makedirs("./data/lit_reviews", exist_ok=True)

        if format == "markdown":
            content = lit_review_generator.export_to_markdown(review)
            file_path = f"./data/lit_reviews/{review_id}.md"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return FileResponse(
                file_path,
                media_type="text/markdown",
                filename=f"literature_review_{review_id}.md"
            )

        elif format == "latex":
            content = lit_review_generator.export_to_latex(review)
            file_path = f"./data/lit_reviews/{review_id}.tex"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return FileResponse(
                file_path,
                media_type="application/x-latex",
                filename=f"literature_review_{review_id}.tex"
            )

        elif format == "json":
            import json
            file_path = f"./data/lit_reviews/{review_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": review.title,
                    "abstract": review.abstract,
                    "sections": [
                        {"title": s.title, "content": s.content}
                        for s in [review.introduction] + review.body_sections + [review.conclusion]
                    ],
                    "bibliography": review.bibliography
                }, f, indent=2)

            return FileResponse(
                file_path,
                media_type="application/json",
                filename=f"literature_review_{review_id}.json"
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lit-review/list")
async def list_literature_reviews():
    """Get list of all generated reviews.

    Returns:
        List of reviews
    """
    try:
        reviews = []
        for review_id, review in lit_review_generator.reviews.items():
            reviews.append({
                "review_id": review_id,
                "title": review.title,
                "research_question": review.research_question,
                "total_papers": review.total_papers,
                "word_count": review.word_count,
                "themes": review.themes_identified,
                "generated_at": review.generated_at.isoformat(),
                "style": review.style.value
            })

        return {
            "count": len(reviews),
            "reviews": sorted(reviews, key=lambda x: x['generated_at'], reverse=True)
        }

    except Exception as e:
        logger.error(f"Error listing reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/lit-review/{review_id}")
async def delete_literature_review(review_id: str):
    """Delete a literature review.

    Args:
        review_id: Review ID

    Returns:
        Success message
    """
    try:
        if review_id not in lit_review_generator.reviews:
            raise HTTPException(status_code=404, detail="Review not found")

        del lit_review_generator.reviews[review_id]

        return {
            "status": "success",
            "message": f"Review {review_id} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GRANT MATCHER ENDPOINTS
# ============================================================================

class GrantMatchRequest(BaseModel):
    """Request to find matching grants"""
    research_gap: str
    keywords: List[str]
    min_match_score: float = 0.3
    max_results: int = 10


class ProposalOutlineRequest(BaseModel):
    """Request to generate proposal outline"""
    grant_id: str
    research_question: str
    research_gap: str
    methodology: Optional[str] = None


@app.post("/api/grants/match")
async def find_matching_grants(request: GrantMatchRequest):
    """Find grants matching research gap."""
    try:
        matches = grant_matcher.find_matching_grants(
            research_gap=request.research_gap,
            keywords=request.keywords,
            min_match_score=request.min_match_score,
            max_results=request.max_results
        )

        return {
            "matches_found": len(matches),
            "matches": [
                {
                    "grant_id": m.grant.grant_id,
                    "title": m.grant.title,
                    "agency": m.grant.agency.value,
                    "funding_max": m.grant.funding_amount_max,
                    "deadline": m.grant.deadline.isoformat(),
                    "days_left": m.days_until_deadline,
                    "urgency": m.urgency,
                    "match_score": round(m.match_score, 3),
                    "matched_keywords": m.matched_keywords,
                    "recommendations": m.recommendations
                }
                for m in matches
            ]
        }
    except Exception as e:
        logger.error(f"Error matching grants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/grants/list")
async def list_active_grants(agency: Optional[str] = None):
    """List active grants."""
    try:
        agency_enum = GrantAgency(agency.lower()) if agency else None
        grants = grant_matcher.list_active_grants(agency=agency_enum)

        return {
            "total": len(grants),
            "grants": [
                {
                    "grant_id": g.grant_id,
                    "title": g.title,
                    "agency": g.agency.value,
                    "funding_max": g.funding_amount_max,
                    "deadline": g.deadline.isoformat(),
                    "keywords": g.keywords[:5]
                }
                for g in grants
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/grants/generate-proposal")
async def generate_proposal_outline(request: ProposalOutlineRequest):
    """Generate proposal outline."""
    try:
        outline = grant_matcher.generate_proposal_outline(
            grant_id=request.grant_id,
            research_question=request.research_question,
            research_gap=request.research_gap,
            methodology=request.methodology
        )

        return {
            "status": "success",
            "proposal": {
                "grant_title": outline.grant_title,
                "project_summary": outline.project_summary,
                "sections": outline.project_description,
                "timeline": outline.timeline,
                "budget_total": sum(outline.budget.values()),
                "word_count": outline.word_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Cross-Domain Knowledge Transfer Endpoints
# =====================================================

class FindAnalogiesRequest(BaseModel):
    problem: str
    target_domain: str
    source_domains: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

class MapConceptsRequest(BaseModel):
    source_domain: str
    target_domain: str
    concepts: List[str]

class GenerateResearchProposalRequest(BaseModel):
    transfer_id: str
    title: Optional[str] = None
    budget_amount: Optional[int] = None


@app.post("/api/domain-transfer/find-analogies")
async def find_domain_analogies(request: FindAnalogiesRequest):
    """
    Find analogies from other scientific domains that could solve a problem.

    Example:
    {
        "problem": "Need to optimize neural network training speed",
        "target_domain": "computer_science",
        "source_domains": ["physics", "biology"],
        "keywords": ["optimization", "convergence", "gradient"]
    }
    """
    try:
        # Convert string domains to Domain enums
        target_domain = Domain(request.target_domain.lower())
        source_domains = None
        if request.source_domains:
            source_domains = [Domain(d.lower()) for d in request.source_domains]

        # Find analogies
        transfers = domain_transfer.find_analogies(
            problem=request.problem,
            target_domain=target_domain,
            source_domains=source_domains,
            keywords=request.keywords
        )

        # Convert to serializable format
        result_transfers = []
        for transfer in transfers:
            result_transfers.append({
                "transfer_id": transfer.transfer_id,
                "source_domain": transfer.source_domain.value,
                "target_domain": transfer.target_domain.value,
                "source_problem": transfer.source_problem,
                "source_solution": transfer.source_solution,
                "adapted_solution": transfer.adapted_solution,
                "confidence_score": transfer.confidence_score,
                "concept_mappings": [
                    {
                        "source_concept": cm.source_concept,
                        "target_concept": cm.target_concept,
                        "mapping_strength": cm.mapping_strength,
                        "mapping_type": cm.mapping_type
                    } for cm in transfer.concept_mappings
                ],
                "key_papers": transfer.key_papers,
                "potential_challenges": transfer.potential_challenges,
                "implementation_steps": transfer.implementation_steps
            })

        return {
            "status": "success",
            "analogies_found": len(result_transfers),
            "transfers": result_transfers,
            "message": f"Found {len(result_transfers)} cross-domain analogies"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/domain-transfer/domains")
async def list_available_domains():
    """
    List all available scientific domains for transfer.

    Returns available domains and their key concepts.
    """
    try:
        domains_info = []
        for domain in Domain:
            knowledge = domain_transfer.domain_knowledge.get(domain, {})
            domains_info.append({
                "domain": domain.value,
                "domain_name": domain.value.replace("_", " ").title(),
                "key_concepts": knowledge.get("concepts", [])[:10],  # Top 10 concepts
                "problem_types": list(knowledge.get("problems", {}).keys()),
                "method_count": len(knowledge.get("methods", []))
            })

        return {
            "status": "success",
            "total_domains": len(domains_info),
            "domains": domains_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/domain-transfer/historical-transfers")
async def get_historical_transfers(source_domain: Optional[str] = None, target_domain: Optional[str] = None):
    """
    Get historical examples of successful cross-domain knowledge transfers.

    Query params:
    - source_domain: Filter by source domain (optional)
    - target_domain: Filter by target domain (optional)
    """
    try:
        transfers = []
        for transfer_name, transfer_data in domain_transfer.successful_transfers.items():
            # Apply filters if provided
            if source_domain and transfer_data["source_domain"].value != source_domain.lower():
                continue
            if target_domain and transfer_data["target_domain"].value != target_domain.lower():
                continue

            transfers.append({
                "name": transfer_name.replace("_", " ").title(),
                "source_domain": transfer_data["source_domain"].value,
                "target_domain": transfer_data["target_domain"].value,
                "source_concept": transfer_data["source_concept"],
                "application": transfer_data["application"],
                "impact": transfer_data.get("impact", "High"),
                "key_papers": transfer_data.get("key_papers", []),
                "year": transfer_data.get("year")
            })

        return {
            "status": "success",
            "transfers_found": len(transfers),
            "historical_transfers": transfers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/domain-transfer/map-concepts")
async def map_concepts_between_domains(request: MapConceptsRequest):
    """
    Map concepts from one domain to another.

    Example:
    {
        "source_domain": "biology",
        "target_domain": "computer_science",
        "concepts": ["natural selection", "mutation", "adaptation"]
    }
    """
    try:
        source_domain = Domain(request.source_domain.lower())
        target_domain = Domain(request.target_domain.lower())

        concept_maps = domain_transfer.map_concepts(
            source_domain=source_domain,
            target_domain=target_domain,
            concepts=request.concepts
        )

        # Convert to serializable format
        mappings = []
        for cm in concept_maps:
            mappings.append({
                "source_concept": cm.source_concept,
                "target_concept": cm.target_concept,
                "mapping_strength": cm.mapping_strength,
                "mapping_type": cm.mapping_type,
                "explanation": f"The concept of '{cm.source_concept}' in {source_domain.value} "
                              f"maps to '{cm.target_concept}' in {target_domain.value} "
                              f"with {cm.mapping_strength:.0%} confidence ({cm.mapping_type} mapping)"
            })

        return {
            "status": "success",
            "source_domain": source_domain.value,
            "target_domain": target_domain.value,
            "mappings_found": len(mappings),
            "concept_mappings": mappings
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/domain-transfer/concept-mappings")
async def list_concept_mappings(source_domain: Optional[str] = None, target_domain: Optional[str] = None):
    """
    List all available concept mappings between domains.

    Query params:
    - source_domain: Filter by source domain (optional)
    - target_domain: Filter by target domain (optional)
    """
    try:
        mappings = []
        for cm in domain_transfer.concept_mappings:
            # Apply filters if provided
            if source_domain and cm.source_domain.value != source_domain.lower():
                continue
            if target_domain and cm.target_domain.value != target_domain.lower():
                continue

            mappings.append({
                "source_domain": cm.source_domain.value,
                "target_domain": cm.target_domain.value,
                "source_concept": cm.source_concept,
                "target_concept": cm.target_concept,
                "mapping_strength": cm.mapping_strength,
                "mapping_type": cm.mapping_type
            })

        return {
            "status": "success",
            "total_mappings": len(mappings),
            "concept_mappings": mappings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/domain-transfer/generate-proposal")
async def generate_research_proposal_from_transfer(request: GenerateResearchProposalRequest):
    """
    Generate a research proposal based on a cross-domain knowledge transfer.

    Example:
    {
        "transfer_id": "transfer_abc123",
        "title": "Applying Quantum Annealing to Machine Learning Optimization",
        "budget_amount": 500000
    }
    """
    try:
        proposal = domain_transfer.generate_research_proposal(
            transfer_id=request.transfer_id,
            title=request.title,
            budget_amount=request.budget_amount
        )

        return {
            "status": "success",
            "proposal_id": proposal.proposal_id,
            "proposal": {
                "title": proposal.title,
                "transfer_id": proposal.transfer_id,
                "abstract": proposal.abstract,
                "introduction": proposal.introduction,
                "background": {
                    "source_domain_context": proposal.source_domain_context,
                    "target_domain_context": proposal.target_domain_context,
                    "transfer_rationale": proposal.transfer_rationale
                },
                "methodology": proposal.methodology,
                "expected_outcomes": proposal.expected_outcomes,
                "timeline": proposal.timeline,
                "budget": {
                    "total": proposal.budget_amount,
                    "breakdown": proposal.budget_breakdown
                },
                "references": proposal.references,
                "generated_at": proposal.generated_at.isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/domain-transfer/transfers/{transfer_id}")
async def get_transfer_details(transfer_id: str):
    """
    Get detailed information about a specific cross-domain transfer.
    """
    try:
        transfer = domain_transfer.get_transfer(transfer_id)
        if not transfer:
            raise HTTPException(status_code=404, detail=f"Transfer {transfer_id} not found")

        return {
            "status": "success",
            "transfer": {
                "transfer_id": transfer.transfer_id,
                "source_domain": transfer.source_domain.value,
                "target_domain": transfer.target_domain.value,
                "source_problem": transfer.source_problem,
                "source_solution": transfer.source_solution,
                "adapted_solution": transfer.adapted_solution,
                "confidence_score": transfer.confidence_score,
                "concept_mappings": [
                    {
                        "source_concept": cm.source_concept,
                        "target_concept": cm.target_concept,
                        "mapping_strength": cm.mapping_strength,
                        "mapping_type": cm.mapping_type
                    } for cm in transfer.concept_mappings
                ],
                "key_papers": transfer.key_papers,
                "potential_challenges": transfer.potential_challenges,
                "implementation_steps": transfer.implementation_steps,
                "created_at": transfer.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# CrewAI Multi-Agent Orchestration Endpoints
# =====================================================

class ResearchCrewRequest(BaseModel):
    topic: str
    max_papers: int = 10
    depth: str = "comprehensive"


class AnalysisCrewRequest(BaseModel):
    paper_ids: List[str]
    focus_area: Optional[str] = None


class DiscoveryCrewRequest(BaseModel):
    research_area: str
    challenges: List[str]
    target_domain: str = "computer_science"


@app.post("/api/crew/research")
async def run_research_crew(request: ResearchCrewRequest, background_tasks: BackgroundTasks):
    """
    Execute research crew for comprehensive topic analysis.

    The research crew coordinates multiple AI agents to:
    1. Find relevant papers
    2. Analyze each paper deeply
    3. Synthesize findings into a coherent report

    This is a multi-agent orchestrated workflow using CrewAI.

    Example:
    {
        "topic": "transformer models in NLP",
        "max_papers": 10,
        "depth": "comprehensive"
    }
    """
    try:
        # Execute crew (this may take a while)
        result = research_crew.research_topic(
            topic=request.topic,
            max_papers=request.max_papers,
            depth=request.depth
        )

        return {
            "status": "success",
            "crew_type": "research",
            "result": result,
            "message": f"Research crew analyzed {request.max_papers} papers on '{request.topic}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {str(e)}")


@app.post("/api/crew/analysis")
async def run_analysis_crew(request: AnalysisCrewRequest):
    """
    Execute analysis crew for knowledge graph and gap discovery.

    The analysis crew coordinates agents to:
    1. Build knowledge graph from papers
    2. Discover research gaps
    3. Generate strategic insights

    Example:
    {
        "paper_ids": ["paper1", "paper2", "paper3"],
        "focus_area": "deep learning optimization"
    }
    """
    try:
        # Get papers from cache
        papers = []
        for paper_id in request.paper_ids:
            if paper_id in paper_cache:
                papers.append({
                    "paper_id": paper_id,
                    "title": paper_cache[paper_id].title,
                    "abstract": paper_cache[paper_id].abstract,
                    "authors": paper_cache[paper_id].authors
                })

        if not papers:
            raise HTTPException(status_code=400, detail="No valid papers found")

        # Execute crew
        result = analysis_crew.analyze_research_landscape(
            papers=papers,
            focus_area=request.focus_area
        )

        return {
            "status": "success",
            "crew_type": "analysis",
            "result": result,
            "message": f"Analysis crew processed {len(papers)} papers"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {str(e)}")


@app.post("/api/crew/discovery")
async def run_discovery_crew(request: DiscoveryCrewRequest):
    """
    Execute discovery crew for breakthrough prediction.

    The discovery crew coordinates agents to:
    1. Generate novel hypotheses
    2. Perform causal analysis
    3. Find cross-domain solutions
    4. Predict breakthroughs

    Example:
    {
        "research_area": "neural architecture search",
        "challenges": [
            "search space too large",
            "evaluation too expensive",
            "transfer learning limited"
        ],
        "target_domain": "computer_science"
    }
    """
    try:
        # Execute crew
        result = discovery_crew.discover_breakthroughs(
            research_area=request.research_area,
            current_challenges=request.challenges,
            target_domain=request.target_domain
        )

        return {
            "status": "success",
            "crew_type": "discovery",
            "result": result,
            "message": f"Discovery crew analyzed {len(request.challenges)} challenges"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew execution failed: {str(e)}")


@app.get("/api/crew/status")
async def get_crew_status():
    """
    Get status of all CrewAI crews.

    Returns information about available crews and their capabilities.
    """
    return {
        "status": "success",
        "crews": {
            "research_crew": {
                "description": "Comprehensive research topic analysis",
                "agents": ["search_specialist", "analyst", "synthesizer"],
                "capabilities": [
                    "Paper discovery",
                    "Deep analysis",
                    "Synthesis and reporting"
                ]
            },
            "analysis_crew": {
                "description": "Knowledge graph and gap discovery",
                "agents": ["kg_builder", "gap_analyst", "reasoning_expert"],
                "capabilities": [
                    "Knowledge graph construction",
                    "Gap discovery",
                    "Strategic insights"
                ]
            },
            "discovery_crew": {
                "description": "Breakthrough prediction and innovation",
                "agents": ["hypothesis_generator", "causal_analyst", "innovation_catalyst"],
                "capabilities": [
                    "Hypothesis generation",
                    "Causal reasoning",
                    "Cross-domain innovation"
                ]
            }
        },
        "framework": "CrewAI",
        "llm_backend": "Anthropic Claude" if research_crew.use_anthropic else "OpenAI GPT-4"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
