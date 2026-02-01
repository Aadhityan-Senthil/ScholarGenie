# ScholarGenie Architecture

## Overview

ScholarGenie is a multi-agent system designed to autonomously discover, analyze, and synthesize scientific papers. The system follows a modular architecture with specialized agents, each responsible for a specific aspect of the paper processing pipeline.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │  Streamlit UI    │              │   REST API       │        │
│  │  (Web Demo)      │              │   (FastAPI)      │        │
│  └──────────────────┘              └──────────────────┘        │
└──────────────────┬──────────────────────────┬──────────────────┘
                   │                          │
┌──────────────────┴──────────────────────────┴──────────────────┐
│                      Orchestration Layer                        │
│                   (FastAPI Backend Service)                     │
│  - Request routing                                              │
│  - Task coordination                                            │
│  - Caching & state management                                   │
└──────────────────┬─────────────────────────────────────────────┘
                   │
┌──────────────────┴─────────────────────────────────────────────┐
│                        Agent Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │PaperFinder   │  │  PDFParser   │  │ Summarizer   │        │
│  │    Agent     │  │    Agent     │  │    Agent     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Extractor   │  │  Presenter   │  │  Evaluator   │        │
│  │    Agent     │  │    Agent     │  │    Agent     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────┬─────────────────────────────────────────────┘
                   │
┌──────────────────┴─────────────────────────────────────────────┐
│                      Utility Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Model        │  │ Embedding    │  │ Vector       │        │
│  │ Manager      │  │ Service      │  │ Store        │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└──────────────────┬─────────────────────────────────────────────┘
                   │
┌──────────────────┴─────────────────────────────────────────────┐
│                    External Services Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   GROBID     │  │   arXiv      │  │  Semantic    │        │
│  │   Service    │  │    API       │  │  Scholar     │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Unpaywall   │  │  Hugging     │  │   Chroma     │        │
│  │     API      │  │   Face       │  │     DB       │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Responsibilities

### 1. PaperFinder Agent

**Purpose:** Discover and retrieve scientific papers from multiple sources.

**Key Functions:**
- Search arXiv API for preprints
- Query Semantic Scholar for published papers
- Resolve Open Access PDFs via Unpaywall
- Handle rate limiting and API errors

**Inputs:**
- Search query (keywords, author names, DOI)
- Maximum results count
- Source preferences

**Outputs:**
- List of paper metadata (title, authors, abstract, PDF URL)
- Open Access status

**Technologies:**
- `arxiv` Python library
- `httpx` for HTTP requests
- `tenacity` for retry logic

---

### 2. PDFParser Agent

**Purpose:** Extract structured content from PDF files.

**Key Functions:**
- Parse PDFs using GROBID (primary)
- Fallback to PyMuPDF if GROBID unavailable
- Extract metadata, sections, figures, references
- Handle malformed PDFs gracefully

**Inputs:**
- PDF file path or URL

**Outputs:**
- `PaperMetadata` object with:
  - Title, authors, abstract
  - Structured sections
  - Figures with captions
  - References with DOIs

**Technologies:**
- GROBID (Java service) for high-fidelity extraction
- PyMuPDF (fitz) for fallback
- BeautifulSoup for TEI XML parsing

---

### 3. Summarizer Agent

**Purpose:** Generate multi-granularity summaries of papers.

**Key Functions:**
- TL;DR (1 sentence)
- Short summary (2-3 sentences)
- Full summary (200-400 words)
- Section-level summaries
- Key bullet points extraction

**Inputs:**
- `PaperMetadata` object

**Outputs:**
- `Summary` object with multiple granularities

**Technologies:**
- LongT5 / Pegasus models from Hugging Face
- Hierarchical summarization for long documents
- Chunk-and-summarize strategy

**Algorithm:**
```
For long documents:
1. Split into chunks (max 4096 tokens)
2. Summarize each chunk
3. Combine chunk summaries
4. Generate final summary from combined text
```

---

### 4. Extractor Agent

**Purpose:** Extract structured information and key insights.

**Key Functions:**
- Research questions and hypotheses
- Methodology mentions (models, datasets)
- Performance metrics
- Key findings
- Limitations and future work

**Inputs:**
- `PaperMetadata` object

**Outputs:**
- `ExtractedData` object with structured fields

**Technologies:**
- Regex patterns for entity extraction
- NLP heuristics for section identification
- Named entity recognition patterns

---

### 5. Presenter Agent

**Purpose:** Generate presentation slides and reports.

**Key Functions:**
- Create PowerPoint presentations
- Generate Markdown reports
- Format content for readability
- Include figures and references

**Inputs:**
- `PaperMetadata`, `Summary`, `ExtractedData`

**Outputs:**
- `.pptx` file (PowerPoint)
- `.md` file (Markdown)

**Slide Structure:**
1. Title slide (title, authors, venue)
2. Overview (TL;DR + summary)
3. Motivation (research question)
4. Methodology
5. Results & findings
6. Key points
7. Limitations & future work
8. References

**Technologies:**
- `python-pptx` for PowerPoint generation
- Markdown with optional PDF export (Pandoc/WeasyPrint)

---

### 6. Evaluator Agent

**Purpose:** Assess summary quality and detect issues.

**Key Functions:**
- Compute ROUGE scores
- Compute BERTScore
- Check summary length
- Detect hallucinations
- Check for redundancy

**Inputs:**
- `PaperMetadata`, `Summary`

**Outputs:**
- Evaluation metrics (ROUGE, BERTScore)
- Quality check results
- Warnings list

**Technologies:**
- `rouge-score` library
- `bert-score` library
- Custom heuristics for hallucination detection

---

## Data Flow

### Paper Ingestion Pipeline

```
1. User provides DOI/arXiv ID/PDF URL
   ↓
2. PaperFinder Agent
   - Searches APIs
   - Resolves PDF URL
   ↓
3. PDFParser Agent
   - Downloads PDF
   - Extracts structured content
   ↓
4. Embedding Service
   - Chunks text
   - Generates embeddings
   ↓
5. Vector Store
   - Indexes chunks
   - Enables semantic search
```

### Summarization Pipeline

```
1. User requests summary for paper_id
   ↓
2. Retrieve PaperMetadata from cache
   ↓
3. Summarizer Agent
   - Generate TL;DR
   - Generate short summary
   - Generate full summary
   - Extract keypoints
   ↓
4. Cache Summary object
   ↓
5. Return to user
```

### Presentation Generation Pipeline

```
1. User requests presentation for paper_id
   ↓
2. Retrieve/Generate Summary
   ↓
3. Retrieve/Generate ExtractedData
   ↓
4. Presenter Agent
   - Build slide structure
   - Format content
   - Add metadata
   ↓
5. Save .pptx file
   ↓
6. Return file to user
```

## Technology Stack

### Core Frameworks
- **FastAPI:** REST API backend
- **Streamlit:** Web UI
- **LangChain:** Agent orchestration (planned for future versions)

### ML/AI Models
- **LongT5:** Long-document summarization
- **Sentence Transformers:** Text embeddings
- **BERT:** Evaluation (BERTScore)

### Storage
- **Chroma:** Vector database
- **FAISS:** Alternative vector store
- **In-memory cache:** Development (Redis recommended for production)

### External Services
- **GROBID:** PDF-to-TEI conversion
- **arXiv API:** Paper discovery
- **Semantic Scholar API:** Paper metadata
- **Unpaywall API:** OA PDF resolution

## Scalability Considerations

### Current Architecture (v1.0)
- In-memory caching
- Single-process backend
- Suitable for: Personal use, small teams, demos

### Production Recommendations
- **Caching:** Redis or Memcached
- **Queue:** Celery + RabbitMQ for background tasks
- **Storage:** PostgreSQL for metadata, S3 for PDFs
- **Vector DB:** FAISS with Milvus or Pinecone for scale
- **Load Balancing:** Multiple backend instances
- **Model Serving:** Separate model server (TorchServe, TensorFlow Serving)

## Configuration

The system is configured via `config.yaml`:

```yaml
summarization:
  model_name: "google/long-t5-tglobal-base"
  device: "cpu"  # or "cuda"

vector_store:
  type: "chroma"
  persist_directory: "./data/chroma_db"

apis:
  grobid:
    url: "http://localhost:8070"
```

Environment variables (`.env`) override config for sensitive data:
```bash
SEMANTIC_SCHOLAR_API_KEY=xxx
UNPAYWALL_EMAIL=user@example.com
```

## Error Handling

### Strategy
1. **Graceful Degradation:** Fallback to simpler methods if advanced fails
   - GROBID fails → PyMuPDF
   - Large model OOM → Smaller model

2. **Retry Logic:** Transient failures with exponential backoff
   - API rate limits
   - Network errors

3. **User Feedback:** Clear error messages and warnings
   - "PDF not available"
   - "Summary quality warning: potential hallucinations"

## Future Enhancements

1. **Multi-language Support:** Extend to non-English papers
2. **Knowledge Graph:** Build citation networks
3. **Custom Fine-tuning:** Domain-specific summarization
4. **Collaborative Features:** Share annotations, comments
5. **Integration:** Zotero, Mendeley plugins
6. **Advanced Extraction:** Table and equation parsing
7. **Real-time Updates:** Monitor new papers on topics

---

For detailed API documentation, see the FastAPI interactive docs at `/docs` when running the backend.
