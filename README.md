# 🎓 ScholarGenie

> **Your Autonomous AI Research Assistant** - Research 10x faster with 19 AI agents working for you.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![100% Free](https://img.shields.io/badge/Pricing-FREE-green)](https://github.com/Aadhityan-Senthil/ScholarGenie)

---

## ⚡ Quick Start (3 Ways)

### 1️⃣ Beautiful CLI (Fastest - 2 Minutes)
```bash
# Windows
start_cli.bat

# Or manually
python scholargenie_v2.py
```
**No setup needed!** Just run and start researching. Works offline after first model download.

### 2️⃣ Full System with Docker (Recommended - Production)
```bash
# Windows
start_backend.bat

# Or manually
docker-compose up -d
```
**Includes:**
- FastAPI Backend (port 8000) - All 19 agents
- GROBID PDF Parser (port 8070) - Advanced PDF extraction
- Streamlit Demo UI (port 8501) - Quick testing

Visit:
- **API Docs**: http://localhost:8000/docs
- **Demo UI**: http://localhost:8501

### 3️⃣ Web Interface (Next.js Frontend)
```bash
cd frontend
npm install
npm run dev
```
Visit **http://localhost:3000** - Modern, responsive UI

---

## 🚀 What Can ScholarGenie Do?

### Paper Discovery → Analysis → Generation

```
1. Search: "quantum computing in drug discovery"
           ↓
2. Finds 100+ relevant papers across arXiv, Semantic Scholar
           ↓
3. Downloads & parses PDFs automatically
           ↓
4. Extracts: methods, datasets, results, citations
           ↓
5. Generates: PowerPoint presentations, research reports
           ↓
6. Identifies: Research gaps & opportunities
```

**All in under 2 minutes. 100% free. No API keys.**

---

## ✨ Features That Make Us Better

### vs Elicit ($10-42/month)
- ✅ **FREE** vs Paid
- ✅ **19 AI Agents** vs Basic GPT-4
- ✅ **Auto-Presentations** (they don't have)
- ✅ **Knowledge Graphs** (they don't have)
- ✅ **100% Local** vs Cloud-only

### vs Braimium/Bohrium (Paid)
- ✅ **All Research Fields** vs Chemistry-only
- ✅ **10 Gap Discovery Methods** vs None
- ✅ **Modern UI** vs Outdated
- ✅ **Downloadable EXE** vs Web-only
- ✅ **Open Source** vs Proprietary

---

## 🎯 Key Features

### 🤖 19 Specialized AI Agents (CrewAI Multi-Agent System)

**Discovery & Retrieval (5 agents)**
1. **PaperFinder** - Multi-source search (arXiv, Semantic Scholar, PubMed, IEEE)
2. **SourceAggregator** - Intelligent result combination
3. **PDFDownloader** - Automatic full-text fetching
4. **MetadataEnricher** - Citation enrichment
5. **DuplicateDetector** - Smart deduplication

**Processing & Analysis (7 agents)**
6. **PDFParser** - GROBID-powered extraction
7. **Summarizer** - LongT5/BART multi-level summaries
8. **KeyInsightExtractor** - SciBERT + GPT-3.5 insights
9. **CitationAnalyzer** - Citation network analysis
10. **MethodologyExtractor** - Research method extraction
11. **ResultsExtractor** - Finding extraction
12. **GapSpotter** - Proprietary 6-method gap detection (DBSCAN + Louvain + RAI)

**Knowledge Management (4 agents)**
13. **GraphBuilder** - Neo4j knowledge graph construction
14. **EmbeddingGenerator** - Sentence-BERT vector embeddings
15. **SemanticSearcher** - ChromaDB vector search
16. **RelationshipMapper** - Paper relationship discovery

**Output Generation (3 agents)**
17. **Presenter** - PowerPoint generation (python-pptx)
18. **ReportGenerator** - PDF report creation
19. **VisualizationAgent** - Interactive graphs (Plotly + NetworkX)

### 📊 Auto-Generate Everything
- **PowerPoint** - Defense-ready presentations
- **Reports** - Markdown/PDF research reports
- **Literature Reviews** - Synthesize 50+ papers
- **Knowledge Graphs** - Visual connections
- **Research Proposals** - Gap-based suggestions

### 🔍 Semantic AI Search
Not just keywords - ask natural questions:
- "What papers combine transformers with reinforcement learning?"
- "Find research about CRISPR in cancer therapy"
- "Show me papers by Yoshua Bengio after 2020"

### 💡 Research Gap Discovery
10 methods to find opportunities:
- Citation gap analysis
- Methodological gaps
- Dataset gaps
- Cross-domain opportunities
- Temporal gaps
- Geographical gaps
- And 4 more!

---

## 🎨 Beautiful CLI

```
   _____      __          __            ______           _
  / ___/_____/ /_  ____  / /___ ______/ ____/__  ____  (_)__
  \__ \/ ___/ __ \/ __ \/ / __ `/ ___/ / __/ _ \/ __ \/ / _ \
 ___/ / /__/ / / / /_/ / / /_/ / /  / /_/ /  __/ / / / /  __/
/____/\___/_/ /_/\____/_/\__,_/_/   \____/\___/_/ /_/_/\___/

        Your Autonomous AI Research Assistant

┌─────────────── System Status ───────────────┐
│  ● Backend API: http://localhost:8000       │
│  ● Mode: Local AI (100% Free)               │
│  ● Models: LongT5 + Sentence Transformers   │
│  ● Status: Ready                             │
└──────────────────────────────────────────────┘
```

Features:
- ✨ ASCII art logo
- 🎨 Color-coded output
- 📊 Interactive tables
- 🎯 Progress bars
- 💬 Natural prompts

---

## 💰 100% Free Forever

### No API Keys Needed
- ❌ No OpenAI ($20/month)
- ❌ No Anthropic ($20/month)
- ❌ No Groq
- ✅ Local AI models
- ✅ Complete privacy

### What's Free?
- ✅ Unlimited searches
- ✅ Unlimited paper analysis
- ✅ Unlimited presentations
- ✅ Unlimited reports
- ✅ All features forever

---

## 📖 Quick Examples

### Example 1: Analyze Famous Paper
```bash
ScholarGenie.exe

> What do you want to research? transformer neural networks
> How many papers? 5

✓ Found 5 papers

1. Attention Is All You Need
   Year: 2017 | arXiv: 1706.03762

> Do you want to analyze a paper? Yes
> Enter paper number: 1

✓ Analysis complete!
→ Presentation saved: 1706.03762_presentation.pptx
→ Report saved: 1706.03762_report.md
```

### Example 2: Literature Review
```bash
# Search your topic
ScholarGenie.exe search "CRISPR gene editing" -n 20

# Batch process
echo 2001.12345 > papers.txt
echo 2002.67890 >> papers.txt

python scripts\scholargenie-cli.py batch-ingest papers.txt

# Generate review (via web interface)
```

### Example 3: Find Research Gaps
```bash
# Web interface
http://localhost:3000/gap-discovery

# Enter: "transformer models in healthcare"
# Get: 10+ unexplored research opportunities
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 14, React 18, TypeScript, Tailwind CSS |
| **Backend** | FastAPI (3,662 lines, 96+ endpoints), Python 3.8+ |
| **Agents** | CrewAI, LangChain, Multi-agent orchestration |
| **AI/ML** | LongT5, Sentence-BERT, BART, SciBERT, GPT-3.5 |
| **Databases** | PostgreSQL (metadata), Neo4j (graphs), Redis (cache) |
| **Vector Stores** | ChromaDB, FAISS |
| **PDF Processing** | GROBID (Docker), PyMuPDF |
| **CLI** | Rich library, Beautiful terminal UI |
| **APIs** | arXiv, Semantic Scholar, PubMed, Unpaywall, CrossRef, IEEE |
| **Deployment** | Docker Compose, Celery task queue |

---

## 📦 Installation

### Option 1: Download Executable (No Installation)
1. Download `ScholarGenie.exe` from [Releases](https://github.com/Aadhityan-Senthil/ScholarGenie/releases)
2. Double-click to run
3. Start researching!

### Option 2: Run from Source
```bash
# Clone
git clone https://github.com/Aadhityan-Senthil/ScholarGenie.git
cd ScholarGenie

# Setup Python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download models
python scripts\download_models.py

# Run
scripts\test_cli.bat
```

See [SETUP.md](SETUP.md) for detailed instructions.

---

## 📁 Project Structure

```
ScholarGenie/
├── backend/              # FastAPI (19 AI agents)
├── frontend/             # Next.js (Modern UI)
├── scripts/              # CLI tools & executable
│   ├── scholargenie.py  # Beautiful CLI
│   ├── build_exe.bat    # Create executable
│   └── test_cli.bat     # Quick start
├── docs/                 # Documentation
└── SETUP.md              # Setup guide
```

---

## 🎯 Use Cases

### PhD Students
- **Literature Review** - Synthesize 50+ papers in minutes
- **Gap Analysis** - Find dissertation opportunities
- **Defense Prep** - Auto-generate presentation slides

### Researchers
- **Stay Updated** - Monitor new papers weekly
- **Find Connections** - Discover related work
- **Quick Summaries** - TL;DR of any paper

### Industry R&D
- **Competitive Analysis** - Track competitor research
- **IP Research** - Find prior art and patents
- **Tech Scouting** - Discover emerging trends

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://langchain.com/)
- [Next.js](https://nextjs.org/)
- [Rich](https://github.com/Textualize/rich)

---

<div align="center">

### 🚀 Ready to 10x Your Research?

[Download](https://github.com/Aadhityan-Senthil/ScholarGenie/releases) · [Documentation](SETUP.md) · [Report Issue](https://github.com/Aadhityan-Senthil/ScholarGenie/issues)

**Built with ❤️ for researchers, by researchers**

</div>
