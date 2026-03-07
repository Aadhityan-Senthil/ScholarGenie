# ScholarGenie Quick Start Guide

## Current Status
✅ Backend API running at http://localhost:8000
✅ Using 100% free local models (no OpenAI needed)
✅ 19 AI agents initialized

## How to Use ScholarGenie

### Option 1: CLI (Command Line) - Fastest Way

Open a new terminal and run:

```bash
# Activate virtual environment
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie
venv\Scripts\activate

# Search for papers on a topic
python scripts\scholargenie-cli.py search "quantum computing" --max-results 5

# Ingest a specific paper and create presentation
python scripts\scholargenie-cli.py ingest --arxiv-id 1706.03762 --index
python scripts\scholargenie-cli.py generate-pptx --arxiv-id 1706.03762 --output transformer_presentation.pptx
```

### Option 2: Streamlit Web UI - Visual Interface

Open a new terminal and run:

```bash
cd C:\Users\aadhi\Desktop\Projects\ScholarGenie
venv\Scripts\activate
cd demo
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Example Workflow

### Research "Transformer Neural Networks"

**Using CLI:**
```bash
# 1. Search for papers
python scripts\scholargenie-cli.py search "transformer neural networks" --max-results 10

# 2. Pick a paper and ingest it (use arXiv ID from results)
python scripts\scholargenie-cli.py ingest --arxiv-id 1706.03762 --index

# 3. Generate presentation
python scripts\scholargenie-cli.py generate-pptx --arxiv-id 1706.03762 --output attention_is_all_you_need.pptx

# 4. Generate markdown report
python scripts\scholargenie-cli.py generate-report --arxiv-id 1706.03762 --output report.md
```

**Using Streamlit:**
1. Go to "🔍 Search Papers" tab
2. Enter "transformer neural networks"
3. Click "Search"
4. Select a paper and click "Ingest"
5. Go to "📊 My Papers" to see ingested papers
6. Generate presentations or reports

## What ScholarGenie Does

1. **Finds Papers** - Searches arXiv and Semantic Scholar
2. **Downloads PDFs** - Gets free open access papers via Unpaywall
3. **Parses PDFs** - Extracts text, figures, tables with GROBID
4. **Summarizes** - Creates TL;DR, section summaries using local AI
5. **Extracts Insights** - Finds methods, datasets, key results
6. **Creates Presentations** - Auto-generates PowerPoint slides
7. **Generates Reports** - Creates markdown/PDF reports
8. **Semantic Search** - Stores papers in vector database for AI search

## Notes

- GROBID service (for PDF parsing) needs to be running: `docker run -d -p 8070:8070 lfoppiano/grobid:0.7.3`
- First time running may download ML models (~500MB)
- Everything runs locally on CPU (no cloud API costs)
