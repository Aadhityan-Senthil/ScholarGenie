# Resources & Tools

This document provides detailed information about all the free and open-source tools used in ScholarGenie, including setup instructions, API key acquisition, and usage guidelines.

## Table of Contents

1. [PDF Processing](#pdf-processing)
2. [Paper Discovery APIs](#paper-discovery-apis)
3. [ML Models & Frameworks](#ml-models--frameworks)
4. [Vector Databases](#vector-databases)
5. [Agent Frameworks](#agent-frameworks)
6. [Presentation & Reports](#presentation--reports)
7. [Development Tools](#development-tools)

---

## PDF Processing

### GROBID

**What it is:** Machine learning library for extracting, parsing, and restructuring raw documents (especially PDFs) into structured TEI XML.

- **License:** Apache 2.0
- **Repository:** https://github.com/kermitt2/grobid
- **Documentation:** https://grobid.readthedocs.io/

**Setup:**

```bash
# Using Docker (recommended)
docker pull lfoppiano/grobid:0.7.3
docker run -d --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.7.3

# Verify it's running
curl http://localhost:8070/api/isalive
```

**Why we use it:**
- Extracts structured metadata, sections, figures, tables, references
- Better quality than simple text extraction
- Handles complex academic paper layouts

**Alternative:** PyMuPDF (automatic fallback)

---

## Paper Discovery APIs

### arXiv API

**What it is:** Free API for searching and accessing arXiv preprints.

- **License:** Free, no authentication required
- **Documentation:** https://arxiv.org/help/api/
- **Rate Limit:** 3 requests/second (enforced by etiquette)

**Setup:**

```bash
pip install arxiv
```

**Usage Example:**

```python
import arxiv

search = arxiv.Search(query="attention mechanism", max_results=10)
for result in search.results():
    print(result.title, result.pdf_url)
```

**API Key:** Not required

**Rate Limiting:** Respect the 3 req/sec limit. Our system implements automatic throttling.

---

### Semantic Scholar API

**What it is:** Free API providing access to 200M+ papers with citation graphs, abstracts, and metadata.

- **License:** Free (API key recommended)
- **Website:** https://www.semanticscholar.org/product/api
- **Documentation:** https://api.semanticscholar.org/

**Setup:**

1. **Without API Key:** 100 requests / 5 minutes
2. **With API Key:** 5,000 requests / 5 minutes

**Get API Key (Free):**

1. Visit https://www.semanticscholar.org/product/api#api-key-form
2. Fill out the form (name, email, intended use)
3. Receive key via email (usually instant)
4. Add to `.env`:

```bash
SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

**Usage Example:**

```python
import httpx

headers = {"x-api-key": "YOUR_KEY"}
response = httpx.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    params={"query": "transformers", "limit": 10},
    headers=headers
)
```

---

### Unpaywall API

**What it is:** Free service to find legal Open Access versions of research papers.

- **License:** Free, requires email in requests
- **Website:** https://unpaywall.org/products/api
- **Documentation:** https://unpaywall.org/products/api

**Setup:**

No signup required, just use your email:

```bash
# In .env
UNPAYWALL_EMAIL=your@email.com
```

**Usage Example:**

```python
import httpx

doi = "10.1038/nature12373"
email = "your@email.com"

response = httpx.get(f"https://api.unpaywall.org/v2/{doi}?email={email}")
data = response.json()

if data.get("is_oa"):
    pdf_url = data["best_oa_location"]["url_for_pdf"]
```

**Rate Limit:** 100,000 requests/day (very generous)

---

## ML Models & Frameworks

### Hugging Face Transformers

**What it is:** State-of-the-art ML library for NLP, providing access to thousands of pretrained models.

- **License:** Apache 2.0
- **Website:** https://huggingface.co/
- **Documentation:** https://huggingface.co/docs/transformers

**Setup:**

```bash
pip install transformers torch
```

---

### LongT5 (Long-Document Summarization)

**What it is:** Google's Transformer model designed for long-sequence tasks (up to 16k tokens).

- **License:** Apache 2.0
- **Paper:** https://arxiv.org/abs/2112.07916
- **Models:**
  - `google/long-t5-local-base` (220M params, fast)
  - `google/long-t5-tglobal-base` (220M params, better quality)
  - `google/long-t5-tglobal-large` (770M params, best quality, needs GPU)

**Usage Example:**

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

inputs = tokenizer("Long text here...", return_tensors="pt", max_length=4096, truncation=True)
outputs = model.generate(**inputs, max_length=512)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**Model Selection Guide:**

| Model | Size | Speed | Quality | GPU Required |
|-------|------|-------|---------|--------------|
| long-t5-local-base | 220M | Fast | Good | Optional |
| long-t5-tglobal-base | 220M | Medium | Better | Recommended |
| long-t5-tglobal-large | 770M | Slow | Best | Required (8GB+ VRAM) |

---

### Pegasus (Abstractive Summarization)

**What it is:** Google's summarization model pretrained on massive datasets.

- **License:** Apache 2.0
- **Paper:** https://arxiv.org/abs/1912.08777
- **Model:** `google/pegasus-large`

**Alternative Models:**

```python
# Scientific papers
"google/pegasus-multi_news"  # For long documents
"google/pegasus-arxiv"       # Pretrained on arXiv
"google/pegasus-pubmed"      # Pretrained on biomedical papers
```

---

### Sentence Transformers (Embeddings)

**What it is:** Framework for state-of-the-art sentence, text, and image embeddings.

- **License:** Apache 2.0
- **Repository:** https://github.com/UKPLab/sentence-transformers
- **Models:** https://www.sbert.net/docs/pretrained_models.html

**Recommended Models:**

```python
# Fast & lightweight (recommended)
"sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, 80MB

# Better quality
"sentence-transformers/all-mpnet-base-v2"  # 768 dim, 420MB

# For scientific text
"allenai/specter"  # Specialized for papers
```

**Setup:**

```bash
pip install sentence-transformers
```

---

## Vector Databases

### Chroma

**What it is:** Open-source embedding database designed for LLM applications.

- **License:** Apache 2.0
- **Website:** https://www.trychroma.com/
- **Documentation:** https://docs.trychroma.com/

**Setup:**

```bash
pip install chromadb
```

**Usage Example:**

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("papers")

collection.add(
    documents=["Text chunk 1", "Text chunk 2"],
    ids=["id1", "id2"],
    metadatas=[{"title": "Paper 1"}, {"title": "Paper 2"}]
)

results = collection.query(query_texts=["attention mechanism"], n_results=5)
```

**Why Chroma:**
- Easy to use
- Persistent storage
- Great for prototyping
- Scales to millions of vectors

---

### FAISS (Alternative)

**What it is:** Facebook AI Similarity Search - efficient similarity search library.

- **License:** MIT
- **Repository:** https://github.com/facebookresearch/faiss
- **Documentation:** https://github.com/facebookresearch/faiss/wiki

**Setup:**

```bash
pip install faiss-cpu  # CPU version
# OR
pip install faiss-gpu  # GPU version (requires CUDA)
```

**When to use FAISS:**
- Need ultra-fast search (billion-scale)
- Have GPU for acceleration
- Want more control over indexing

---

## Agent Frameworks

### LangChain

**What it is:** Framework for developing applications powered by language models.

- **License:** MIT
- **Repository:** https://github.com/langchain-ai/langchain
- **Documentation:** https://python.langchain.com/

**Setup:**

```bash
pip install langchain langchain-community
```

**Usage in ScholarGenie:**
- Agent orchestration
- Tool integration
- Prompt management

**Alternatives:**
- **LlamaIndex:** For data indexing and retrieval
- **AutoGen:** For multi-agent conversations

---

## Presentation & Reports

### python-pptx

**What it is:** Python library for creating and updating PowerPoint files.

- **License:** MIT
- **Repository:** https://github.com/scanny/python-pptx
- **Documentation:** https://python-pptx.readthedocs.io/

**Setup:**

```bash
pip install python-pptx
```

**Usage Example:**

```python
from pptx import Presentation

prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[0])
title = slide.shapes.title
title.text = "Paper Title"

prs.save("presentation.pptx")
```

---

### Markdown to PDF

**Tools:**

1. **Pandoc** (Recommended)
   - Install: https://pandoc.org/installing.html
   - Convert: `pandoc input.md -o output.pdf`

2. **WeasyPrint** (Python-native)
   ```bash
   pip install weasyprint
   ```

3. **md-to-pdf** (Node.js)
   ```bash
   npm install -g md-to-pdf
   ```

---

## Development Tools

### Testing

**pytest** - Testing framework

```bash
pip install pytest pytest-cov pytest-asyncio
```

**Run tests:**

```bash
# All tests
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend --cov-report=html

# Specific test
pytest backend/tests/test_summarizer.py -v
```

---

### Code Quality

**flake8** - Linting

```bash
pip install flake8
flake8 backend/ --max-line-length=127
```

**black** - Code formatting

```bash
pip install black
black backend/ demo/ scripts/
```

**ruff** - Fast Python linter

```bash
pip install ruff
ruff check backend/
```

---

## API Key Summary

| Service | Required? | How to Get | Rate Limit |
|---------|-----------|------------|------------|
| arXiv | No | N/A | 3 req/sec |
| Semantic Scholar | Recommended | https://www.semanticscholar.org/product/api#api-key-form | 5k/5min with key |
| Unpaywall | Email only | Just use your email | 100k/day |
| Hugging Face | Optional | https://huggingface.co/settings/tokens | For private models |

---

## Compute Requirements

### Minimum (CPU-only)

```
CPU: 4 cores
RAM: 8GB
Disk: 10GB
Model: long-t5-local-base
```

### Recommended (with GPU)

```
CPU: 8 cores
RAM: 16GB
GPU: 8GB VRAM (RTX 3070 or better)
Disk: 50GB
Model: long-t5-tglobal-large
```

### Cloud Options (Free Tier)

- **Google Colab:** Free GPU (T4)
- **Kaggle:** Free GPU (P100)
- **Hugging Face Spaces:** Free CPU/GPU hosting

---

## Additional Resources

### Learning Materials

- **LangChain Docs:** https://python.langchain.com/docs/get_started/introduction
- **Hugging Face Course:** https://huggingface.co/course
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/tutorial/
- **Streamlit Docs:** https://docs.streamlit.io/

### Academic Papers

- **GROBID:** "GROBID - Information Extraction from Scientific Publications" (https://hal.science/hal-01673305)
- **LongT5:** "LongT5: Efficient Text-To-Text Transformer for Long Sequences" (https://arxiv.org/abs/2112.07916)
- **BERT:** "BERT: Pre-training of Deep Bidirectional Transformers" (https://arxiv.org/abs/1810.04805)

### Community

- **Discord:** (Create a community server)
- **GitHub Discussions:** Report issues and feature requests
- **Stack Overflow:** Tag questions with `scholargenie`

---

## License Compatibility

All tools used are compatible with commercial use:

- Apache 2.0: GROBID, Transformers, LongT5, Chroma
- MIT: FAISS, LangChain, python-pptx
- Free APIs: arXiv, Semantic Scholar, Unpaywall (with attribution)

---

**Last Updated:** 2025-01-03
**ScholarGenie Version:** 1.0.0
