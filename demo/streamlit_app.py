"""Streamlit demo interface for ScholarGenie."""

import os
import streamlit as st
import requests
from datetime import datetime
import pandas as pd

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="ScholarGenie",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .paper-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e8f4f8;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📚 ScholarGenie</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Scientific Paper Analysis</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["🔍 Search Papers", "📥 Ingest Paper", "📊 My Papers", "🔎 Semantic Search", "📈 Statistics"]
    )

    st.divider()

    st.header("About")
    st.info("""
    **ScholarGenie** is a multi-agent system that:
    - Discovers papers from arXiv & Semantic Scholar
    - Parses PDFs with GROBID
    - Generates multi-level summaries
    - Extracts key insights
    - Creates presentations & reports
    """)


# Helper functions
def search_papers(query, max_results=10):
    """Search for papers."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/search",
            json={"query": query, "max_results": max_results}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching papers: {e}")
        return []


def ingest_paper(doi=None, arxiv_id=None, pdf_url=None):
    """Ingest a paper."""
    try:
        payload = {}
        if doi:
            payload["doi"] = doi
        elif arxiv_id:
            payload["arxiv_id"] = arxiv_id
        elif pdf_url:
            payload["pdf_url"] = pdf_url

        response = requests.post(
            f"{BACKEND_URL}/api/ingest",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error ingesting paper: {e}")
        return None


def get_papers():
    """Get list of ingested papers."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/papers")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching papers: {e}")
        return {"papers": [], "count": 0}


def summarize_paper(paper_id):
    """Generate summary for a paper."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/summarize",
            json={"paper_id": paper_id}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


def extract_data(paper_id):
    """Extract structured data from paper."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/extract",
            json={"paper_id": paper_id}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return None


def generate_pptx(paper_id):
    """Generate PowerPoint presentation."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/generate-pptx",
            json={"paper_id": paper_id}
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error generating presentation: {e}")
        return None


def generate_report(paper_id):
    """Generate report."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/generate-report",
            json={"paper_id": paper_id, "format": "markdown"}
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None


def semantic_search(query, top_k=10):
    """Perform semantic search."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/semantic-search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return None


def get_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stats")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        return None


# Page: Search Papers
if page == "🔍 Search Papers":
    st.header("Search Scientific Papers")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input(
            "Enter search query",
            placeholder="e.g., transformer language models"
        )

    with col2:
        max_results = st.number_input("Max results", min_value=1, max_value=50, value=10)

    if st.button("🔍 Search", type="primary"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_papers(search_query, max_results)

            if results:
                st.success(f"Found {len(results)} papers")

                for i, paper in enumerate(results, 1):
                    with st.expander(f"{i}. {paper['title']}", expanded=i <= 3):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Authors:** {', '.join(paper['authors'][:5])}")
                            if paper.get('venue'):
                                st.markdown(f"**Venue:** {paper['venue']}")
                            if paper.get('year'):
                                st.markdown(f"**Year:** {paper['year']}")

                            if paper.get('abstract'):
                                st.markdown("**Abstract:**")
                                st.text(paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract'])

                        with col2:
                            st.markdown(f"**Source:** {paper['source']}")
                            st.markdown(f"**Open Access:** {'✅' if paper['is_open_access'] else '❌'}")

                            if paper.get('pdf_url'):
                                st.markdown(f"[📄 PDF]({paper['pdf_url']})")

                        # Quick ingest button
                        if paper.get('pdf_url'):
                            if st.button(f"📥 Ingest Paper", key=f"ingest_{i}"):
                                with st.spinner("Ingesting..."):
                                    result = ingest_paper(pdf_url=paper['pdf_url'])
                                    if result:
                                        st.success(f"✅ {result['message']}")
            else:
                st.warning("No papers found")
        else:
            st.warning("Please enter a search query")


# Page: Ingest Paper
elif page == "📥 Ingest Paper":
    st.header("Ingest Paper")

    st.markdown("""
    Add a paper to your collection by providing one of:
    - DOI (Digital Object Identifier)
    - arXiv ID
    - Direct PDF URL
    """)

    ingest_method = st.radio(
        "Choose ingest method",
        ["DOI", "arXiv ID", "PDF URL"]
    )

    if ingest_method == "DOI":
        doi = st.text_input("Enter DOI", placeholder="10.48550/arXiv.1706.03762")

        if st.button("📥 Ingest by DOI", type="primary"):
            if doi:
                with st.spinner("Ingesting paper..."):
                    result = ingest_paper(doi=doi)
                    if result:
                        st.success(f"✅ {result['message']}")
                        st.json(result)
            else:
                st.warning("Please enter a DOI")

    elif ingest_method == "arXiv ID":
        arxiv_id = st.text_input("Enter arXiv ID", placeholder="1706.03762")

        if st.button("📥 Ingest by arXiv ID", type="primary"):
            if arxiv_id:
                with st.spinner("Ingesting paper..."):
                    result = ingest_paper(arxiv_id=arxiv_id)
                    if result:
                        st.success(f"✅ {result['message']}")
                        st.json(result)
            else:
                st.warning("Please enter an arXiv ID")

    else:  # PDF URL
        pdf_url = st.text_input("Enter PDF URL", placeholder="https://arxiv.org/pdf/1706.03762.pdf")

        if st.button("📥 Ingest by PDF URL", type="primary"):
            if pdf_url:
                with st.spinner("Ingesting paper..."):
                    result = ingest_paper(pdf_url=pdf_url)
                    if result:
                        st.success(f"✅ {result['message']}")
                        st.json(result)
            else:
                st.warning("Please enter a PDF URL")


# Page: My Papers
elif page == "📊 My Papers":
    st.header("My Papers")

    data = get_papers()
    papers = data.get("papers", [])

    if papers:
        st.info(f"You have {len(papers)} papers in your collection")

        # Create DataFrame
        df = pd.DataFrame(papers)

        # Display papers
        for i, paper in enumerate(papers, 1):
            with st.expander(f"{i}. {paper['title']}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Paper ID:** `{paper['paper_id']}`")
                    st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")

                with col2:
                    if paper.get('year'):
                        st.markdown(f"**Year:** {paper['year']}")
                    if paper.get('venue'):
                        st.markdown(f"**Venue:** {paper['venue']}")

                with col3:
                    st.markdown(f"**Has Summary:** {'✅' if paper['has_summary'] else '❌'}")
                    st.markdown(f"**Has Extracted Data:** {'✅' if paper['has_extracted_data'] else '❌'}")

                # Actions
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.button("📝 Summarize", key=f"sum_{i}"):
                        with st.spinner("Generating summary..."):
                            result = summarize_paper(paper['paper_id'])
                            if result:
                                st.session_state[f"summary_{paper['paper_id']}"] = result
                                st.success("Summary generated!")

                with col2:
                    if st.button("🔍 Extract Data", key=f"ext_{i}"):
                        with st.spinner("Extracting data..."):
                            result = extract_data(paper['paper_id'])
                            if result:
                                st.session_state[f"extracted_{paper['paper_id']}"] = result
                                st.success("Data extracted!")

                with col3:
                    if st.button("📊 Generate PPTX", key=f"pptx_{i}"):
                        with st.spinner("Generating presentation..."):
                            pptx_data = generate_pptx(paper['paper_id'])
                            if pptx_data:
                                st.download_button(
                                    label="⬇️ Download PPTX",
                                    data=pptx_data,
                                    file_name=f"{paper['paper_id']}.pptx",
                                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                                )

                with col4:
                    if st.button("📄 Generate Report", key=f"report_{i}"):
                        with st.spinner("Generating report..."):
                            report_data = generate_report(paper['paper_id'])
                            if report_data:
                                st.download_button(
                                    label="⬇️ Download Report",
                                    data=report_data,
                                    file_name=f"{paper['paper_id']}_report.md",
                                    mime="text/markdown"
                                )

                # Display summary if available
                if f"summary_{paper['paper_id']}" in st.session_state:
                    st.markdown("### Summary")
                    summary = st.session_state[f"summary_{paper['paper_id']}"]["summary"]

                    st.markdown(f"**TL;DR:** {summary['tldr']}")
                    st.markdown(f"**Summary:** {summary['short_summary']}")

                    if summary.get('keypoints'):
                        st.markdown("**Key Points:**")
                        for point in summary['keypoints'][:5]:
                            st.markdown(f"- {point}")

                # Display extracted data if available
                if f"extracted_{paper['paper_id']}" in st.session_state:
                    st.markdown("### Extracted Data")
                    extracted = st.session_state[f"extracted_{paper['paper_id']}"]

                    if extracted.get('methods'):
                        st.markdown(f"**Methods:** {', '.join(extracted['methods'][:5])}")

                    if extracted.get('datasets'):
                        st.markdown(f"**Datasets:** {', '.join(extracted['datasets'][:5])}")

                    if extracted.get('key_findings'):
                        st.markdown("**Key Findings:**")
                        for finding in extracted['key_findings'][:3]:
                            st.markdown(f"- {finding}")

    else:
        st.info("No papers in your collection yet. Go to 'Search Papers' or 'Ingest Paper' to add some!")


# Page: Semantic Search
elif page == "🔎 Semantic Search":
    st.header("Semantic Search")

    st.markdown("""
    Search across ingested papers using natural language queries.
    The system uses embeddings to find semantically similar content.
    """)

    query = st.text_input("Enter your query", placeholder="e.g., attention mechanisms in neural networks")

    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

    if st.button("🔎 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = semantic_search(query, top_k)

            if results and results.get("results"):
                st.success(f"Found {len(results['results'])} results")

                for i, result in enumerate(results['results'], 1):
                    with st.expander(f"Result {i} (Distance: {result['distance']:.3f})"):
                        st.markdown(f"**Text:**")
                        st.text(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

                        st.markdown(f"**Metadata:**")
                        st.json(result['metadata'])
            else:
                st.info("No results found. Make sure you have ingested some papers first!")
        else:
            st.warning("Please enter a query")


# Page: Statistics
elif page == "📈 Statistics":
    st.header("System Statistics")

    stats = get_stats()

    if stats:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Papers Ingested", stats['papers_ingested'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Papers Summarized", stats['papers_summarized'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Papers Extracted", stats['papers_extracted'])
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Vector Store Count", stats['vector_store_count'])
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown(f"**Last Updated:** {stats['timestamp']}")

        # Get papers data
        papers_data = get_papers()
        if papers_data['papers']:
            st.subheader("Papers Overview")

            df = pd.DataFrame(papers_data['papers'])

            # Year distribution
            if 'year' in df.columns:
                st.markdown("### Papers by Year")
                year_counts = df['year'].value_counts().sort_index()
                st.bar_chart(year_counts)

            # Summary status
            st.markdown("### Processing Status")
            status_data = {
                'Has Summary': df['has_summary'].sum(),
                'Has Extracted Data': df['has_extracted_data'].sum(),
                'Total Papers': len(df)
            }
            st.bar_chart(status_data)

    else:
        st.error("Unable to fetch statistics")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    ScholarGenie v1.0.0 | Built with ❤️ using LangChain, LongT5, GROBID, and Streamlit
</div>
""", unsafe_allow_html=True)
