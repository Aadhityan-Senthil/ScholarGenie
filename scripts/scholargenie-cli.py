#!/usr/bin/env python
"""ScholarGenie Command-Line Interface."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.paper_finder import PaperFinderAgent
from backend.agents.pdf_parser import PDFParserAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.extractor import ExtractorAgent
from backend.agents.presenter import PresenterAgent
from backend.utils.storage import VectorStore
from backend.utils.embeddings import EmbeddingService


def search_papers(args):
    """Search for papers."""
    print(f"🔍 Searching for: {args.query}")

    finder = PaperFinderAgent()
    results = finder.search(args.query, max_results=args.max_results)

    print(f"\n✅ Found {len(results)} papers:\n")

    for i, paper in enumerate(results, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join([a['name'] for a in paper.get('authors', [])])}")
        print(f"   Year: {paper.get('year', 'N/A')}")
        print(f"   Source: {paper.get('source', 'N/A')}")
        print(f"   Open Access: {'✅' if paper.get('is_open_access') else '❌'}")

        if args.verbose and paper.get('abstract'):
            print(f"   Abstract: {paper['abstract'][:200]}...")

        print()


def ingest_paper(args):
    """Ingest a paper."""
    print("📥 Ingesting paper...")

    finder = PaperFinderAgent()
    parser = PDFParserAgent()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # Get paper
    paper_data = None

    if args.doi:
        print(f"Fetching by DOI: {args.doi}")
        paper_data = finder.get_paper_by_doi(args.doi)
    elif args.arxiv_id:
        print(f"Fetching by arXiv ID: {args.arxiv_id}")
        paper_data = finder.get_paper_by_arxiv_id(args.arxiv_id)
    elif args.pdf_url:
        print(f"Fetching from URL: {args.pdf_url}")

    if not paper_data and not args.pdf_url:
        print("❌ Paper not found")
        return

    # Download and parse PDF
    if paper_data:
        pdf_url = paper_data.get('pdf_url')
        if not pdf_url:
            print("❌ No PDF available")
            return
        paper = parser.download_and_parse(pdf_url, paper_id=paper_data.get('paper_id'))
    else:
        paper = parser.download_and_parse(args.pdf_url)

    print(f"\n✅ Parsed: {paper.title}")
    print(f"   Authors: {', '.join([a.name for a in paper.authors])}")
    print(f"   Sections: {len(paper.sections)}")

    # Generate embeddings and store
    if args.index:
        print("\n📊 Generating embeddings...")
        full_text = paper.get_full_text()
        chunks, embeddings = embedding_service.chunk_and_embed(full_text)

        metadata = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "year": paper.year
        }

        vector_store.add_paper(paper.paper_id, chunks, embeddings, metadata)
        print(f"✅ Indexed {len(chunks)} chunks")

    print(f"\n✅ Paper ID: {paper.paper_id}")


def generate_pptx(args):
    """Generate PowerPoint presentation."""
    print(f"📊 Generating presentation for: {args.paper_id or args.doi}")

    # This requires the paper to be already ingested
    # For CLI, we'll fetch and process on the fly

    finder = PaperFinderAgent()
    parser = PDFParserAgent()
    summarizer = SummarizerAgent()
    extractor = ExtractorAgent()
    presenter = PresenterAgent()

    # Get paper
    paper_data = None

    if args.doi:
        paper_data = finder.get_paper_by_doi(args.doi)
    elif args.arxiv_id:
        paper_data = finder.get_paper_by_arxiv_id(args.arxiv_id)

    if not paper_data:
        print("❌ Paper not found")
        return

    # Parse PDF
    print("📄 Parsing PDF...")
    paper = parser.download_and_parse(paper_data['pdf_url'], paper_id=paper_data['paper_id'])

    # Summarize
    print("📝 Generating summary...")
    summary = summarizer.summarize_paper(paper)

    # Extract
    print("🔍 Extracting data...")
    extracted_data = extractor.extract(paper)

    # Generate presentation
    print("🎨 Creating presentation...")
    output_path = args.output or f"{paper.paper_id}_presentation.pptx"
    result_path = presenter.generate_pptx(paper, summary, extracted_data, output_path)

    print(f"\n✅ Presentation saved to: {result_path}")


def generate_report(args):
    """Generate report."""
    print(f"📄 Generating report for: {args.paper_id or args.doi}")

    finder = PaperFinderAgent()
    parser = PDFParserAgent()
    summarizer = SummarizerAgent()
    extractor = ExtractorAgent()
    presenter = PresenterAgent()

    # Get paper
    paper_data = None

    if args.doi:
        paper_data = finder.get_paper_by_doi(args.doi)
    elif args.arxiv_id:
        paper_data = finder.get_paper_by_arxiv_id(args.arxiv_id)

    if not paper_data:
        print("❌ Paper not found")
        return

    # Parse PDF
    print("📄 Parsing PDF...")
    paper = parser.download_and_parse(paper_data['pdf_url'], paper_id=paper_data['paper_id'])

    # Summarize
    print("📝 Generating summary...")
    summary = summarizer.summarize_paper(paper)

    # Extract
    print("🔍 Extracting data...")
    extracted_data = extractor.extract(paper)

    # Generate report
    print("📄 Creating report...")
    output_path = args.output or f"{paper.paper_id}_report.md"
    result_path = presenter.generate_markdown_report(paper, summary, extracted_data, output_path)

    print(f"\n✅ Report saved to: {result_path}")


def batch_ingest(args):
    """Batch ingest papers from file."""
    print(f"📦 Batch ingesting from: {args.file}")

    with open(args.file, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} papers to ingest\n")

    finder = PaperFinderAgent()
    parser = PDFParserAgent()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        print(f"[{i}/{len(lines)}] Processing: {line}")

        try:
            # Determine type (DOI, arXiv ID, or URL)
            if line.startswith('10.'):  # DOI
                paper_data = finder.get_paper_by_doi(line)
                if paper_data and paper_data.get('pdf_url'):
                    paper = parser.download_and_parse(paper_data['pdf_url'], paper_id=paper_data['paper_id'])
                else:
                    print(f"  ❌ No PDF found for {line}")
                    continue

            elif 'arxiv.org' in line or line.replace('.', '').isdigit():  # arXiv
                arxiv_id = line.split('/')[-1].replace('abs', '').replace('pdf', '').strip()
                paper_data = finder.get_paper_by_arxiv_id(arxiv_id)
                if paper_data:
                    paper = parser.download_and_parse(paper_data['pdf_url'], paper_id=paper_data['paper_id'])
                else:
                    print(f"  ❌ Paper not found: {line}")
                    continue

            else:  # Assume URL
                paper = parser.download_and_parse(line)

            # Index
            full_text = paper.get_full_text()
            chunks, embeddings = embedding_service.chunk_and_embed(full_text)

            metadata = {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "year": paper.year
            }

            vector_store.add_paper(paper.paper_id, chunks, embeddings, metadata)

            print(f"  ✅ Ingested: {paper.title}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Batch ingestion complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ScholarGenie - AI-Powered Scientific Paper Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for papers')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--max-results', type=int, default=10, help='Maximum results')
    search_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a paper')
    ingest_parser.add_argument('--doi', type=str, help='Paper DOI')
    ingest_parser.add_argument('--arxiv-id', type=str, help='arXiv ID')
    ingest_parser.add_argument('--pdf-url', type=str, help='Direct PDF URL')
    ingest_parser.add_argument('--index', action='store_true', help='Index in vector store')

    # Generate PPTX command
    pptx_parser = subparsers.add_parser('generate-pptx', help='Generate PowerPoint presentation')
    pptx_parser.add_argument('--paper-id', type=str, help='Paper ID')
    pptx_parser.add_argument('--doi', type=str, help='Paper DOI')
    pptx_parser.add_argument('--arxiv-id', type=str, help='arXiv ID')
    pptx_parser.add_argument('--output', type=str, help='Output file path')

    # Generate report command
    report_parser = subparsers.add_parser('generate-report', help='Generate report')
    report_parser.add_argument('--paper-id', type=str, help='Paper ID')
    report_parser.add_argument('--doi', type=str, help='Paper DOI')
    report_parser.add_argument('--arxiv-id', type=str, help='arXiv ID')
    report_parser.add_argument('--output', type=str, help='Output file path')

    # Batch ingest command
    batch_parser = subparsers.add_parser('batch-ingest', help='Batch ingest from file')
    batch_parser.add_argument('file', type=str, help='File with paper identifiers (one per line)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'search':
        search_papers(args)
    elif args.command == 'ingest':
        ingest_paper(args)
    elif args.command == 'generate-pptx':
        generate_pptx(args)
    elif args.command == 'generate-report':
        generate_report(args)
    elif args.command == 'batch-ingest':
        batch_ingest(args)


if __name__ == '__main__':
    main()
