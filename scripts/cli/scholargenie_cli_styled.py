#!/usr/bin/env python
"""ScholarGenie - Modern Styled CLI Interface."""

import sys
import os
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init(autoreset=True)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.agents.paper_finder import PaperFinderAgent

# ASCII Art Logo
LOGO = f"""
{Fore.CYAN}  _______________________________________________________________
{Fore.CYAN} |                                                               |
{Fore.CYAN} |  {Fore.MAGENTA}>>> {Fore.CYAN}SCHOLAR{Fore.MAGENTA}GENIE {Fore.YELLOW}v1.0                                {Fore.CYAN}|
{Fore.CYAN} |  {Fore.WHITE}AI-Powered Research Paper Discovery & Analysis       {Fore.CYAN}|
{Fore.CYAN} |_______________________________________________________________|
{Style.RESET_ALL}
{Fore.WHITE}  ================================================================
{Fore.GREEN}  [*] {Fore.WHITE}Autonomous Multi-Agent Research Assistant
{Fore.GREEN}  [*] {Fore.WHITE}Free & Open Source - No API Keys Required
{Fore.WHITE}  ================================================================
"""

def print_header():
    """Print the styled header."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print(LOGO)
    print(f"{Fore.CYAN}[{Fore.GREEN}+{Fore.CYAN}] Backend API: {Fore.GREEN}http://localhost:8000")
    print(f"{Fore.CYAN}[{Fore.GREEN}+{Fore.CYAN}] Mode: {Fore.YELLOW}Local AI (Free, No API Keys)")
    print(f"{Fore.CYAN}[{Fore.GREEN}+{Fore.CYAN}] Status: {Fore.GREEN}Ready")
    print(f"{Fore.WHITE}{'-' * 79}\n")

def print_searching(query):
    """Print searching animation."""
    print(f"{Fore.CYAN}[{Fore.YELLOW}*{Fore.CYAN}] {Fore.WHITE}Searching for: {Fore.YELLOW}'{query}'")
    print(f"{Fore.CYAN}[{Fore.YELLOW}*{Fore.CYAN}] {Fore.WHITE}Querying arXiv and Semantic Scholar...")
    print()

def print_results(results):
    """Print search results in styled format."""
    print(f"{Fore.GREEN}[{Fore.CYAN}+{Fore.GREEN}] {Fore.WHITE}Found {Fore.CYAN}{len(results)}{Fore.WHITE} papers\n")

    for i, paper in enumerate(results, 1):
        # Title
        title = paper.get('title', 'Untitled')
        print(f"{Fore.MAGENTA}+-- Paper #{i}")
        print(f"{Fore.MAGENTA}|")
        print(f"{Fore.MAGENTA}+-- {Fore.WHITE}{title}")

        # Authors
        authors = paper.get('authors', [])
        if authors:
            author_names = [a.get('name', 'Unknown') for a in authors[:3]]
            author_str = ', '.join(author_names)
            if len(authors) > 3:
                author_str += f" +{len(authors) - 3} more"
            print(f"{Fore.MAGENTA}|   {Fore.CYAN}Authors: {Fore.WHITE}{author_str}")

        # Year & Source
        year = paper.get('year', 'N/A')
        source = paper.get('source', 'N/A').upper()
        print(f"{Fore.MAGENTA}|   {Fore.CYAN}Year: {Fore.YELLOW}{year}  {Fore.CYAN}Source: {Fore.GREEN}{source}")

        # arXiv ID
        if paper.get('arxiv_id'):
            arxiv_id = paper.get('arxiv_id')
            print(f"{Fore.MAGENTA}|   {Fore.CYAN}arXiv ID: {Fore.YELLOW}{arxiv_id}")

        # Abstract preview
        abstract = paper.get('abstract', '')
        if abstract:
            preview = abstract[:150] + '...' if len(abstract) > 150 else abstract
            print(f"{Fore.MAGENTA}|   {Fore.CYAN}Abstract: {Fore.WHITE}{preview}")

        print(f"{Fore.MAGENTA}+{'-' * 77}\n")

def print_footer():
    """Print footer with options."""
    print(f"{Fore.WHITE}{'=' * 79}")
    print(f"{Fore.CYAN}Next Steps:")
    print(f"  {Fore.GREEN}1.{Fore.WHITE} Run full CLI: {Fore.YELLOW}venv\\Scripts\\activate {Fore.WHITE}&&{Fore.YELLOW} python scripts\\scholargenie-cli.py")
    print(f"  {Fore.GREEN}2.{Fore.WHITE} Start Web UI: {Fore.YELLOW}cd demo {Fore.WHITE}&&{Fore.YELLOW} streamlit run streamlit_app.py")
    print(f"  {Fore.GREEN}3.{Fore.WHITE} API Docs: {Fore.CYAN}http://localhost:8000/docs")
    print(f"{Fore.WHITE}{'=' * 79}")

def main():
    """Main CLI entry point."""
    try:
        print_header()

        # Search query
        query = "transformer neural networks"
        print_searching(query)

        # Execute search
        finder = PaperFinderAgent()
        results = finder.search(query, max_results=3)

        # Print results
        print_results(results)

        # Footer
        print_footer()

    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}[!] Error: {Fore.WHITE}{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
