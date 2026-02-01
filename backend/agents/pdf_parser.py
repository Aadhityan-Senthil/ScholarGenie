"""PDF parsing agent using GROBID and PyMuPDF fallback."""

import os
import logging
import re
from typing import Dict, Any, List, Optional
import yaml
import httpx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from pathlib import Path

from backend.utils.metadata import PaperMetadata, Author, Section, Reference, Figure

logger = logging.getLogger(__name__)


class PDFParserAgent:
    """Agent for parsing PDFs and extracting structured content."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize PDF parser agent.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.parser_config = self.config["pdf_parser"]
        self.grobid_url = os.getenv("GROBID_URL", self.config["apis"]["grobid"]["url"])
        self.timeout = self.config["apis"]["grobid"]["timeout"]

    def parse_pdf(self, pdf_path: str, paper_id: Optional[str] = None) -> PaperMetadata:
        """Parse a PDF file and extract structured content.

        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper identifier

        Returns:
            Parsed paper metadata
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Try GROBID first
        if self.parser_config["primary"] == "grobid":
            try:
                return self._parse_with_grobid(pdf_path, paper_id)
            except Exception as e:
                logger.warning(f"GROBID parsing failed: {e}. Falling back to PyMuPDF.")

        # Fallback to PyMuPDF
        return self._parse_with_pymupdf(pdf_path, paper_id)

    def _parse_with_grobid(self, pdf_path: str, paper_id: Optional[str] = None) -> PaperMetadata:
        """Parse PDF using GROBID service.

        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper identifier

        Returns:
            Parsed paper metadata
        """
        logger.info(f"Parsing PDF with GROBID: {pdf_path}")

        with open(pdf_path, 'rb') as f:
            files = {'input': f}

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files=files,
                    data={
                        "consolidateHeader": "1",
                        "consolidateCitations": "1",
                        "includeRawCitations": "1",
                        "includeRawAffiliations": "1"
                    }
                )
                response.raise_for_status()
                tei_xml = response.text

        # Parse TEI XML
        return self._parse_tei_xml(tei_xml, pdf_path, paper_id)

    def _parse_tei_xml(
        self,
        tei_xml: str,
        pdf_path: str,
        paper_id: Optional[str] = None
    ) -> PaperMetadata:
        """Parse GROBID TEI XML output.

        Args:
            tei_xml: TEI XML string
            pdf_path: Original PDF path
            paper_id: Optional paper identifier

        Returns:
            Parsed paper metadata
        """
        soup = BeautifulSoup(tei_xml, 'xml')

        # Extract title
        title_elem = soup.find('title', type='main')
        title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"

        # Extract authors
        authors = []
        for author_elem in soup.find_all('author'):
            persname = author_elem.find('persName')
            if persname:
                forename = persname.find('forename')
                surname = persname.find('surname')

                name_parts = []
                if forename:
                    name_parts.append(forename.get_text(strip=True))
                if surname:
                    name_parts.append(surname.get_text(strip=True))

                name = ' '.join(name_parts) if name_parts else "Unknown"

                # Extract affiliation
                affiliation_elem = author_elem.find('affiliation')
                affiliation = None
                if affiliation_elem:
                    org = affiliation_elem.find('orgName')
                    if org:
                        affiliation = org.get_text(strip=True)

                authors.append(Author(name=name, affiliation=affiliation))

        # Extract abstract
        abstract_elem = soup.find('abstract')
        abstract = None
        if abstract_elem:
            abstract = abstract_elem.get_text(strip=True, separator=' ')

        # Extract sections
        sections = []
        body = soup.find('body')

        if body:
            for div in body.find_all('div', recursive=False):
                head = div.find('head')
                section_title = head.get_text(strip=True) if head else "Section"

                # Extract paragraphs
                paragraphs = []
                for p in div.find_all('p'):
                    text = p.get_text(strip=True, separator=' ')
                    if text:
                        paragraphs.append(text)

                content = '\n\n'.join(paragraphs)

                if content:
                    sections.append(Section(
                        title=section_title,
                        content=content,
                        level=1
                    ))

        # Extract references
        references = []
        bibl_list = soup.find('listBibl')
        if bibl_list:
            for bibl in bibl_list.find_all('biblStruct'):
                ref_title_elem = bibl.find('title', type='main')
                ref_title = ref_title_elem.get_text(strip=True) if ref_title_elem else ""

                # Extract reference authors
                ref_authors = []
                for ref_author in bibl.find_all('author'):
                    persname = ref_author.find('persName')
                    if persname:
                        surname = persname.find('surname')
                        if surname:
                            ref_authors.append(surname.get_text(strip=True))

                # Extract year
                ref_year = None
                date_elem = bibl.find('date')
                if date_elem and date_elem.get('when'):
                    try:
                        ref_year = int(date_elem['when'][:4])
                    except:
                        pass

                # Extract DOI
                ref_doi = None
                idno = bibl.find('idno', type='DOI')
                if idno:
                    ref_doi = idno.get_text(strip=True)

                if ref_title:
                    references.append(Reference(
                        title=ref_title,
                        authors=ref_authors,
                        year=ref_year,
                        doi=ref_doi
                    ))

        # Extract figures (captions)
        figures = []
        for i, figure_elem in enumerate(soup.find_all('figure')):
            figdesc = figure_elem.find('figDesc')
            if figdesc:
                caption = figdesc.get_text(strip=True)
                fig_id = figure_elem.get('xml:id', f"fig_{i}")

                figures.append(Figure(
                    id=fig_id,
                    caption=caption
                ))

        # Generate paper ID if not provided
        if not paper_id:
            import uuid
            paper_id = str(uuid.uuid4())

        return PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            figures=figures,
            pdf_path=pdf_path,
            source="grobid"
        )

    def _parse_with_pymupdf(self, pdf_path: str, paper_id: Optional[str] = None) -> PaperMetadata:
        """Parse PDF using PyMuPDF (fallback method).

        Args:
            pdf_path: Path to PDF file
            paper_id: Optional paper identifier

        Returns:
            Parsed paper metadata
        """
        logger.info(f"Parsing PDF with PyMuPDF: {pdf_path}")

        doc = fitz.open(pdf_path)

        # Extract metadata
        metadata = doc.metadata
        title = metadata.get("title", "Unknown Title")
        if not title or title == "Unknown Title":
            # Try to extract from first page
            first_page = doc[0]
            text = first_page.get_text()
            lines = text.split('\n')
            if lines:
                title = lines[0].strip()

        # Extract authors from metadata
        authors = []
        author_str = metadata.get("author", "")
        if author_str:
            for name in author_str.split(';'):
                authors.append(Author(name=name.strip()))

        # Extract full text
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        doc.close()

        # Simple section extraction using heuristics
        sections = self._extract_sections_heuristic(full_text)

        # Try to extract abstract
        abstract = self._extract_abstract_heuristic(full_text)

        # Generate paper ID if not provided
        if not paper_id:
            import uuid
            paper_id = str(uuid.uuid4())

        return PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            pdf_path=pdf_path,
            source="pymupdf"
        )

    def _extract_abstract_heuristic(self, text: str) -> Optional[str]:
        """Extract abstract using simple heuristics.

        Args:
            text: Full text

        Returns:
            Abstract text if found
        """
        # Look for "Abstract" section
        pattern = r'(?i)abstract\s*\n+(.*?)(?:\n\s*\n|\n(?:introduction|1\.|keywords))'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            abstract = match.group(1).strip()
            # Clean up
            abstract = re.sub(r'\s+', ' ', abstract)
            return abstract

        return None

    def _extract_sections_heuristic(self, text: str) -> List[Section]:
        """Extract sections using simple heuristics.

        Args:
            text: Full text

        Returns:
            List of sections
        """
        sections = []

        # Common section patterns
        section_pattern = r'^(?:\d+\.?\s+)?([A-Z][A-Za-z\s]+)\n'

        # Split by potential section headers
        lines = text.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            # Check if line looks like a section header
            if re.match(section_pattern, line):
                # Save previous section
                if current_section and current_content:
                    sections.append(Section(
                        title=current_section,
                        content='\n'.join(current_content).strip(),
                        level=1
                    ))

                current_section = line.strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # Add last section
        if current_section and current_content:
            sections.append(Section(
                title=current_section,
                content='\n'.join(current_content).strip(),
                level=1
            ))

        # If no sections found, create one big section
        if not sections:
            sections.append(Section(
                title="Full Text",
                content=text,
                level=1
            ))

        return sections

    def download_and_parse(
        self,
        pdf_url: str,
        save_dir: str = "./downloads",
        paper_id: Optional[str] = None
    ) -> PaperMetadata:
        """Download PDF and parse it.

        Args:
            pdf_url: URL to PDF
            save_dir: Directory to save downloaded PDF
            paper_id: Optional paper identifier

        Returns:
            Parsed paper metadata
        """
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename
        if not paper_id:
            import uuid
            paper_id = str(uuid.uuid4())

        pdf_path = os.path.join(save_dir, f"{paper_id}.pdf")

        # Download PDF
        logger.info(f"Downloading PDF from {pdf_url}")
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                f.write(response.content)

        logger.info(f"PDF downloaded to {pdf_path}")

        # Parse
        return self.parse_pdf(pdf_path, paper_id)
