"""
PDF parsing utilities for basketball rules documents.
Extracts text and creates structured chunks with metadata.
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


class RulesPDFParser:
    """
    Parses basketball rules PDF files and creates structured chunks.

    Supports hierarchical chunking with metadata extraction for:
    - Rule type (FIBA/NBA)
    - Chapter/Article information
    - Page numbers
    """

    def __init__(self, pdf_path: Path, rule_type: str):
        """
        Initialize the parser.

        Args:
            pdf_path: Path to the PDF file
            rule_type: Type of rules ("FIBA" or "NBA")
        """
        self.pdf_path = pdf_path
        self.rule_type = rule_type.upper()
        self.reader = None

    def load_pdf(self) -> None:
        """Load the PDF file using pypdf."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        self.reader = PdfReader(str(self.pdf_path))

    def extract_text_from_page(self, page_number: int) -> str:
        """
        Extract text from a specific page.

        Args:
            page_number: Page number (0-indexed)

        Returns:
            Extracted text from the page
        """
        if not self.reader:
            raise ValueError("PDF not loaded. Call load_pdf() first.")

        if page_number >= len(self.reader.pages):
            raise ValueError(f"Page number {page_number} out of range.")

        page = self.reader.pages[page_number]
        return page.extract_text()

    def extract_all_text(self) -> List[Dict[str, Any]]:
        """
        Extract text from all pages with metadata.

        Returns:
            List of dictionaries containing page text and metadata
        """
        if not self.reader:
            self.load_pdf()

        pages_data = []
        for page_num, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if text.strip():  # Only include non-empty pages
                pages_data.append(
                    {
                        "page_number": page_num + 1,  # 1-indexed for user display
                        "text": text,
                    }
                )

        return pages_data

    def create_chunks(
        self, max_chunk_size: int = 1000, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Create text chunks from the PDF with overlapping windows.

        Args:
            max_chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of chunks with metadata
        """
        if not self.reader:
            self.load_pdf()

        chunks = []
        chunk_id = 0

        for page_num, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue

            # Split text into sentences for better chunk boundaries
            sentences = re.split(r"(?<=[.!?])\s+", text)
            current_chunk = ""
            current_chunk_sentences = []

            for sentence in sentences:
                # Check if adding this sentence would exceed max size
                if (
                    len(current_chunk) + len(sentence) > max_chunk_size
                    and current_chunk
                ):
                    # Save current chunk
                    chunks.append(
                        self._create_chunk_metadata(
                            chunk_id=f"{self.rule_type.lower()}_chunk_{chunk_id}",
                            content=current_chunk.strip(),
                            page_number=page_num + 1,
                        )
                    )
                    chunk_id += 1

                    # Start new chunk with overlap from previous sentences
                    if overlap > 0:
                        overlap_sents = []
                        char_count = 0
                        for s in reversed(current_chunk_sentences):
                            char_count += len(s)
                            overlap_sents.append(s)
                            if char_count >= overlap:
                                break
                        overlap_sents.reverse()
                        current_chunk = " ".join(overlap_sents) + " "
                        current_chunk_sentences = overlap_sents
                    else:
                        current_chunk = ""
                        current_chunk_sentences = []

                current_chunk += sentence + " "
                current_chunk_sentences.append(sentence)

            # Add remaining chunk if any
            if current_chunk.strip():
                chunks.append(
                    self._create_chunk_metadata(
                        chunk_id=f"{self.rule_type.lower()}_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        page_number=page_num + 1,
                    )
                )
                chunk_id += 1

        return chunks

    def create_article_based_chunks(self) -> List[Dict[str, Any]]:
        """
        Create chunks based on article/rule numbers (advanced parsing).

        This attempts to identify article headers and create chunks per article.
        Pattern matching is simplified and may need adjustment
        based on actual PDF format.

        Returns:
            List of chunks organized by articles
        """
        if not self.reader:
            self.load_pdf()

        chunks = []
        chunk_id = 0

        # Common patterns for article headers in basketball rules
        # Example: "Article 25", "Art. 33", "Rule 4", etc.
        article_pattern = re.compile(r"(?:Article|Art\.?|Rule)\s+(\d+)", re.IGNORECASE)

        for page_num, page in enumerate(self.reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue

            # Find all article matches in this page
            matches = list(article_pattern.finditer(text))

            if not matches:
                # No article headers found, treat as regular chunk
                chunks.append(
                    self._create_chunk_metadata(
                        chunk_id=f"{self.rule_type.lower()}_chunk_{chunk_id}",
                        content=text.strip(),
                        page_number=page_num + 1,
                    )
                )
                chunk_id += 1
                continue

            # Capture text before the first article header
            first_match_start = matches[0].start()
            if first_match_start > 0:
                pre_text = text[:first_match_start].strip()
                if pre_text:
                    chunks.append(
                        self._create_chunk_metadata(
                            chunk_id=f"{self.rule_type.lower()}_pre_{chunk_id}",
                            content=pre_text,
                            page_number=page_num + 1,
                        )
                    )
                    chunk_id += 1

            # Split text by article boundaries
            for i, match in enumerate(matches):
                article_num = match.group(1)
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                article_text = text[start_pos:end_pos].strip()

                chunks.append(
                    self._create_chunk_metadata(
                        chunk_id=f"{self.rule_type.lower()}_art{article_num}_{chunk_id}",
                        content=article_text,
                        page_number=page_num + 1,
                        article=f"Art {article_num}",
                    )
                )
                chunk_id += 1

        return chunks

    def _create_chunk_metadata(
        self,
        chunk_id: str,
        content: str,
        page_number: int,
        article: str = None,
        clause: str = None,
    ) -> Dict[str, Any]:
        """
        Create a chunk dictionary with metadata.

        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content of the chunk
            page_number: Source page number
            article: Article number (optional)
            clause: Clause number (optional)

        Returns:
            Dictionary with chunk data and metadata
        """
        return {
            "chunk_id": chunk_id,
            "rule_type": self.rule_type,
            "content": content,
            "page_number": page_number,
            "article": article or "N/A",
            "clause": clause or "N/A",
        }


def parse_rules_pdf(
    pdf_path: Path, rule_type: str, chunk_method: str = "sliding_window"
) -> List[Dict[str, Any]]:
    """
    Convenience function to parse a rules PDF file.

    Args:
        pdf_path: Path to the PDF file
        rule_type: Type of rules ("FIBA" or "NBA")
        chunk_method: Chunking method ("sliding_window" or "article_based")

    Returns:
        List of chunks with metadata
    """
    parser = RulesPDFParser(pdf_path, rule_type)

    if chunk_method == "article_based":
        return parser.create_article_based_chunks()
    else:
        return parser.create_chunks()
