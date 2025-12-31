"""
PDF document loader.
"""
import os
from pathlib import Path
from typing import List, Generator
import logging

from ..models import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads PDF documents from a directory."""
    
    def __init__(self, docs_dir: str):
        """
        Initialize PDF loader.
        
        Args:
            docs_dir: Path to directory containing PDF files
        """
        self.docs_dir = Path(docs_dir)
        
    def load_all(self) -> List[Document]:
        """
        Load all PDF documents from the directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        for doc in self.load_generator():
            documents.append(doc)
        return documents
    
    def load_generator(self) -> Generator[Document, None, None]:
        """
        Load PDF documents as a generator for memory efficiency.
        
        Yields:
            Document objects
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf is required. Install it with: pip install pypdf")
        
        if not self.docs_dir.exists():
            logger.warning(f"Directory does not exist: {self.docs_dir}")
            return
            
        pdf_files = list(self.docs_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.docs_dir}")
        
        for pdf_path in pdf_files:
            try:
                content = self._extract_text(pdf_path)
                doc = Document(
                    name=pdf_path.name,
                    path=str(pdf_path.absolute()),
                    content=content,
                    metadata={
                        "file_size": pdf_path.stat().st_size,
                        "page_count": self._get_page_count(pdf_path)
                    }
                )
                logger.info(f"Loaded document: {pdf_path.name} ({len(content)} chars)")
                yield doc
            except Exception as e:
                logger.error(f"Failed to load {pdf_path}: {e}")
                continue
    
    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text content from a PDF file."""
        import pypdf
        
        text_parts = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file."""
        import pypdf
        
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            return len(reader.pages)
