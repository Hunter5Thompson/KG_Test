"""
File Processing for Knowledge Graph Extraction
Supports: PDF, DOCX, TXT, MD
"""
from typing import List, Optional
from pathlib import Path
import tempfile
from dataclasses import dataclass

# Text extraction libraries
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import markdown


@dataclass
class ProcessedFile:
    """Extracted file content"""
    filename: str
    content: str
    format: str
    char_count: int
    error: Optional[str] = None


class FileProcessor:
    """
    Extract text from various file formats
    
    Supported formats:
    - PDF (.pdf)
    - Word (.docx)
    - Text (.txt)
    - Markdown (.md)
    
    Example:
        >>> processor = FileProcessor()
        >>> result = processor.process_uploaded_file(file_bytes, "document.pdf")
        >>> print(result.content)
    """
    
    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.md'}
    
    def __init__(self):
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required libraries are available"""
        missing = []
        if PdfReader is None:
            missing.append("pypdf")
        if DocxDocument is None:
            missing.append("python-docx")
        
        if missing:
            print(f"⚠️  Optional dependencies missing: {', '.join(missing)}")
            print(f"   Install with: pip install {' '.join(missing)}")
    
    def is_supported(self, filename: str) -> bool:
        """Check if file format is supported"""
        return Path(filename).suffix.lower() in self.SUPPORTED_FORMATS
    
    def process_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str
    ) -> ProcessedFile:
        """
        Process uploaded file and extract text
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename
        
        Returns:
            ProcessedFile with extracted content
        """
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                content = self._extract_pdf(file_bytes)
            elif file_ext == '.docx':
                content = self._extract_docx(file_bytes)
            elif file_ext == '.txt':
                content = self._extract_txt(file_bytes)
            elif file_ext == '.md':
                content = self._extract_markdown(file_bytes)
            else:
                return ProcessedFile(
                    filename=filename,
                    content="",
                    format=file_ext,
                    char_count=0,
                    error=f"Unsupported format: {file_ext}"
                )
            
            return ProcessedFile(
                filename=filename,
                content=content,
                format=file_ext,
                char_count=len(content),
                error=None
            )
        
        except Exception as e:
            return ProcessedFile(
                filename=filename,
                content="",
                format=file_ext,
                char_count=0,
                error=f"Extraction failed: {str(e)}"
            )
    
    def _extract_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF"""
        if PdfReader is None:
            raise ImportError("pypdf not installed")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            reader = PdfReader(tmp_path)
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n\n".join(text_parts)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _extract_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX"""
        if DocxDocument is None:
            raise ImportError("python-docx not installed")
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            doc = DocxDocument(tmp_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def _extract_txt(self, file_bytes: bytes) -> str:
        """Extract text from TXT"""
        # Try common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Fallback: replace errors
        return file_bytes.decode('utf-8', errors='replace')
    
    def _extract_markdown(self, file_bytes: bytes) -> str:
        """Extract text from Markdown (convert to plain text)"""
        md_text = self._extract_txt(file_bytes)
        # Convert markdown to HTML then strip tags for plain text
        html = markdown.markdown(md_text)
        # Simple tag stripping (for POC purposes)
        import re
        text = re.sub(r'<[^>]+>', '', html)
        return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, question mark, or exclamation
            for punct in ['. ', '! ', '? ']:
                last_punct = text[start:end].rfind(punct)
                if last_punct > chunk_size * 0.5:  # At least 50% into chunk
                    end = start + last_punct + len(punct)
                    break
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks