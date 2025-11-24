# File: app/utils/pdf_extractor.py
"""
Utility untuk mengekstrak dan memecah teks dari file PDF dengan optimasi chunk filtering.
"""
from pathlib import Path
import re
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Impor dari struktur baru
from app.config.settings import settings
from app.core.exceptions import PDFExtractionError
from app.core.logger import app_logger
from app.utils.text_preprocessing import text_preprocessor

class PDFExtractor:
    """Utility for extracting and splitting text from PDF files with quality filtering."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
        preprocess_text: bool = True
    ):
        self.chunk_size = chunk_size or getattr(settings, 'rag_chunk_size', 1000)
        self.chunk_overlap = chunk_overlap or getattr(settings, 'rag_chunk_overlap', 200)
        self.min_chunk_size = min_chunk_size or max(50, int(self.chunk_size * 0.05))
        self.preprocess_text = preprocess_text

        # Optimized separators untuk PDF documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # length_function=len
            # separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Regex untuk detect base64/hex encoding patterns
        self.image_encoding_pattern = re.compile(
            r'^[A-Za-z0-9+/=]{100,}$|'  # Base64 (min 100 chars untuk meaningful detection)
            r'^[0-9a-fA-F]{100,}$|'      # Hex encoding
            r'AAAB[A-Za-z0-9+/=]+',      # Common PDF image encoding prefix
            re.MULTILINE
        )
    
    def _is_image_encoding(self, text: str) -> bool:
        """
        Detect apakah chunk adalah image encoding (bukan actual content).
        
        Return True jika chunk suspicious = mostly encoding/noise.
        """
        # Skip chunks yang terlalu short
        if len(text) < 100:
            return False
        
        # Skip chunks yang mostly whitespace
        if len(text.strip()) < 50:
            return False
        
        # Hitung ratio dari encoding-like characters
        # Base64: A-Z, a-z, 0-9, +, /, =
        # Hex: 0-9, a-f, A-F
        suspicious_chars = sum(
            1 for c in text 
            if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
        )
        
        ratio = suspicious_chars / len(text)
        
        # Jika > 80% adalah base64/hex-like chars, probably image encoding
        if ratio > 0.85:
            return True
        
        # Exact regex match
        if self.image_encoding_pattern.search(text[:200]):  # Check first 200 chars
            return True
        
        return False
    
    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Filter chunks: 
        - Terlalu kecil (noise)
        - Adalah image encoding (tidak useful)
        - Mostly whitespace
        """
        before_count = len(chunks)
        filtered_chunks = []
        
        skipped_reasons = {
            "too_small": 0,
            "image_encoding": 0,
            "empty": 0
        }
        
        for chunk in chunks:
            text = chunk.page_content.strip()
            
            # Skip completely empty
            if not text:
                skipped_reasons["empty"] += 1
                continue
            
            # Skip image encoding
            if self._is_image_encoding(text):
                skipped_reasons["image_encoding"] += 1
                app_logger.debug(
                    f"Skipped image encoding chunk (page {chunk.metadata.get('page', '?')}): "
                    f"{text[:50]}..."
                )
                continue
            
            # Skip too small
            if len(text) < self.min_chunk_size:
                skipped_reasons["too_small"] += 1
                continue
            
            filtered_chunks.append(chunk)
        
        # Log summary
        total_skipped = sum(skipped_reasons.values())
        if total_skipped > 0:
            app_logger.info(
                f"Filtered {total_skipped}/{before_count} chunks: "
                f"too_small={skipped_reasons['too_small']}, "
                f"image_encoding={skipped_reasons['image_encoding']}, "
                f"empty={skipped_reasons['empty']}"
            )
        
        return filtered_chunks
    
    def _analyze_chunks(self, chunks: List[Document]) -> dict:
        """
        Analisis distribution chunk size untuk monitoring.
        Berguna untuk debugging splitting strategy.
        """
        if not chunks:
            return {}
        
        sizes = [len(c.page_content) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) // len(sizes),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "median_size": sorted(sizes)[len(sizes) // 2],
            "target_size": self.chunk_size
        }
    
    def extract(self, file_path: Path, original_filename: str = None) -> List[Document]:
        """
        Extract dan split text dari PDF dengan preprocessing dan filtering.
        
        Args:
            file_path: Path ke PDF file
            original_filename: Display name untuk logging (opsional)
            
        Returns:
            List[Document]: Chunks dengan metadata lengkap
        """
        try:
            display_filename = original_filename or file_path.name
            app_logger.info(
                f"Mengekstrak teks dari {display_filename} "
                f"(chunk_size: {self.chunk_size}, preprocessing: {self.preprocess_text})"
            )

            # 1. Load PDF
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()

            if not pages:
                raise PDFExtractionError(
                    "Tidak ada halaman yang diekstrak dari PDF",
                    details={"file": str(file_path)}
                )

            # 2. Apply text preprocessing jika enabled
            if self.preprocess_text:
                for page in pages:
                    original_content = page.page_content
                    cleaned_content = text_preprocessor.preprocess_text(original_content)
                    page.page_content = cleaned_content

                    # Track preprocessing stats
                    page.metadata.update({
                        "preprocessed": True,
                        "original_length": len(original_content),
                        "processed_length": len(cleaned_content),
                        "filename": display_filename
                    })

            # 3. Split documents
            chunks = self.text_splitter.split_documents(pages)

            if not chunks:
                raise PDFExtractionError(
                    "Tidak ada chunk teks yang dihasilkan dari PDF",
                    details={"file": str(file_path), "pages": len(pages)}
                )

            # 4. Filter out chunks yang terlalu kecil
            chunks = self._filter_chunks(chunks)

            if not chunks:
                raise PDFExtractionError(
                    "Tidak ada chunks valid (semua filtered)",
                    details={"file": str(file_path)}
                )

            # 5. Ensure semua chunks punya filename
            for chunk in chunks:
                chunk.metadata['filename'] = display_filename

            # 6. Analyze dan log hasil
            analysis = self._analyze_chunks(chunks)
            app_logger.info(
                f"Ekstrak {len(pages)} halaman → {analysis['total_chunks']} chunks "
                f"(avg: {analysis['avg_size']} chars, "
                f"range: {analysis['min_size']}-{analysis['max_size']})"
            )

            return chunks

        except PDFExtractionError:
            raise
        except Exception as e:
            app_logger.error(f"Error mengekstrak PDF {file_path}: {e}")
            raise PDFExtractionError(
                f"Gagal mengekstrak teks dari PDF: {e}",
                details={"file": str(file_path), "error": str(e)}
            )

# Global instance
pdf_extractor = PDFExtractor()