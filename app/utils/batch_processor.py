# File: app/utils/batch_processor.py
"""
Utility untuk memproses dokumen dalam batch berdasarkan batas token.
(Dipindah dari utils.py)
"""
from typing import List
from langchain_core.documents import Document

# Impor dari struktur baru
from app.core.logger import app_logger
from app.core.telemetry import TokenCounter

class BatchProcessor:
    """Utility for processing documents in batches based on token limits."""
    
    def __init__(self, max_tokens_per_batch=300000):
        """
        Initialize with token limit per batch.
        """
        self.max_tokens_per_batch = max_tokens_per_batch
        self.token_counter = TokenCounter()
    
    def split_into_batches(self, docs: List[Document]) -> List[List[Document]]:
        """
        Split documents into batches based on token limit.
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = self.token_counter.count_tokens(doc.page_content)
            
            if current_tokens + doc_tokens > self.max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        app_logger.info(f"Split {len(docs)} dokumen menjadi {len(batches)} batch")
        return batches


# Global instance
batch_processor = BatchProcessor()