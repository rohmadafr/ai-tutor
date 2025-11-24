from typing import Optional, Dict, Any


class TutorServiceException(Exception):
    """Base exception for tutor service."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class CacheException(TutorServiceException):
    """Exception for cache-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CACHE_ERROR", **kwargs)


class RAGException(TutorServiceException):
    """Exception for RAG-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RAG_ERROR", **kwargs)


class LLMException(TutorServiceException):
    """Exception for LLM-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="LLM_ERROR", **kwargs)


class EmbeddingException(TutorServiceException):
    """Exception for embedding-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="EMBEDDING_ERROR", **kwargs)


class RedisException(TutorServiceException):
    """Exception for Redis-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="REDIS_ERROR", **kwargs)


class PDFExtractionError(TutorServiceException):
    """Exception for PDF extraction-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PDF_EXTRACTION_ERROR", **kwargs)


class FileProcessingError(TutorServiceException):
    """Exception for file processing-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FILE_PROCESSING_ERROR", **kwargs)