from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Document(BaseModel):
    """Model for a document."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    source: str = Field(..., description="Document source/file path")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Document tags")


class DocumentChunk(BaseModel):
    """Model for a chunk of a document."""
    id: str = Field(..., description="Chunk ID")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk content")
    embedding: Optional[List[float]] = Field(None, description="Chunk embedding vector")
    chunk_index: int = Field(..., description="Index of chunk in document")
    start_pos: int = Field(..., description="Start position in original document")
    end_pos: int = Field(..., description="End position in original document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    vector_distance: Optional[float] = Field(None, description="Distance score from search (lower = more similar)")
    filepath: Optional[str] = Field(None, description="File path or URL for accessing the material")


class SearchResult(BaseModel):
    """Model for search results."""
    chunk: DocumentChunk = Field(..., description="Document chunk")
    score: float = Field(..., description="Similarity score")
    document: Optional[Document] = Field(None, description="Parent document if available")

