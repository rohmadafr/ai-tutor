from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat completion."""
    query: str = Field(..., description="User query/question")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    course_id: Optional[str] = Field(None, description="Course identifier for course-specific knowledge")
    chatroom_id: Optional[str] = Field(None, description="Chatroom identifier for conversation tracking")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    temperature: Optional[float] = Field(None, description="Override default temperature")
    max_tokens: Optional[int] = Field(None, description="Override max tokens")
    stream: bool = Field(default=False, description="Enable streaming response")

    # Additional control parameters
    force_refresh: bool = Field(default=False, description="Force fresh generation, skip cache")
    skip_rag: bool = Field(default=False, description="Skip RAG document retrieval")
    max_docs: Optional[int] = Field(None, description="Maximum documents to retrieve from RAG")
    max_context: Optional[int] = Field(None, description="Maximum conversation context messages")
    use_comprehensive_model: bool = Field(default=False, description="Use comprehensive model instead of efficient")
    use_personalization: bool = Field(default=False, description="Enable response personalization")


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    response: str = Field(..., description="Generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    cached: bool = Field(default=False, description="Whether response was from cache")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata including model and cost")

    # Single source indicator that covers everything
    response_source: str = Field(default="unknown", description="Response source: cache_raw, cache_personalized, knowledge_base, llm_fallback")


class MessageHistory(BaseModel):
    """Model for message history."""
    id: str = Field(..., description="Unique message identifier")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")