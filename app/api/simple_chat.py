"""
Simple Chat API using SimpleChatService
Clean, minimal API for RAG + Semantic Cache demo
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import time

from ..models.chat import ChatRequest, ChatResponse
from ..services.simple_chat_service import SimpleChatService
from ..services.unified_rag_service import UnifiedRAGService
from ..services.custom_cache_service import CustomCacheService
from ..core.logger import api_logger
from ..config.settings import settings

router = APIRouter(prefix="/chat", tags=["chat"])

# Global service instances - clean separation
_chat_service: Optional[SimpleChatService] = None
_rag_service: Optional[UnifiedRAGService] = None
_cache_service: Optional[CustomCacheService] = None


async def get_chat_service() -> SimpleChatService:
    """Get simple chat service instance"""
    global _chat_service
    if _chat_service is None:
        _chat_service = SimpleChatService()
    return _chat_service


async def get_rag_service() -> UnifiedRAGService:
    """Get RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = UnifiedRAGService()
    return _rag_service


async def get_cache_service() -> CustomCacheService:
    """Get cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CustomCacheService()
    return _cache_service


@router.post("/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """
    Simple chat completion with RAG + Semantic Cache
    Clean logic: Cache → RAG → Cache response
    """
    try:
        # Basic validation
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 10000:
            raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")

        # Get service
        chat_service = await get_chat_service()

        # Log request
        api_logger.info(
            "📥 CHAT REQUEST: query='%s...', user_id='%s', course_id='%s'",
            request.query[:50] + "..." if len(request.query) > 50 else request.query,
            request.user_id or "anonymous",
            request.course_id or "global"
        )

        start_time = time.time()

        # Process chat - SimpleChatService handles cache/RAG logic internally
        response = await chat_service.chat(
            query=request.query,
            user_id=request.user_id,
            course_id=request.course_id
        )

        # Calculate response time
        response_time = (time.time() - start_time) * 1000

        # Log response
        api_logger.info(
            "📤 CHAT RESPONSE: source='%s', cached=%s, sources_count=%d, latency_ms=%.2f",
            response.get("source", "unknown"),
            response.get("cached", False),
            len(response.get("sources", [])),
            response_time
        )

        # Create ChatResponse
        return ChatResponse(
            response=response.get("response", ""),
            sources=response.get("sources", []),
            cached=response.get("cached", False),
            token_usage={
                "input_tokens": 0,  # TODO: Implement token counting
                "output_tokens": 0,
                "total_tokens": 0
            },
            latency_ms=response_time,
            timestamp=time.time(),
            metadata={
                "cache_type": response.get("cache_type", "none"),
                "response_source": response.get("source", "unknown"),
                "model_used": "gpt-4o-mini"
            },
            response_source=f"{'cache_' if response.get('cached') else ''}{response.get('source', 'unknown')}"
        )

    except Exception as e:
        api_logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@router.delete("/cache")
async def clear_cache(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Clear semantic cache"""
    try:
        cache_service = await get_cache_service()
        cleared_count = await cache_service.clear_user_cache(user_id) if user_id else 0

        return {
            "message": f"Successfully cleared cache for {user_id or 'all users'}",
            "cleared_entries": cleared_count
        }

    except Exception as e:
        api_logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completion-lcel", response_model=ChatResponse)
async def chat_completion_lcel(request: ChatRequest) -> ChatResponse:
    """
    LCEL-powered chat completion with hybrid threshold logic.
    Same cache service, different RAG implementation using LangChain Expression Language.
    """
    try:
        # Basic validation
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if len(request.query) > 10000:
            raise HTTPException(status_code=400, detail="Query too long (max 10000 characters)")

        # Get service
        chat_service = await get_chat_service()

        # Log request
        api_logger.info(
            "📥 LCEL CHAT REQUEST: query='%s...', user_id='%s', course_id='%s'",
            request.query[:50] + "..." if len(request.query) > 50 else request.query,
            request.user_id or "anonymous",
            request.course_id or "global"
        )

        start_time = time.time()

        # Process LCEL chat (uses settings.rag_distance_threshold automatically)
        response = await chat_service.chat_lcel(
            query=request.query,
            user_id=request.user_id,
            course_id=request.course_id
        )

        # Calculate response time
        response_time = (time.time() - start_time) * 1000

        # Log response
        api_logger.info(
            "📤 LCEL CHAT RESPONSE: source='%s', method='%s', context_quality='%s', latency_ms=%.2f",
            response.get("source", "unknown"),
            response.get("method", "unknown"),
            response.get("context_quality", "unknown"),
            response_time
        )

        # Create ChatResponse with actual token usage from response if available
        token_usage = response.get("token_usage", {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        })

        return ChatResponse(
            response=response.get("response", ""),
            sources=response.get("sources", []),
            cached=response.get("cached", False),
            token_usage=token_usage,
            latency_ms=response_time,
            timestamp=time.time(),
            metadata={
                "method": response.get("method", "lcel"),
                "context_quality": response.get("context_quality", "unknown"),
                "rag_threshold": response.get("rag_threshold", 0.3),
                "model_used": settings.openai_model_comprehensive,
                "response_source": response.get("method", "lcel")
            },
            response_source=response.get("method", "lcel")
        )

    except Exception as e:
        api_logger.error(f"LCEL chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
