from fastapi import APIRouter, HTTPException
from ..services.simple_chat_service import SimpleChatService
from ..core.logger import api_logger
router = APIRouter(prefix="/health", tags=["health"])


async def get_chat_service() -> SimpleChatService:
    """Dependency to get chat service instance."""
    global _chat_service
    if _chat_service is None:
        try:
            _chat_service = SimpleChatService()
        except Exception as e:
            api_logger.error("Failed to initialize chat service: %s", str(e))
            raise HTTPException(status_code=503, detail="Service unavailable")
    return _chat_service
