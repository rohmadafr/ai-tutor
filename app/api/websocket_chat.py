"""
WebSocket API for LCEL Streaming Chat
Real-time bidirectional chat using LangChain Expression Language patterns
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Any, Optional
import json
import asyncio
import time

from ..services.simple_chat_service import SimpleChatService
from ..core.logger import api_logger
from ..config.settings import settings

router = APIRouter(prefix="/ws", tags=["websocket"])

class WebSocketManager:
    """
    Manages WebSocket connections for real-time chat streaming
    """

    def __init__(self):
        # Store multiple connections: user_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Store WebSocket connection (already accepted in main function)"""
        self.active_connections[user_id] = websocket
        api_logger.info(f"WebSocket connected: user_id={user_id}")

    def disconnect(self, user_id: str):
        """Remove WebSocket connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            api_logger.info(f"WebSocket disconnected: user_id={user_id}")

    async def send_message(self, user_id: str, message: Dict[str, Any]):
        """Send message to specific user"""
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                api_logger.error(f"Failed to send message to {user_id}: {e}")
                # Remove broken connection
                self.disconnect(user_id)
                return False
        return False

    async def broadcast(self, message: Dict[str, Any]):
        """Send message to all connected users"""
        disconnected_users = []
        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                api_logger.error(f"Failed to broadcast to {user_id}: {e}")
                disconnected_users.append(user_id)

        # Clean up broken connections
        for user_id in disconnected_users:
            self.disconnect(user_id)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global WebSocket manager
ws_manager = WebSocketManager()

# Service singleton for consistent instances
_chat_service: Optional[SimpleChatService] = None

def get_chat_service() -> SimpleChatService:
    """Get chat service instance (singleton pattern)"""
    global _chat_service
    if _chat_service is None:
        _chat_service = SimpleChatService()
    return _chat_service


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID for the connection"),
    course_id: str = Query(..., description="Required course ID for context")
):
    """
    WebSocket endpoint for SimpleChatService streaming chat with ChatResponse structure
    Usage: ws://localhost:8000/ws/stream?user_id=test123&course_id=course456

    NOTE: course_id is now REQUIRED for all connections
    
    Returns:
    - Multiple messages with type "chat_chunk" (streaming content)
    - Final message with type "chat_response" (ChatResponse structure with sources, metadata, etc.)
    """

    try:
        # Accept WebSocket connection immediately
        await websocket.accept()

        # Connect to WebSocket manager
        await ws_manager.connect(websocket, user_id)

        # Send welcome message
        await ws_manager.send_message(user_id, {
            "type": "connection",
            "status": "connected",
            "user_id": user_id,
            "course_id": course_id,
            "timestamp": time.time(),
            "message": "WebSocket connected. Ready for streaming chat!",
            "config": {
                "model_comprehensive": settings.openai_model_comprehensive,
                "model_personalized": settings.openai_model_personalized,
                "rag_threshold": settings.rag_distance_threshold,
                "streaming": True
            }
        })

        # Get chat service instance
        chat_service = get_chat_service()

        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            api_logger.info(f"📥 WebSocket message from {user_id}: {message.get('type', 'unknown')}")

            # Handle different message types
            message_type = message.get("type", "chat_request")

            if message_type == "chat_request":
                await handle_chat_request_stream(user_id, course_id, message, chat_service)
            elif message_type == "ping":
                await ws_manager.send_message(user_id, {
                    "type": "pong",
                    "timestamp": time.time()
                })
            elif message_type == "disconnect":
                break
            else:
                await ws_manager.send_message(user_id, {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}",
                    "timestamp": time.time()
                })

    except WebSocketDisconnect:
        api_logger.info(f"WebSocket disconnected normally: user_id={user_id}")
    except Exception as e:
        api_logger.error(f"WebSocket error for {user_id}: {e}")
        # Send error message if still connected
        if user_id:
            await ws_manager.send_message(user_id, {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            })
    finally:
        # Clean up connection
        if user_id:
            ws_manager.disconnect(user_id)


async def handle_chat_request_stream(
    user_id: str, 
    course_id: Optional[str], 
    message: Dict[str, Any], 
    chat_service: SimpleChatService
):
    """
    Handle chat request message and stream SimpleChatService response
    
    NEW: Returns ChatResponse structure at the end with sources, metadata, etc.
    """
    try:
        query = message.get("query", "").strip()
        use_personalization = message.get("use_personalization", False)
        chatroom_id = message.get("chatroom_id")

        # Debug logging
        api_logger.info(f"📨 Chat request received:")
        api_logger.info(f"   user_id: {user_id}")
        api_logger.info(f"   course_id: {course_id}")
        api_logger.info(f"   chatroom_id: {chatroom_id}")
        api_logger.info(f"   query: {query}")

        if not query:
            await ws_manager.send_message(user_id, {
                "type": "error",
                "error": "Query cannot be empty",
                "timestamp": time.time()
            })
            return

        # Send processing started message
        await ws_manager.send_message(user_id, {
            "type": "chat_started",
            "query": query,
            "timestamp": time.time(),
            "metadata": {
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "use_personalization": use_personalization,
                "service": "simple_chat_service"
            }
        })

        # Start timing
        start_time = time.time()

        # Collect metadata during streaming
        full_response = ""
        chunks_count = 0
        source_documents = []
        response_metadata = {}
        cached = False
        personalized = False
        model_used = settings.openai_model_comprehensive
        response_source = "rag"
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cost_usd = 0.0

        # Stream response from SimpleChatService
        async for item in chat_service.chat_with_database_stream(
            query=query,
            user_id=user_id,
            course_id=course_id,
            chatroom_id=chatroom_id,
            use_personalization=use_personalization
        ):
            # Handle both string chunks and dict items
            if isinstance(item, str):
                # OLD format: Direct string chunk
                chunk = item
                full_response += chunk
                chunks_count += 1

                # Send content chunk
                await ws_manager.send_message(user_id, {
                    "type": "chat_chunk",
                    "chunk": chunk,
                    "chunk_index": chunks_count,
                    "timestamp": time.time()
                })
            elif isinstance(item, dict) and item.get("type") == "content":
                # NEW format: Dictionary with type/data structure
                chunk = item["data"]
                full_response += chunk
                chunks_count += 1

                # Send content chunk
                await ws_manager.send_message(user_id, {
                    "type": "chat_chunk",
                    "chunk": chunk,
                    "chunk_index": chunks_count,
                    "timestamp": time.time()
                })
            elif isinstance(item, dict) and item.get("type") == "metadata":
                # Handle metadata items
                metadata = item["data"]
                response_source = metadata.get("source", "rag")
                cached = response_source.startswith("cache")
                personalized = metadata.get("response_type") == "cache_hit_personalized"
                model_used = metadata.get("model_used", settings.openai_model_comprehensive)

                # Handle source_documents - ensure it's always a list
                source_documents = metadata.get("source_documents", [])
                input_tokens = metadata.get("input_tokens", 0)
                output_tokens = metadata.get("output_tokens", 0)
                total_tokens = metadata.get("total_tokens", input_tokens + output_tokens)  # Calculate if not provided
                cost_usd = metadata.get("cost_usd", 0.0)

                # Store full metadata for final response
                response_metadata = metadata

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

            elif isinstance(item, dict) and item.get("type") == "error":
                # Handle streaming errors
                await ws_manager.send_message(user_id, {
                    "type": "error",
                    "error": item["data"],
                    "timestamp": time.time()
                })
                return

        # Calculate total response time
        response_time_ms = (time.time() - start_time) * 1000

        # Send final ChatResponse structure
        chat_response = {
            "type": "chat_response",
            "response": full_response,
            "sources": source_documents,
            "cached": cached,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            },
            "latency_ms": round(response_time_ms, 2),
            "timestamp": time.time(),
            "metadata": {
                "source": response_source,
                "source_type": response_metadata.get("source_type", "knowledge_base"),
                "response_type": response_metadata.get("response_type", "rag_response"),
                "model_used": model_used,
                "personalized": personalized,
                "course_id": course_id,
                "cost_usd": cost_usd
            },
            "response_source": response_source,
            "chunks_count": chunks_count,
            "done": True  # Signal streaming completion
        }

        # Send final response with all metadata
        await ws_manager.send_message(user_id, chat_response)

        api_logger.info(
            f"✅ Streaming complete: chunks={chunks_count}, "
            f"sources={len(source_documents)}, latency={response_time_ms:.2f}ms"
        )

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        api_logger.error(f"Chat request failed for {user_id}: {e}")
        api_logger.error(f"Full traceback: {error_trace}")

        await ws_manager.send_message(user_id, {
            "type": "error",
            "error": str(e),
            "timestamp": time.time()
        })


@router.get("/status")
async def websocket_status() -> Dict[str, Any]:
    """Get WebSocket connection status and statistics"""
    return {
        "websocket_status": "active",
        "active_connections": ws_manager.get_connection_count(),
        "connected_users": list(ws_manager.active_connections.keys()),
        "endpoint": "/ws/stream",
        "message_types": [
            "chat_request",
            "ping",
            "disconnect"
        ],
        "response_types": [
            "connection",
            "chat_started",
            "chat_chunk",
            "chat_response",
            "error",
            "pong"
        ],
        "supported_features": {
            "streaming": True,
            "database_integration": True,
            "cache_integration": True,
            "token_tracking": True,
            "bidirectional": True,
            "personalization": True,
            "sources_metadata": True
        }
    }