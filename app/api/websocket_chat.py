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


# @router.websocket("/ws/chat-lcel")
# async def websocket_chat_lcel(
#     websocket: WebSocket,
#     user_id: str = Query(..., description="User ID for the connection"),
#     course_id: Optional[str] = Query(None, description="Optional course ID for context")
# ):
#     """
#     WebSocket endpoint for LCEL streaming chat with query parameters.
#     Usage: ws://localhost:8000/ws/chat-lcel?user_id=test123&course_id=course456
#     """
#     try:
#         # Accept WebSocket connection immediately
#         await websocket.accept()

#         # Connect to WebSocket manager
#         await ws_manager.connect(websocket, user_id)

#         # Send welcome message
#         await ws_manager.send_message(user_id, {
#             "type": "connection",
#             "status": "connected",
#             "user_id": user_id,
#             "course_id": course_id,
#             "timestamp": time.time(),
#             "message": "WebSocket connected. Ready for LCEL streaming chat!",
#             "config": {
#                 "model": settings.openai_model_comprehensive,
#                 "rag_threshold": settings.rag_distance_threshold,
#                 "streaming": True
#             }
#         })

#         # Main message loop
#         while True:
#             # Receive message from client
#             data = await websocket.receive_text()
#             message = json.loads(data)

#             api_logger.info(f"📥 WebSocket message from {user_id}: {message.get('type', 'unknown')}")

#             # Handle different message types
#             message_type = message.get("type", "chat_request")

#             if message_type == "chat_request":
#                 await handle_chat_request(user_id, course_id, message)
#             elif message_type == "ping":
#                 await ws_manager.send_message(user_id, {
#                     "type": "pong",
#                     "timestamp": time.time()
#                 })
#             elif message_type == "disconnect":
#                 break
#             else:
#                 await ws_manager.send_message(user_id, {
#                     "type": "error",
#                     "error": f"Unknown message type: {message_type}",
#                     "timestamp": time.time()
#                 })

#     except WebSocketDisconnect:
#         api_logger.info(f"WebSocket disconnected normally: user_id={user_id}")
#     except Exception as e:
#         api_logger.error(f"WebSocket error for {user_id}: {e}")
#         # Send error message if still connected
#         if user_id:
#             await ws_manager.send_message(user_id, {
#                 "type": "error",
#                 "error": str(e),
#                 "timestamp": time.time()
#             })
#     finally:
#         # Clean up connection
#         if user_id:
#             ws_manager.disconnect(user_id)


# async def handle_chat_request(user_id: str, course_id: Optional[str], message: Dict[str, Any]):
#     """
#     Handle chat request message and stream LCEL response
#     """
#     try:
#         query = message.get("query", "").strip()

#         if not query:
#             await ws_manager.send_message(user_id, {
#                 "type": "error",
#                 "error": "Query cannot be empty",
#                 "timestamp": time.time()
#             })
#             return

#         # Get chat service (sync function)
#         chat_service = get_chat_service()

#         # Send processing started message
#         await ws_manager.send_message(user_id, {
#             "type": "chat_started",
#             "query": query,
#             "timestamp": time.time(),
#             "metadata": {
#                 "course_id": course_id,
#                 "rag_threshold": settings.rag_distance_threshold,
#                 "method": "lcel_streaming"
#             }
#         })

#         # Start timing
#         start_time = time.time()

#         # Check cache first
#         cached_result = await chat_service.cache_service.query(query, user_id, course_id)

#         if cached_result is not None:
#             # Cache hit - send complete response
#             response_time = (time.time() - start_time) * 1000
#             await ws_manager.send_message(user_id, {
#                 "type": "chat_response",
#                 "response": cached_result,
#                 "source": "cache",
#                 "cached": True,
#                 "sources": [],
#                 "latency_ms": round(response_time, 2),
#                 "context_quality": "cached",
#                 "timestamp": time.time(),
#                 "done": True
#             })
#             return

#         # Cache miss - stream LCEL RAG response
#         await ws_manager.send_message(user_id, {
#             "type": "chat_streaming",
#             "message": "Cache miss. Generating response with LCEL RAG...",
#             "timestamp": time.time()
#         })

#         # Stream LCEL response
#         full_response = ""
#         async for chunk in chat_service.chat_stream_lcel(
#             query=query,
#             user_id=user_id,
#             course_id=course_id
#         ):
#             if chunk and chunk.strip():
#                 full_response += chunk
#                 await ws_manager.send_message(user_id, {
#                     "type": "chat_chunk",
#                     "chunk": chunk,
#                     "timestamp": time.time()
#                 })

#                 # Small delay to prevent overwhelming
#                 await asyncio.sleep(0.01)

#         # Send final message
#         response_time = (time.time() - start_time) * 1000
#         await ws_manager.send_message(user_id, {
#             "type": "chat_complete",
#             "response": full_response,
#             "source": "rag",
#             "cached": False,
#             "latency_ms": round(response_time, 2),
#             "timestamp": time.time(),
#             "done": True
#         })

#     except Exception as e:
#         api_logger.error(f"Chat request failed for {user_id}: {e}")
#         await ws_manager.send_message(user_id, {
#             "type": "error",
#             "error": str(e),
#             "timestamp": time.time()
#         })


# @router.get("/ws/status")
# async def websocket_status() -> Dict[str, Any]:
#     """Get WebSocket connection status and statistics"""
#     return {
#         "websocket_status": "active",
#         "active_connections": ws_manager.get_connection_count(),
#         "connected_users": list(ws_manager.active_connections.keys()),
#         "endpoint": "/ws/chat-lcel",
#         "message_types": [
#             "chat_request",
#             "ping",
#             "disconnect"
#         ],
#         "response_types": [
#             "connection",
#             "chat_started",
#             "chat_chunk",
#             "chat_complete",
#             "error",
#             "pong"
#         ],
#         "supported_features": {
#             "streaming": True,
#             "lcel_patterns": True,
#             "cache_integration": True,
#             "bidirectional": True,
#             "threshold_control": True
#         }
#     }


@router.websocket("/stream")
async def websocket_stream(
    websocket: WebSocket,
    user_id: str = Query(..., description="User ID for the connection"),
    course_id: Optional[str] = Query(None, description="Optional course ID for context")
):
    """
    WebSocket endpoint for SimpleChatService streaming chat with real-time database integration
    Usage: ws://localhost:8000/ws/stream?user_id=test123&course_id=course456
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


async def handle_chat_request_stream(user_id: str, course_id: Optional[str], message: Dict[str, Any], chat_service: SimpleChatService):
    """
    Handle chat request message and stream SimpleChatService response
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
        api_logger.info(f"   message: {message}")

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

        # Stream response from SimpleChatService
        full_response = ""
        chunks_count = 0

        async for chunk in chat_service.chat_with_database_stream(
            query=query,
            user_id=user_id,
            course_id=course_id,
            chatroom_id=chatroom_id,
            use_personalization=use_personalization
        ):
            if chunk and chunk.strip():
                full_response += chunk
                chunks_count += 1

                # Send content chunk
                await ws_manager.send_message(user_id, {
                    "type": "chat_chunk",
                    "chunk": chunk,
                    "chunk_index": chunks_count,
                    "timestamp": time.time()
                })

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

        # Send final completion message
        response_time = (time.time() - start_time) * 1000

        await ws_manager.send_message(user_id, {
            "type": "chat_complete",
            "response": full_response,
            "chunks_count": chunks_count,
            "response_length": len(full_response),
            "latency_ms": round(response_time, 2),
            "timestamp": time.time(),
            "done": True
        })

    except Exception as e:
        api_logger.error(f"Chat request failed for {user_id}: {e}")
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
            "chat_complete",
            "error",
            "pong"
        ],
        "supported_features": {
            "streaming": True,
            "database_integration": True,
            "cache_integration": True,
            "token_tracking": True,
            "bidirectional": True,
            "personalization": True
        }
    }