"""
Main FastAPI application entry point.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.config.settings import settings
from app.core.logger import app_logger

from app.api.health import router as health_router
from app.api.simple_chat import router as simple_chat_router
from app.api.websocket_chat import router as websocket_router
from app.api.documents import router as documents_router
from app.core.exceptions import TutorServiceException
from app.services.simple_chat_service import SimpleChatService

# Global service instance
_simple_chat_service: SimpleChatService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    app_logger.info("Starting Tutor Services API")

    try:
        # Initialize simple chat service (clean RAG + cache)
        global _simple_chat_service
        _simple_chat_service = SimpleChatService()

        app_logger.info("Tutor Services API started successfully")
        app_logger.info("API available at http://%s:%s", settings.api_host, settings.api_port)
        app_logger.info("Health check at http://%s:%s/health", settings.api_host, settings.api_port)
        app_logger.info("Legacy chat API at http://%s:%s/chat/completion", settings.api_host, settings.api_port)
        app_logger.info("LCEL chat API at http://%s:%s/chat/completion-lcel", settings.api_host, settings.api_port)
        app_logger.info("Document upload API at http://%s:%s/documents/upload", settings.api_host, settings.api_port)
        app_logger.info("Document search API at http://%s:%s/documents/search", settings.api_host, settings.api_port)
        app_logger.info("WebSocket streaming at ws://%s:%s/ws/chat-lcel", settings.api_host, settings.api_port)
        app_logger.info("WebSocket status at http://%s:%s/ws/status", settings.api_host, settings.api_port)
        app_logger.info("API docs at http://%s:%s/docs", settings.api_host, settings.api_port)

        yield

    except Exception as e:
        app_logger.error("Failed to start Tutor Services API: %s", str(e))
        raise
    finally:
        # Shutdown
        app_logger.info("Tutor Services API shutdown complete")


# Create FastAPI application with lifespan
app = FastAPI(
    title="Tutor Services API",
    description="RAG Chatbot with Semantic Caching - MVP Implementation",
    version="1.0.0",
    debug=settings.api_debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simple_chat_router)
app.include_router(websocket_router)
app.include_router(documents_router)
app.include_router(health_router)


@app.exception_handler(TutorServiceException)
async def tutor_service_exception_handler(request, exc: TutorServiceException):
    """Handle custom TutorServiceException."""
    app_logger.error(
        "Tutor service error",
        error_code=getattr(exc, 'error_code', None),
        message=exc.message,
        details=getattr(exc, 'details', None),
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": getattr(exc, 'error_code', 'TUTOR_SERVICE_ERROR'),
            "message": exc.message,
            "details": getattr(exc, 'details', None)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    app_logger.error(
        "Unhandled exception",
        error=str(exc),
        exc_info=True,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred"
        }
    )

# Development server configuration
if __name__ == "__main__":
    app_logger.info("Starting Tutor Services API in development mode")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
        access_log=True
    )