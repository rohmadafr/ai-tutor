"""
Simple Chat Service - Clean Orchestration Layer
Single responsibility: Check cache → If miss, use RAG → Cache response
No duplicate logic - each service handles its own responsibilities
"""
import time
from typing import Dict, Any, Optional, AsyncGenerator
from .unified_rag_service import UnifiedRAGService, rag_service
from .custom_cache_service import CustomCacheService
from ..core.logger import chat_logger
from ..config.settings import settings

class SimpleChatService:
    """
    Clean chat service that orchestrates cache and RAG without duplicate logic
    Responsibilities:
    - Check cache first
    - If cache miss: use RAG
    - Cache the RAG response
    """

    def __init__(self):
        """Initialize chat service with RAG and Cache"""
        self.rag_service = UnifiedRAGService()
        self.new_rag_service = rag_service  # New RAG service with database integration
        self.cache_service = CustomCacheService()
        chat_logger.info("SimpleChatService initialized - clean orchestration")

    async def chat(self, query: str, user_id: Optional[str] = None, course_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main chat method - clean orchestration without duplicate logic

        Args:
            query: User query
            user_id: Optional user ID
            course_id: Optional course ID for context filtering

        Returns:
            Complete response with metadata from either cache or RAG
        """
        try:
            # Step 1: Query cache service (following reference notebook pattern)
            # cache_service.query() handles both hit/miss and personalization internally
            cached_result = await self.cache_service.query(query, user_id, course_id)

            if cached_result is not None:
                # Cache hit (could be raw or personalized)
                chat_logger.info(f"Cache hit for user={user_id}, course={course_id}")
                return {
                    "response": cached_result,
                    "source": "cache_raw",  # Default to raw since no personalization
                    "cached": True,
                    "sources": [],  # Cache responses don't have RAG sources by design
                    "user_id": user_id,
                    "course_id": course_id,
                    "model": "cached",
                    "personalized": False
                }

            # Step 2: Cache miss - use RAG (RAG service handles document retrieval)
            rag_response = await self.rag_service.query(query, course_id)

            # Step 3: Store the RAG response in cache for future use
            if rag_response.get("answer"):
                # Generate embedding and store response using reference pattern
                embedding = await self.cache_service.generate_embedding(query)
                await self.cache_service.store_response(
                    prompt=query,
                    response=rag_response["answer"],
                    embedding=embedding,
                    user_id=user_id or "anonymous",
                    model=settings.openai_model_comprehensive,
                    course_id=course_id
                )

            chat_logger.info(f"RAG response for user={user_id}, course={course_id}")
            return {
                "response": rag_response["answer"],
                "source": "rag",
                "cached": False,
                "sources": rag_response.get("sources", []),
                "user_id": user_id,
                "course_id": course_id,
                "model": rag_response.get("model_used", settings.openai_model_comprehensive),
                "personalized": rag_response.get("personalized", False)
            }

        except Exception as e:
            chat_logger.error(f"Chat request failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request.",
                "source": "error",
                "cached": False,
                "sources": [],
                "user_id": user_id,
                "course_id": course_id,
                "error": str(e)
            }


# =================================================================
# LCEL PATTERN IMPLEMENTATION
# =================================================================
# LangChain Expression Language pattern - separate orchestration layer
# =================================================================

    # async def chat_lcel(self, query: str, user_id: Optional[str] = None,
    #                    course_id: Optional[str] = None, rag_threshold: Optional[float] = None) -> Dict[str, Any]:
    #     """
    #     LCEL chat method - separate orchestrator using LangChain patterns.
    #     Same cache logic, different RAG implementation.

    #     Args:
    #         query: User query
    #         user_id: Optional user ID
    #         course_id: Optional course ID for context filtering
    #         rag_threshold: Custom RAG threshold (overrides settings.rag_distance_threshold)

    #     Returns:
    #         Complete response with metadata from LCEL processing
    #     """
    #     try:
    #         # Step 1: Cache check (using same cache service!)
    #         cached_result = await self.cache_service.query(query, user_id, course_id)

    #         if cached_result is not None:
    #             # Cache hit (same logic as original chat)
    #             chat_logger.info(f"LCEL Cache hit for user={user_id}, course={course_id}")
    #             return {
    #                 "response": cached_result,
    #                 "source": "cache",
    #                 "cached": True,
    #                 "sources": [],
    #                 "user_id": user_id,
    #                 "course_id": course_id,
    #                 "method": "lcel_cached"
    #             }

    #         # Step 2: Cache miss - Use LCEL RAG service (different RAG implementation)
    #         rag_response = await lcel_rag_service.query(
    #             question=query,
    #             course_id=course_id,
    #             rag_threshold=rag_threshold or settings.rag_distance_threshold,
    #             top_k=settings.rag_top_k
    #         )

    #         # Step 3: Cache the LCEL response (same caching logic!)
    #         if rag_response.get("answer"):
    #             embedding = await self.cache_service.generate_embedding(query)
    #             await self.cache_service.store_response(
    #                 prompt=query,
    #                 response=rag_response["answer"],
    #                 embedding=embedding,
    #                 user_id=user_id or "anonymous",
    #                 model=settings.openai_model_comprehensive,
    #                 course_id=course_id
    #             )

    #         chat_logger.info(f"LCEL RAG response for user={user_id}, course={course_id}")
    #         return {
    #             "response": rag_response["answer"],
    #             "source": "rag",
    #             "cached": False,
    #             "sources": rag_response.get("sources", []),
    #             "user_id": user_id,
    #             "course_id": course_id,
    #             "context_quality": rag_response.get("context_quality", "unknown"),
    #             "rag_threshold": rag_response.get("rag_threshold", settings.rag_distance_threshold),
    #             "response_time_ms": rag_response.get("response_time_ms", 0),
    #             "method": "lcel_rag"
    #         }

    #     except Exception as e:
    #         chat_logger.error(f"LCEL Chat request failed: {e}")
    #         return {
    #             "response": "I apologize, but I encountered an error while processing your request with LCEL.",
    #             "source": "error",
    #             "cached": False,
    #             "sources": [],
    #             "user_id": user_id,
    #             "course_id": course_id,
    #             "error": str(e),
    #             "method": "lcel_error"
    #         }

    # async def chat_stream_lcel(self, query: str, user_id: Optional[str] = None,
    #                           course_id: Optional[str] = None, rag_threshold: Optional[float] = None):
    #     """
    #     LCEL streaming chat - orchestrator for streaming responses.
    #     Cache check first, then LCEL streaming.

    #     Yields:
    #         Response chunks as they're generated
    #     """
    #     try:
    #         # Cache check first (same logic!)
    #         cached_result = await self.cache_service.query(query, user_id, course_id)

    #         if cached_result is not None:
    #             for char in cached_result:
                    # yield char
    #             chat_logger.info(f"LCEL Streaming cache hit for user={user_id}")
    #             return

    #         # Cache miss: Use LCEL RAG streaming
    #         async for chunk in lcel_rag_service.stream(
    #             question=query,
    #             course_id=course_id,
    #             rag_threshold=rag_threshold or settings.rag_distance_threshold,
    #             top_k=settings.rag_top_k
    #         ):
    #             if chunk:
    #                 yield chunk

    #         chat_logger.info(f"LCEL Streaming completed for user={user_id}, course={course_id}")

    #     except Exception as e:
    #         chat_logger.error(f"LCEL Streaming chat failed: {e}")
    #         yield f"Error: {str(e)}"

    # =====================================
    # NEW IMPLEMENTATION WITH PROPER FLOW LOGIC
    # =====================================

    async def _track_chat_interaction(
        self,
        query: str,
        response_data: Dict[str, Any],
        user_id: Optional[str],
        course_id: Optional[str],
        chatroom_id: Optional[str]
    ):
        """Track chat interaction using existing db_models methods - NO REDUNDANCY"""
        try:
            from ..schemas.db_models import RequestTracking, Message, Response
            from ..core.database import async_db

            async with async_db.get_session() as db:
                # Step 1: Track request using async method
                request_tracking = await RequestTracking.acreate_request(
                    db=db,
                    request_type="chat",
                    service="simple_chat_service",
                    endpoint="chat_with_database",
                    user_id=user_id,
                    course_id=course_id,
                    parameters={
                        "query": query,
                        "chatroom_id": chatroom_id,
                        "personalized": response_data.get("personalized", False),
                        "cached": response_data.get("cached", False)
                    }
                )

                # Step 2: Create message using async method
                message = await Message.acreate_message(
                    db=db,
                    chatroom_id=chatroom_id or "default",
                    user_id=user_id or "anonymous",
                    message_text=query
                )

                # Step 3: Create response using async method
                await Response.acreate_response(
                    db=db,
                    message_id=message.message_id,
                    chatroom_id=chatroom_id or "default",
                    user_id=user_id or "anonymous",
                    response_text=response_data["response"],
                    model_used=response_data.get("model_used", settings.openai_model_comprehensive),
                    response_type=response_data.get("response_type", "rag_response"),
                    source_type=response_data.get("source_type", "knowledge_base"),
                    response_time_ms=response_data.get("response_time_ms", 0.0),
                    input_tokens=response_data.get("input_tokens", 0),
                    output_tokens=response_data.get("output_tokens", 0),
                    cost_usd=response_data.get("cost_usd", 0.0),
                    cache_hit=response_data.get("cached", False),
                    cache_similarity_score=response_data.get("cache_similarity_score"),
                    personalized=response_data.get("personalized", False)
                )

                chat_logger.info(f"Tracked chat interaction for user={user_id}, chatroom={chatroom_id}")

        except Exception as e:
            chat_logger.error(f"Failed to track chat interaction: {e}")
            # Don't fail the main response if tracking fails

    async def chat_with_database_stream(
        self,
        query: str,
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        chatroom_id: Optional[str] = None,
        use_personalization: bool = False
    ):
        """
        Streaming version of chat_with_database() with consistent parameters.

        Clean orchestration:
        1. Cache check (non-streaming via cache_service.query())
        2. Cache hit: Raw response or Personalized streaming response
        3. Cache miss: RAG streaming response (includes personalization if requested)
        4. Store complete response in cache (via cache_service.store_response())
        """
        try:
            # Step 1: Check cache using custom_cache_service (non-streaming) - measure time
            cache_query_start_time = time.time()
            cached_result = await self.cache_service.query(query, user_id, course_id)
            cache_query_time_ms = (time.time() - cache_query_start_time) * 1000

            if cached_result is not None:
                # Cache hit scenario
                chat_logger.info(f"Cache HIT for user={user_id}, course={course_id}")

                if use_personalization:
                    # Cache hit + Personalized streaming response - PANGGIL RAGService!
                    user_context_text, history_text = await self._get_context_components(
                        chatroom_id, user_id, course_id
                    )

                    if user_context_text or history_text:
                        # Panggil streaming personalization method dari RAGService + collect for tracking
                        personalization_start_time = time.time()
                        full_response = ""

                        async for chunk in self.new_rag_service._personalize_response_stream(
                            cached_result, user_context_text, history_text, query
                        ):
                            if chunk:
                                full_response += chunk
                                yield chunk

                        personalization_time_ms = (time.time() - personalization_start_time) * 1000

                        # Track personalized cache hit interaction
                        result_data = {
                            "response": full_response,
                            "source": "cache_personalized",
                            "cached": True,
                            "personalized": True,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "model_used": settings.openai_model_personalized,  # Personalization model (gpt-4.1-nano)
                            "response_type": "cache_hit_personalized",
                            "source_type": "redis_cache",
                            "response_time_ms": personalization_time_ms,
                            "cache_similarity_score": None
                        }

                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)
                    else:
                        # No context, return raw cached response + track
                        result_data = {
                            "response": cached_result,
                            "source": "cache_raw",
                            "cached": True,
                            "personalized": False,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "model_used": "cached",
                            "response_type": "cache_hit_raw",
                            "source_type": "redis_cache",
                            "response_time_ms": cache_query_time_ms
                        }

                        # Track cache hit interaction
                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

                        for char in cached_result:
                            yield char
                else:
                    # Cache hit + Raw response (single chunk) + track
                    result_data = {
                        "response": cached_result,
                        "source": "cache_raw",
                        "cached": True,
                        "personalized": False,
                        "user_id": user_id,
                        "course_id": course_id,
                        "chatroom_id": chatroom_id,
                        "model_used": "cached",
                        "response_type": "cache_hit_raw",
                        "source_type": "redis_cache",
                        "response_time_ms": cache_query_time_ms
                    }

                    # Track cache hit interaction
                    await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

                    # Stream the cached result
                    for char in cached_result:
                        yield char

                return

            # Cache miss scenario - use RAG streaming
            chat_logger.info(f"Cache MISS for user={user_id}, course={course_id} - Using RAG streaming")

            # Stream response from RAG service (includes personalization if requested) + timing
            rag_start_time = time.time()
            full_response = ""
            async for chunk in self.new_rag_service.stream(
                question=query,
                course_id=course_id,
                chatroom_id=chatroom_id,
                user_id=user_id,
                use_personalization=use_personalization
            ):
                if chunk:
                    full_response += chunk
                    yield chunk

            rag_time_ms = (time.time() - rag_start_time) * 1000

            # Track RAG streaming interaction
            result_data = {
                "response": full_response,
                "source": "rag",
                "cached": False,
                "personalized": use_personalization,
                "user_id": user_id,
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "model_used": settings.openai_model_personalized if use_personalization else settings.openai_model_comprehensive,
                "response_type": "rag_response",
                "source_type": "knowledge_base",
                "response_time_ms": rag_time_ms
            }

            await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

            # Step 2: Store complete response in cache (non-streaming)
            if full_response.strip():
                try:
                    embedding = await self.cache_service.generate_embedding(query)
                    await self.cache_service.store_response(
                        prompt=query,
                        response=full_response,
                        embedding=embedding,
                        user_id=user_id or "anonymous",
                        model=settings.openai_model_comprehensive,
                        course_id=course_id
                    )
                    chat_logger.info(f"Response cached for user={user_id}, course={course_id}")
                except Exception as cache_error:
                    chat_logger.warning(f"Failed to cache response: {cache_error}")

        except Exception as e:
            chat_logger.error(f"Chat with database stream failed: {e}")
            yield f"Maaf, saya mengalami kesalahan dalam memproses permintaan Anda: {str(e)}"

    async def _get_context_components(
        self,
        chatroom_id: Optional[str],
        user_id: Optional[str],
        course_id: Optional[str]
    ) -> tuple[str, str]:
        """
        Get user context and conversation history from database.
        Helper method for consistency with RAGService pattern.
        """
        if not user_id or not course_id:
            return "", ""

        try:
            from ..schemas.db_models import UserContext, Chatroom
            from ..core.database import async_db

            async with async_db.get_session() as db:
                # Get user context
                user_context = await UserContext.aget_or_create(db, user_id, course_id)
                user_context_text = user_context.get_context()

                # Get conversation history
                history_text = ""
                if chatroom_id:
                    chatroom = await db.get(Chatroom, chatroom_id)
                    if chatroom:
                        history_text = await chatroom.get_conversation_history(db, limit=5)

                return user_context_text, history_text

        except Exception as e:
            chat_logger.error(f"Failed to get context components: {e}")
            return "", ""

    async def chat_with_database(
        self,
        query: str,
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        chatroom_id: Optional[str] = None,
        use_personalization: bool = False
    ) -> Dict[str, Any]:
        """
        Chat method dengan flow logic yang benar:
        1. Redis cache check (custom_cache_service.py)
        2. Cache miss → RAG + GPT-4o-mini → Store cache (general response)
        3. Cache hit → Raw response or Personalized response (GPT-4o-nano)
        """

        try:
            # Step 1: Check cache using custom_cache_service - measure actual time
            cache_query_start_time = time.time()
            cached_result = await self.cache_service.query(query, user_id, course_id)
            cache_query_time_ms = (time.time() - cache_query_start_time) * 1000

            if cached_result is not None:
                # Cache hit scenario
                chat_logger.info(f"Cache HIT for user={user_id}, course={course_id}")

                if use_personalization:
                    # Cache hit + Personalized response - PANGGIL RAGService!
                    user_context_text, history_text = await self._get_context_components(
                        chatroom_id, user_id, course_id
                    )

                    if user_context_text or history_text:
                        personalization_start_time = time.time()
                        personalized_result = await self.new_rag_service._personalize_response(
                            cached_result, user_context_text, history_text, query
                        )
                        personalization_time_ms = (time.time() - personalization_start_time) * 1000

                        result_data = {
                            "response": personalized_result["response"],
                            "source": "cache_personalized",
                            "cached": True,
                            "personalized": True,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "model_used": personalized_result.get("model_used"),
                            "response_type": "cache_hit_personalized",
                            "source_type": "redis_cache",
                            "response_time_ms": personalization_time_ms,
                            "cache_similarity_score": None  # Cache hit tidak ada similarity score
                        }

                        # Track cache hit + personalization interaction
                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

                        return result_data
                    else:
                        # No context available
                        return {
                            "response": cached_result,
                            "source": "cache_raw",
                            "cached": True,
                            "personalized": False,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id
                        }
                else:
                    # Cache hit + Raw response
                    # Calculate actual cache response time from service
                    cache_start_time = time.time()
                    # (query already executed above, so measure time since then)
                    actual_cache_time_ms = (time.time() - cache_start_time) * 1000

                    result_data = {
                        "response": cached_result,
                        "source": "cache_raw",
                        "cached": True,
                        "personalized": False,
                        "user_id": user_id,
                        "course_id": course_id,
                        "chatroom_id": chatroom_id,
                        "model_used": "cached",
                        "response_type": "cache_hit_raw",
                        "source_type": "redis_cache",
                        "response_time_ms": cache_query_time_ms
                    }

                    # Track cache hit interaction
                    await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

                return result_data

            # Cache miss scenario - use RAG
            chat_logger.info(f"Cache MISS for user={user_id}, course={course_id} - Using RAG")
            return await self._handle_cache_miss(query, user_id, course_id, chatroom_id, use_personalization)

        except Exception as e:
            chat_logger.error(f"Chat with database failed: {e}")
            return {
                "response": "Maaf, saya mengalami kesalahan dalam memproses permintaan Anda.",
                "source": "error",
                "cached": False,
                "personalized": False,
                "user_id": user_id,
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "error": str(e)
            }

    async def _handle_cache_miss(
        self,
        query: str,
        user_id: Optional[str],
        course_id: Optional[str],
        chatroom_id: Optional[str],
        use_personalization: bool
    ) -> Dict[str, Any]:
        """
        Handle cache miss scenario:
        - Generate response using RAG + GPT-4o-mini (general response)
        - Store general response to cache (handled by custom_cache_service)
        - Return personalized response if requested
        """

        try:
            # Step 1: Generate response using RAG service
            rag_response = await self.new_rag_service.generate_response(
                question=query,
                course_id=course_id,
                chatroom_id=chatroom_id,
                user_id=user_id,
                use_personalization=False  # Generate general response first
            )

            general_response = rag_response["response"]
            model_used = rag_response["model_used"]

            # Step 2: Store general response in cache for future use
            if general_response.strip():
                try:
                    embedding = await self.cache_service.generate_embedding(query)
                    await self.cache_service.store_response(
                        prompt=query,
                        response=general_response,
                        embedding=embedding,
                        user_id=user_id or "anonymous",
                        model=model_used,
                        course_id=course_id
                    )
                    chat_logger.info(f"General response cached for user={user_id}, course={course_id}")
                except Exception as cache_error:
                    chat_logger.warning(f"Failed to cache general response: {cache_error}")

            # Step 3: Handle personalization if requested
            if use_personalization:
                # Get user context for personalization
                user_context_text, history_text = await self._get_context_components(
                    chatroom_id, user_id, course_id
                )

                if user_context_text or history_text:
                    personalized_result = await self.new_rag_service._personalize_response(
                        general_response, user_context_text, history_text, query
                    )
                    final_response = personalized_result["response"]
                    final_model = personalized_result["model_used"]
                else:
                    final_response = general_response
                    final_model = model_used
            else:
                final_response = general_response
                final_model = model_used

            result_data = {
                "response": final_response,
                "source": "rag",
                "cached": False,
                "personalized": use_personalization,
                "user_id": user_id,
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "model_used": final_model,
                "response_type": "rag_response",
                "source_type": "knowledge_base",
                "source_documents": rag_response.get("source_documents", []),
                "response_time_ms": rag_response.get("response_time_ms", 0)
            }

            # Track interaction in database
            await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id)

            chat_logger.info(f"RAG response generated for user={user_id}, course={course_id}")
            return result_data

        except Exception as e:
            chat_logger.error(f"Cache miss handling failed: {e}")
            return {
                "response": "Maaf, saya tidak dapat memproses permintaan Anda saat ini.",
                "source": "error",
                "cached": False,
                "personalized": False,
                "user_id": user_id,
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "error": str(e)
            }

    
    