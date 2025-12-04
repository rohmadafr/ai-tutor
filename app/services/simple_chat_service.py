"""
Simple Chat Service - Clean Orchestration Layer
Single responsibility: Check cache → If miss, use RAG → Cache response
No duplicate logic - each service handles its own responsibilities
"""
import time
import datetime
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


# =====================================
# NEW IMPLEMENTATION WITH PROPER FLOW LOGIC
# =====================================

    async def _track_chat_interaction(
        self,
        query: str,
        response_data: Dict[str, Any],
        user_id: Optional[str],
        course_id: Optional[str],
        chatroom_id: Optional[str],
        is_streaming: bool = False
    ):
        """Track chat interaction using existing db_models methods - NO REDUNDANCY"""
        try:
            # Filter out out-of-context responses from database tracking
            source_type = response_data.get("source_type", "")

            # Debug log untuk cek source_type
        
            # Skip tracking for OOT responses to avoid polluting database
            if source_type == "out_of_context":
                chat_logger.info(f"Skipping database tracking for out-of-context response")
                return
            from ..schemas.db_models import RequestTracking, Message, Response
            from ..core.database import async_db

            async with async_db.get_session() as db:
                # Step 1: Track request using async method
                endpoint = "chat_with_database_streaming" if is_streaming else "chat_with_database"
                request_tracking = await RequestTracking.acreate_request(
                    db=db,
                    request_type="chat",
                    service="simple_chat_service",
                    endpoint=endpoint,
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
                    chatroom_id=chatroom_id if chatroom_id else None,
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
                    latency_ms=response_data.get("latency_ms", 0.0),
                    input_tokens=response_data.get("input_tokens", 0),
                    output_tokens=response_data.get("output_tokens", 0),
                    cost_usd=response_data.get("cost_usd", 0.0),
                    cache_hit=response_data.get("cached", False),
                    cache_similarity_score=response_data.get("cache_similarity_score"),
                    personalized=response_data.get("personalized", False)
                )

                # Step 4: Update tracking status and add token usage
                input_tokens = response_data.get("input_tokens", 0)
                output_tokens = response_data.get("output_tokens", 0)
                total_tokens = response_data.get("total_tokens", input_tokens + output_tokens)  # Use total_tokens if available

                # Update the request tracking with completion data
                await self._update_tracking_completion(
                    db, request_tracking.tracking_id, total_tokens,
                    len(response_data.get("response", "")), response_data
                )

                chat_logger.info(f"Tracked chat interaction for user={user_id}, chatroom={chatroom_id}, tokens={total_tokens}")

        except Exception as e:
            chat_logger.error(f"Failed to track chat interaction: {e}")
            # Don't fail the main response if tracking fails

    async def _update_tracking_completion(
        self,
        db,
        tracking_id: str,
        tokens_used: int,
        result_count: int,
        response_data: Dict[str, Any]
    ):
        """Update request tracking with completion data."""
        try:
            from ..schemas.db_models import RequestTracking

            # Get the tracking record
            tracking = await db.get(RequestTracking, tracking_id)
            if tracking:
                tracking.status = "completed"
                tracking.tokens_used = tokens_used
                tracking.result_count = result_count
                tracking.completed_at = datetime.datetime.utcnow()
                tracking.request_metadata = {
                    "latency_ms": response_data.get("latency_ms", 0.0),
                    "cached": response_data.get("cached", False),
                    "personalized": response_data.get("personalized", False),
                    "response_type": response_data.get("response_type", "rag_response"),
                    "source_type": response_data.get("source_type", "knowledge_base")
                }

                await db.commit()
                await db.refresh(tracking)

        except Exception as e:
            chat_logger.error(f"Failed to update tracking completion: {e}")

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
            chat_logger.info(f"🔍 [CHAT DEBUG] Starting query: '{query}' | User: {user_id} | Course: {course_id} | Personalization: {use_personalization}")
            cached_result = await self.cache_service.query(query, user_id, course_id)
            cache_query_time_ms = (time.time() - cache_query_start_time) * 1000

            if cached_result is not None and cached_result.get("cache_status") != "miss":
                chat_logger.info(f"🔍 [CHAT DEBUG] Cache HIT! Found {len(cached_result.get('sources', []))} cached sources")
                # Cache hit scenario
                chat_logger.info(f"Cache HIT for user={user_id}, course={course_id}")

                # Extract sources from cached result (returned by custom_cache_service.query)
                cached_sources = cached_result.get("sources", [])
                cached_response = cached_result.get("response", "")

                if use_personalization:
                    # Cache hit + Personalized streaming response - PANGGIL RAGService!
                    user_context_text, history_text = await self._get_context_components(
                        chatroom_id, user_id, course_id
                    )

                    if user_context_text or history_text:
                        # Panggil streaming personalization method dari RAGService + collect for tracking
                        personalization_start_time = time.time()
                        full_response = ""
                        personalization_token_data = {}

                        # Get personalization generator with new metadata format
                        # Pass only the response string, not the full dictionary
                        personalization_gen = self.new_rag_service._personalize_response_stream(
                            cached_response, user_context_text, history_text, query, cached_sources
                        )

                        # Process streaming items with metadata
                        personalization_token_data = {}

                        async for item in personalization_gen:
                            if item["type"] == "content":
                                # Yield content chunks for streaming
                                full_response += item["data"]
                                yield item["data"]
                            elif item["type"] == "metadata":
                                # Capture token data for tracking - ensure it's a dictionary
                                item_data = item["data"]
                                if isinstance(item_data, dict):
                                    personalization_token_data = item_data
                                    # Merge source documents from cache into personalization metadata
                                    if cached_sources and "source_documents" not in personalization_token_data:
                                        personalization_token_data["source_documents"] = cached_sources
                                else:
                                    chat_logger.warning(f"Expected dictionary for personalization metadata, got {type(item_data)}: {item_data}")
                                    personalization_token_data = {"source_documents": cached_sources}

                                # Forward personalization metadata to WebSocket
                                yield item
                            elif item["type"] == "error":
                                # Handle errors
                                yield item["data"]

                        personalization_time_ms = (time.time() - personalization_start_time) * 1000

                        # Ensure personalization_token_data is a dictionary before using .get()
                        if not isinstance(personalization_token_data, dict):
                            chat_logger.warning(f"personalization_token_data is not a dictionary: {type(personalization_token_data)}, resetting to empty dict")
                            personalization_token_data = {}

                        # Track personalized cache hit interaction
                        result_data = {
                            "response": full_response,
                            "source": personalization_token_data.get("source", "cache_personalized"),
                            "cached": True,
                            "personalized": True,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "model_used": personalization_token_data.get("model_used", settings.openai_model_personalized),
                            "response_type": personalization_token_data.get("response_type", "cache_hit_personalized"),
                            "source_type": personalization_token_data.get("source_type", "redis_cache"),
                            "latency_ms": personalization_time_ms,
                            "cache_similarity_score": None,
                            "input_tokens": personalization_token_data.get("input_tokens", 0),
                            "output_tokens": personalization_token_data.get("output_tokens", 0),
                            "total_tokens": personalization_token_data.get("total_tokens", 0),
                            "cost_usd": personalization_token_data.get("cost_usd", 0.0),
                            "source_documents": cached_sources  # ✅ Include sources from cache in final response
                        }

                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=True)
                    else:
                        # No context, return raw cached response + track
                        result_data = {
                            "response": cached_response,
                            "source": "cache_raw",
                            "cached": True,
                            "personalized": False,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "model_used": "cached",
                            "response_type": "cache_hit_raw",
                            "source_type": "redis_cache",
                            "latency_ms": cache_query_time_ms,
                            "input_tokens": 0,  # Cache hit = no LLM tokens
                            "output_tokens": 0,  # Cache hit = no LLM tokens
                            "cost_usd": 0.0,    # Cache hit = no cost
                            "source_documents": cached_sources  # ✅ Include sources from cache
                        }

                        # Track cache hit interaction
                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=False)

                        # Cache hit should NOT stream content, but send metadata for WebSocket
                        yield {
                            "type": "metadata",
                            "data": {
                                "source": result_data.get("source", "cache_raw"),
                                "source_type": result_data.get("source_type", "redis_cache"),
                                "response_type": result_data.get("response_type", "cache_hit_raw"),
                                "model_used": result_data.get("model_used", "cached"),
                                "input_tokens": result_data.get("input_tokens", 0),
                                "output_tokens": result_data.get("output_tokens", 0),
                                "total_tokens": result_data.get("total_tokens", 0),
                                "cost_usd": result_data.get("cost_usd", 0.0),
                                "source_documents": result_data.get("source_documents", []),
                                "personalized": result_data.get("personalized", False)
                            }
                        }
                        # Then yield the cached response
                        yield cached_response
                else:
                    # Cache hit + Raw response (single chunk) + track
                    result_data = {
                        "response": cached_response,  # Use extracted cached_response
                        "source": "cache_raw",
                        "cached": True,
                        "personalized": False,
                        "user_id": user_id,
                        "course_id": course_id,
                        "chatroom_id": chatroom_id,
                        "model_used": "cached",
                        "response_type": "cache_hit_raw",
                        "source_type": "redis_cache",
                        "latency_ms": cache_query_time_ms,
                        "source_documents": cached_sources  # ✅ Include sources from cache
                    }

                    # Track cache hit interaction
                    await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=False)

                    # Cache hit should NOT stream content, but send metadata for WebSocket
                    yield {
                        "type": "metadata",
                        "data": {
                            "source": result_data.get("source", "cache_raw"),
                            "source_type": result_data.get("source_type", "redis_cache"),
                            "response_type": result_data.get("response_type", "cache_hit_raw"),
                            "model_used": result_data.get("model_used", "cached"),
                            "input_tokens": result_data.get("input_tokens", 0),
                            "output_tokens": result_data.get("output_tokens", 0),
                            "total_tokens": result_data.get("total_tokens", 0),
                            "cost_usd": result_data.get("cost_usd", 0.0),
                            "source_documents": result_data.get("source_documents", []),
                            "personalized": result_data.get("personalized", False)
                        }
                    }
                    # Then yield cached response
                    yield cached_response

                return

            # Cache miss scenario - use RAG streaming
            chat_logger.info(f"🔍 [CHAT DEBUG] Cache MISS! Proceeding to RAG for course={course_id}")

            # Extract embedding from cache miss result for reuse
            cached_embedding = cached_result.get("embedding") if cached_result else None
            if cached_embedding:
                chat_logger.info(f"🔄 [EMBEDDING REUSE] Using cached embedding to avoid duplicate generation")

            # Stream response from RAG service (includes personalization if requested) + timing
            rag_start_time = time.time()
            full_response = ""
            rag_token_data = {}
            general_response = ""  # Store the general response for caching (only when personalization is requested)

            async for item in self.new_rag_service.generate_response_stream(
                question=query,
                course_id=course_id,
                chatroom_id=chatroom_id,
                user_id=user_id,
                use_personalization=use_personalization,
                cached_embedding=cached_embedding  # 🔥 PASS CACHED EMBEDDING
            ):
                if item["type"] == "content":
                    # Yield content chunks for streaming
                    # These are now either general response (no personalization) or personalized response
                    full_response += item["data"]
                    yield item["data"]
                elif item["type"] == "metadata":
                    # Capture token data for tracking - ensure it's a dictionary
                    item_data = item["data"]
                    if isinstance(item_data, dict):
                        rag_token_data = item_data
                        # Check if this metadata includes the general_response (for personalization cases)
                        if "general_response" in item_data and item_data["general_response"]:
                            general_response = item_data["general_response"]
                    else:
                        chat_logger.warning(f"Expected dictionary for metadata, got {type(item_data)}: {item_data}")
                        rag_token_data = {}
                elif item["type"] == "error":
                    # Handle errors
                    yield item["data"]

            rag_time_ms = (time.time() - rag_start_time) * 1000

            # Track RAG streaming interaction with real token data
            # Ensure rag_token_data is a dictionary before using .get()
            if not isinstance(rag_token_data, dict):
                chat_logger.warning(f"rag_token_data is not a dictionary: {type(rag_token_data)}, resetting to empty dict")
                rag_token_data = {}

            result_data = {
                "response": full_response,
                "source": rag_token_data.get("source", "rag"),
                "cached": False,
                "personalized": use_personalization,
                "user_id": user_id,
                "course_id": course_id,
                "chatroom_id": chatroom_id,
                "model_used": rag_token_data.get("model_used", settings.openai_model_personalized if use_personalization else settings.openai_model_comprehensive),
                "response_type": rag_token_data.get("response_type", "rag_response"),
                "source_type": rag_token_data.get("source_type", "knowledge_base"),
                "source_documents": rag_token_data.get("source_documents", []),
                "latency_ms": rag_time_ms,
                "input_tokens": rag_token_data.get("input_tokens", 0),
                "output_tokens": rag_token_data.get("output_tokens", 0),
                "total_tokens": rag_token_data.get("total_tokens", 0),
                "cost_usd": rag_token_data.get("cost_usd", 0.0)
            }

            await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=True)

            # Store GENERAL response in cache for future use (not personalized response)
            response_to_cache = general_response if general_response else full_response

            if response_to_cache.strip():
                try:
                    # Reuse embedding from cache miss
                    embedding = cached_result.get("embedding") if cached_result else None
                    if embedding:
                        chat_logger.info(f"Reusing embedding from cache miss - storing {'general' if use_personalization and general_response else 'non-personalized'} response")

                        # Extract sources from RAG response for cache storage
                        source_documents = rag_token_data.get("source_documents", [])

                        await self.cache_service.store_response(
                            prompt=query,
                            response=response_to_cache,  # Store general response, not personalized
                            embedding=embedding,
                            user_id=user_id or "anonymous",
                            model=rag_token_data.get("model_used", settings.openai_model_comprehensive),
                            course_id=course_id,
                            sources=source_documents
                        )
                        chat_logger.info(f"RAG response cached for user={user_id}, course={course_id} - {'with personalization requested' if use_personalization else 'no personalization'}")
                    else:
                        chat_logger.warning("No embedding available for cache storage")
                except Exception as cache_error:
                    chat_logger.warning(f"Failed to cache RAG response: {cache_error}")

            # Yield final accumulated metadata for WebSocket clients
            # This ensures metadata is always sent, regardless of personalization
            yield {
                "type": "metadata",
                "data": {
                    "source": result_data.get("source", "rag"),
                    "source_type": result_data.get("source_type", "knowledge_base"),
                    "response_type": result_data.get("response_type", "rag_response"),
                    "model_used": result_data.get("model_used", settings.openai_model_comprehensive),
                    "input_tokens": result_data.get("input_tokens", 0),
                    "output_tokens": result_data.get("output_tokens", 0),
                    "total_tokens": result_data.get("total_tokens", 0),
                    "cost_usd": result_data.get("cost_usd", 0.0),
                    "source_documents": result_data.get("source_documents", []),
                    "personalized": result_data.get("personalized", False)
                }
            }

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

                # Extract sources from cached result (returned by custom_cache_service.query)
                cached_sources = cached_result.get("sources", [])
                cached_response = cached_result.get("response", "")

                
                if use_personalization:
                    # Cache hit + Personalized response - PANGGIL RAGService!
                    user_context_text, history_text = await self._get_context_components(
                        chatroom_id, user_id, course_id
                    )

                    if user_context_text or history_text:
                        personalization_start_time = time.time()
                        # Pass only the response string, not the full dictionary
                        personalized_result = await self.new_rag_service._personalize_response(
                            cached_response, user_context_text, history_text, query, cached_sources
                        )
                        personalization_time_ms = (time.time() - personalization_start_time) * 1000

                        # Extract token data from nested structure
                        token_data = personalized_result.get("tokens", {})
                        input_tokens = token_data.get("input", 0)
                        output_tokens = token_data.get("output", 0)
                        total_tokens = input_tokens + output_tokens
                        cost_usd = token_data.get("cost", 0.0)

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
                            "latency_ms": personalization_time_ms,
                            "cache_similarity_score": None,  # Cache hit tidak ada similarity score
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "cost_usd": cost_usd,
                            "source_documents": cached_sources  # ✅ Use sources from cache query result
                        }

                        # Track cache hit + personalization interaction
                        await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=False)

                        return result_data
                    else:
                        # No context available
                        return {
                            "response": cached_response,  # Use extracted cached_response
                            "source": "cache_raw",
                            "cached": True,
                            "personalized": False,
                            "user_id": user_id,
                            "course_id": course_id,
                            "chatroom_id": chatroom_id,
                            "source_documents": cached_sources  # ✅ Include sources from cache
                        }
                else:
                    # Cache hit + Raw response
                    # Calculate actual cache response time from service
                    cache_start_time = time.time()
                    # (query already executed above, so measure time since then)
                    actual_cache_time_ms = (time.time() - cache_start_time) * 1000

                    result_data = {
                        "response": cached_response,  # Use extracted cached_response
                        "source": "cache_raw",
                        "cached": True,
                        "personalized": False,
                        "user_id": user_id,
                        "course_id": course_id,
                        "chatroom_id": chatroom_id,
                        "model_used": "cached",
                        "response_type": "cache_hit_raw",
                        "source_type": "redis_cache",
                        "latency_ms": cache_query_time_ms,
                        "source_documents": cached_sources  # ✅ Include sources from cache
                    }

                    # Track cache hit interaction
                    await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=False)

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

            # Extract token data from RAG response
            rag_input_tokens = rag_response.get("input_tokens", 0)
            rag_output_tokens = rag_response.get("output_tokens", 0)
            rag_total_tokens = rag_response.get("total_tokens", 0)
            rag_cost_usd = rag_response.get("cost_usd", 0.0)

            # Step 2: Store general response in cache for future use
            if general_response.strip():
                try:
                    # Get embedding from cache miss result to avoid duplicate generation
                    embedding = cached_result.get("embedding") if cached_result else None
                    if not embedding:
                        # Fallback: generate embedding only if not available from cache miss
                        embedding = await self.cache_service.generate_embedding(query)
                    # Extract sources from RAG response for cache storage
                    source_documents = rag_response.get("source_documents", [])

                    await self.cache_service.store_response(
                        prompt=query,
                        response=general_response,
                        embedding=embedding,
                        user_id=user_id or "anonymous",
                        model=model_used,
                        course_id=course_id,
                        sources=source_documents  
                    )
                    chat_logger.info(f"General response cached for user={user_id}, course={course_id} with {len(source_documents)} sources")
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
                        general_response, user_context_text, history_text, query, source_documents
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
                "source_documents": personalized_result.get("source_documents", rag_response.get("source_documents", [])),
                "latency_ms": rag_response.get("latency_ms", 0),
                "input_tokens": rag_input_tokens,  # Only input tokens
                "output_tokens": rag_output_tokens, # Only output tokens
                "total_tokens": rag_total_tokens,    # Total tokens
                "cost_usd": rag_cost_usd
            }

            # Track interaction in database
            await self._track_chat_interaction(query, result_data, user_id, course_id, chatroom_id, is_streaming=False)

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

    
    