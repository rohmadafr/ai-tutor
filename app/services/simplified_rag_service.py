"""
Simplified RAG Service

This replaces the complex RAGService with a cleaner implementation
that uses the UnifiedVectorStore for document storage and retrieval.
"""
from typing import List, Dict, Any, Optional
import time
import uuid
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config.settings import settings
from ..core.embeddings import EmbeddingService
from ..core.llm_client import LLMClient
from ..core.telemetry import TelemetryLogger, TokenCounter
from ..core.exceptions import RAGException
from ..repositories.unified_vector_store import unified_vector_store
from ..models.document import DocumentChunk
from ..core.logger import rag_logger


class SimplifiedRAGService:
    """
    Simplified RAG service that focuses on business logic only.
    Storage operations are delegated to UnifiedVectorStore.

    Replaces: RAGService (complex implementation)
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        llm_client: Optional[LLMClient] = None,
        telemetry: Optional[TelemetryLogger] = None
    ):
        """Initialize simplified RAG service with dependency injection."""
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_client = llm_client or LLMClient()
        self.telemetry = telemetry or TelemetryLogger()
        self.token_counter = TokenCounter()
        self.vector_store = unified_vector_store

        # Configuration - use distance threshold like semantic cache
        self.chunk_size = settings.rag_chunk_size or 1000
        self.chunk_overlap = settings.rag_chunk_overlap or 200
        self.top_k = settings.rag_top_k or 5
        # Use same threshold as semantic cache for consistency
        self.distance_threshold = settings.cache_threshold or 0.2

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        rag_logger.info("SimplifiedRAGService initialized - replacing complex RAGService")

    async def ingest_document(
        self,
        text: str,
        material_id: Optional[str] = None,
        course_id: str = "default",
        source_file: str = "unknown"
    ) -> str:
        """
        Ingest document into RAG knowledge base.
        Uses UnifiedVectorStore for storage.
        """
        start_time = time.time()

        try:
            # Generate material_id if not provided
            if not material_id:
                import hashlib
                material_id = hashlib.md5(text.encode()).hexdigest()

            # Check if document already exists
            if await self.vector_store.check_document_exists(material_id):
                rag_logger.info("Document already exists: material_id=%s", material_id)
                return material_id

            # Split document into chunks
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                raise RAGException("No chunks generated from document")

            # Generate embeddings for chunks
            chunk_vectors = []
            for chunk in chunks:
                vector = await self.embedding_service.generate_embedding(chunk)
                chunk_vectors.append(vector)

            # Store in unified vector store (knowledge base port 6379)
            await self.vector_store.store_documents(
                texts=chunks,
                vectors=chunk_vectors,
                material_id=material_id,
                course_id=course_id,
                source_file=source_file
            )

            ingestion_time = (time.time() - start_time) * 1000

            # Count actual tokens using TokenCounter
            text_tokens = self.token_counter.count_tokens(text)

            # Log telemetry
            self.telemetry.log(
                user_id="system",
                method="rag_ingestion",
                latency_ms=ingestion_time,
                input_tokens=text_tokens,
                output_tokens=0,
                cache_status="na",
                response_source="rag_ingestion",
                metadata={
                    "material_id": material_id,
                    "chunk_count": len(chunks),
                    "text_length": len(text),
                    "text_tokens": text_tokens,
                    "course_id": course_id,
                    "source_file": source_file
                }
            )

            rag_logger.info(
                "Document ingested: material_id=%s, chunk_count=%d, ingestion_time_ms=%.2f",
                material_id,
                len(chunks),
                ingestion_time
            )

            return material_id

        except Exception as e:
            rag_logger.error("Document ingestion failed: %s", str(e))
            raise RAGException(f"Ingestion failed: {str(e)}")

    async def retrieve_documents(
        self,
        query: str,
        course_id: Optional[str] = None,
        material_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant documents for query.
        Uses UnifiedVectorStore for retrieval with course-based filtering.
        """
        start_time = time.time()

        try:
            # Generate query embedding
            query_vector = await self.embedding_service.generate_embedding(query)

            # Search in knowledge base via unified store with course filtering
            search_results = await self.vector_store.search_knowledge_base(
                query_vector=query_vector,
                course_id=course_id,
                material_ids=material_ids,
                top_k=top_k or self.top_k,
                threshold=self.distance_threshold
            )

            # Convert to DocumentChunk objects
            chunks = []
            for result in search_results:
                chunk = DocumentChunk(
                    id=result.get("chunk_id", f"{result.get('material_id', '')}_{len(chunks)}"),
                    document_id=result.get("material_id", ""),
                    content=result.get("text", ""),
                    chunk_index=len(chunks),
                    start_pos=0,
                    end_pos=len(result.get("text", "")),
                    metadata={
                        "course_id": result.get("course_id", ""),
                        "source_file": result.get("source_file", ""),
                        "material_id": result.get("material_id", ""),
                        "chunk_id": result.get("chunk_id", "")
                    },
                    created_at=time.time(),
                    similarity_score=result.get("vector_distance", 1.0)
                )
                chunks.append(chunk)

            search_time = (time.time() - start_time) * 1000

            rag_logger.info(
                "Document retrieval: query_length=%d, results_count=%d, search_time_ms=%.2f",
                len(query),
                len(chunks),
                search_time
            )

            return chunks

        except Exception as e:
            rag_logger.error("Document retrieval failed: %s", str(e))
            raise RAGException(f"Retrieval failed: {str(e)}")

    async def generate_answer(
        self,
        question: str,
        context_chunks: List[DocumentChunk],
        user_memory: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context.
        Business logic only - no storage operations here.
        Returns dictionary with response and metadata.
        """
        if not context_chunks:
            return {
                "response": "I don't have enough information to answer your question.",
                "model": "fallback",
                "latency_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0,
                "context_chunks": 0
            }

        start_time = time.time()

        try:
            # Build context from chunks
            context = self._build_context(context_chunks)

            # Build messages for LLM
            messages = self._build_rag_messages(question, context, user_memory)

            # Generate answer using LLM client (use async versions)
            if hasattr(self.llm_client, 'acall_comprehensive'):
                # Use the async comprehensive model for RAG answers
                result = await self.llm_client.acall_comprehensive(
                    prompt=self._messages_to_prompt(messages),
                    max_tokens=500
                )
                answer = result["response"]
                model_used = result.get("model", "comprehensive")
                input_tokens = result.get("input_tokens", 0)
                output_tokens = result.get("output_tokens", 0)
            else:
                # Fallback to chat completion
                from ..models.chat import ChatRequest
                request = ChatRequest(
                    query=self._messages_to_prompt(messages),
                    max_tokens=500,
                    temperature=0.1
                )
                response = await self.llm_client.chat_completion(request)
                answer = response.response
                model_used = "chat_completion"
                input_tokens = response.token_usage.get("prompt_tokens", 0)
                output_tokens = response.token_usage.get("completion_tokens", 0)

            generation_time = (time.time() - start_time) * 1000
            total_tokens = input_tokens + output_tokens

            # Calculate cost using token counter
            cost_usd = self.llm_client.token_counter.calculate_cost(
                input_tokens, output_tokens, model_used
            )

            # Log telemetry
            self.telemetry.log(
                user_id="system",
                method="rag_answer_generation",
                latency_ms=generation_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_status="na",
                response_source=model_used,
                metadata={
                    "question_length": len(question),
                    "context_chunks": len(context_chunks),
                    "has_user_memory": user_memory is not None,
                    "model_used": model_used,
                    "cost_usd": cost_usd
                }
            )

            rag_logger.info(
                "Answer generated: question_length=%d, context_chunks=%d, generation_time_ms=%.2f",
                len(question),
                len(context_chunks),
                generation_time
            )

            return {
                "response": answer,
                "model": model_used,
                "latency_ms": generation_time,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "context_chunks": len(context_chunks)
            }

        except Exception as e:
            rag_logger.error("Answer generation failed: %s", str(e))
            raise RAGException(f"Generation failed: {str(e)}")

    def _build_context(self, chunks: List[DocumentChunk]) -> str:
        """Build context string from document chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Document {i+1} - Distance: {chunk.similarity_score:.3f}]\n{chunk.content}"
            )
        return "\n\n".join(context_parts)

    def _build_rag_messages(
        self,
        question: str,
        context: str,
        user_memory: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, str]]:
        """Build RAG messages for LLM."""
        system_prompt = """You are a helpful AI assistant that answers questions based on provided context.

Instructions:
1. Use only the information from the provided context to answer the question
2. If the context doesn't contain enough information, say so clearly
3. Cite the relevant document sources when possible
4. Provide accurate and comprehensive answers
5. Keep your answers focused and relevant"""

        if user_memory:
            personalization = self._build_personalization_context(user_memory)
            system_prompt += f"\n\nPersonalization Context:\n{personalization}"

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
            }
        ]

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)

    def _build_personalization_context(self, user_memory: Dict[str, List[str]]) -> str:
        """Build personalization context from user memory."""
        parts = []
        if user_memory.get("preferences"):
            parts.append(f"User Preferences: {', '.join(user_memory['preferences'])}")
        if user_memory.get("history"):
            parts.append(f"User History: {', '.join(user_memory['history'][-3:])}")
        if user_memory.get("goals"):
            parts.append(f"User Goals: {', '.join(user_memory['goals'])}")
        return "\n".join(parts) if parts else "No specific user context available."

    async def get_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics."""
        try:
            # Get overall stats from unified store
            store_stats = await self.vector_store.get_stats()

            return {
                "knowledge_base": store_stats.get("knowledge_base", {}),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "distance_threshold": self.distance_threshold,
                "embedding_service_initialized": bool(self.embedding_service),
                "llm_client_initialized": bool(self.llm_client),
                "text_splitter_initialized": bool(self.text_splitter),
                "simplified_implementation": True
            }

        except Exception as e:
            rag_logger.error("Failed to get RAG stats: %s", str(e))
            return {"error": str(e)}