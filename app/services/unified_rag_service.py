"""
Unified RAG Service with integrated Vector Store
Single service for RAG + Vector Operations - no external dependencies
Based on efficient batch processing pattern from ai-services
"""
import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
import redis.asyncio as redis
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# RedisVL imports
try:
    from redisvl.index import AsyncSearchIndex
    from redisvl.query import VectorQuery, FilterQuery
    from redisvl.query.filter import Tag, FilterExpression
    from redisvl.schema import IndexSchema
    REDISVL_AVAILABLE = True
except ImportError:
    REDISVL_AVAILABLE = False

from ..config.settings import settings
from ..core.logger import rag_logger
from ..core.exceptions import RedisException
from ..core.telemetry import TokenCounter
from ..utils.batch_processor import batch_processor


class UnifiedRAGService:
    """
    Unified RAG Service with integrated Vector Store
    Single service handling: Embeddings, LLM, Redis operations, Document management
    """

    def __init__(self):
        """Initialize the RAG service with embedded vector store"""
        if not REDISVL_AVAILABLE:
            raise RedisException("RedisVL is required for RAG operations")

        # Core components
        self.embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key
        )

        self.llm = ChatOpenAI(
            model=settings.openai_model_comprehensive,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            openai_api_key=settings.openai_api_key
        )

        # Vector Store Configuration (embedded)
        self.knowledge_base_url = settings.redis_knowledge_url
        self.client: redis.Redis = None
        self.index: AsyncSearchIndex = None
        self.index_name = "lms_global_index"
        self.prefix = "chunk"
        self.vector_dimension = settings.vector_dimension or 1536
        self.connected = False

        # Batch processing components
        self.token_counter = TokenCounter()
        rag_logger.info("Batch processor initialized with token counter")

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        Anda adalah AI Tutor Assistant yang membantu menjawab pertanyaan user pada sebuah Learning Management System
        berdasarkan konteks atau knowledge base pada course yang diberikan.

        Konteks:
        {context}

        Pertanyaan: {question}

        Jawab pertanyaan berdasarkan konteks yang diberikan. Jika konteks tidak mengandung
        informasi yang cukup untuk menjawab pertanyaan, katakan "Saya tidak memiliki informasi
        yang cukup untuk menjawab pertanyaan ini."

        Jawab dalam bahasa yang sama dengan pertanyaan.
        """)

        rag_logger.info("UnifiedRAGService initialized with embedded Vector Store")

    def _extract_usage_metadata(self, llm_response) -> Dict[str, int]:
        """
        Extract token usage from LangChain AIMessage response.

        Supports both old and new LangChain versions:
        - New: response.usage_metadata
        - Old: response.response_metadata['token_usage']

        Returns:
            Dict with input_tokens, output_tokens, total_tokens
        """
        try:
            # Try new format first (LangChain >= 0.1.0)
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                usage = llm_response.usage_metadata
                if isinstance(usage, dict):
                    return {
                        "input_tokens": usage.get('input_tokens', 0),
                        "output_tokens": usage.get('output_tokens', 0),
                        "total_tokens": usage.get('total_tokens', 0)
                    }
                else:
                    rag_logger.warning(f"usage_metadata is not a dict: {type(usage)}, value: {usage}")
                    # Fall through to old format or estimation

            # Try old format (LangChain < 0.1.0)
            if hasattr(llm_response, 'response_metadata'):
                # Ensure response_metadata is a dictionary, not string
                response_metadata = llm_response.response_metadata
                if isinstance(response_metadata, dict):
                    token_usage = response_metadata.get('token_usage', {})
                    return {
                        "input_tokens": token_usage.get('prompt_tokens', 0),
                        "output_tokens": token_usage.get('completion_tokens', 0),
                        "total_tokens": token_usage.get('total_tokens', 0)
                    }
                else:
                    rag_logger.warning(f"response_metadata is not a dict: {type(response_metadata)}, value: {response_metadata}")
                    # Fall through to token estimation

            # Fallback: estimate tokens using TokenCounter
            rag_logger.warning("No usage metadata found in LLM response, estimating tokens")
            if hasattr(llm_response, 'content'):
                output_tokens = self.token_counter.count_tokens(llm_response.content)
                return {
                    "input_tokens": 0,  # Can't estimate without prompt
                    "output_tokens": output_tokens,
                    "total_tokens": output_tokens
                }

        except Exception as e:
            rag_logger.error(f"Failed to extract usage metadata: {e}")

        # Default fallback
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _define_index_schema(self) -> IndexSchema:
        """Define RedisVL index schema for knowledge base"""
        schema_definition = {
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
                "storage_type": "hash"
            },
            "fields": [
                # Filterable metadata (Tag)
                {"name": "material_id", "type": "tag"},
                {"name": "course_id", "type": "tag"},
                {"name": "page", "type": "tag"},

                # Content (Text)
                {"name": "text", "type": "text"},
                {"name": "filename", "type": "text"},

                # Vector (HNSW)
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": self.vector_dimension,
                        "distance_metric": "COSINE",
                        "datatype": "FLOAT32"
                    }
                }
            ]
        }

        return IndexSchema.from_dict(schema_definition)

    async def connect(self) -> None:
        """Connect to Redis and create index if needed"""
        if self.connected and self.index:
            try:
                await self.client.ping()
                return
            except Exception:
                rag_logger.warning("Redis connection lost, reconnecting...")
                self.connected = False
                self.index = None

        try:
            # Connect to Redis
            self.client = await redis.from_url(
                self.knowledge_base_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()

            # Initialize index using external YAML schema (RedisVL best practice)
            try:
                from redisvl.index import AsyncSearchIndex
                self.index = AsyncSearchIndex.from_yaml(
                    "app/schemas/knowledge_base_schema.yaml",
                    redis_client=self.client
                )
                await self.index.create(overwrite=False)
            except Exception as yaml_error:
                rag_logger.warning(f"Failed to load YAML schema: {yaml_error}")
                # Fallback to manual schema definition
                schema = self._define_index_schema()
                self.index = AsyncSearchIndex(schema, redis_client=self.client)
                await self.index.create(overwrite=False)

            self.connected = True
            rag_logger.info(f"Connected to Redis and initialized index '{self.index_name}'")

        except Exception as e:
            if "Index already exists" in str(e):
                rag_logger.info(f"Index '{self.index_name}' already exists")
                # Connect to existing index using YAML schema
                try:
                    self.index = AsyncSearchIndex.from_yaml(
                        "app/schemas/knowledge_base_schema.yaml",
                        redis_client=self.client
                    )
                except Exception as yaml_error:
                    rag_logger.warning(f"Failed to load YAML schema for existing index: {yaml_error}")
                    # Fallback to manual schema
                    schema = self._define_index_schema()
                    self.index = AsyncSearchIndex(schema, redis_client=self.client)
                self.connected = True
            else:
                rag_logger.error(f"Failed to connect to Redis: {e}")
                raise RedisException(f"Connection failed: {e}")

    async def _ensure_connection(self):
        """Ensure connection to Redis"""
        if not self.connected:
            await self.connect()

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to knowledge base with efficient batch processing"""
        try:
            await self._ensure_connection()

            # Convert to LangChain Document format
            langchain_docs = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # Preserve all metadata from the document, especially page numbers
                langchain_doc = Document(
                    page_content=content,
                    metadata={
                        "material_id": str(metadata.get("material_id", f"doc_{i}")),
                        "course_id": str(metadata.get("course_id", "default")),
                        "filename": str(metadata.get("filename", "unknown")),
                        "page": str(metadata.get("page", 0)),  # Preserve original page number
                        "source": metadata.get("source", ""),  # Preserve source if available
                        "preprocessed": metadata.get("preprocessed", False),
                        "original_length": metadata.get("original_length", 0),
                        "processed_length": metadata.get("processed_length", 0),
                        **{k: v for k, v in metadata.items() if k not in ["material_id", "course_id", "filename", "page", "source", "preprocessed", "original_length", "processed_length"]}
                    }
                )
                langchain_docs.append(langchain_doc)

            # Count total tokens to check if we need batching
            total_tokens = self.token_counter.count_documents_tokens(langchain_docs)

            rag_logger.info(f"Adding {len(langchain_docs)} documents with {total_tokens} tokens to vector store")

            if total_tokens <= 250000:
                # Single batch processing
                await self._process_documents_batch(langchain_docs)
            else:
                # Multi-batch processing
                rag_logger.info(f"Using batch processing for {total_tokens} tokens")
                batches = batch_processor.split_into_batches(langchain_docs)
                rag_logger.info(f"Split into {len(batches)} batches for embedding generation")

                total_processed = 0
                for i, batch in enumerate(batches, 1):
                    rag_logger.info(f"Processing batch {i}/{len(batches)} with {len(batch)} documents")
                    await self._process_documents_batch(batch)
                    total_processed += len(batch)

                rag_logger.info(f"✓ Berhasil memuat {total_processed} chunks ke Redis dalam {len(batches)} batch.")

            return []  # No document IDs needed for upload endpoint

        except Exception as e:
            rag_logger.error(f"Failed to add documents: {e}")
            raise RedisException(f"Document storage failed: {str(e)}")

    async def _process_documents_batch(self, documents: List[Document]):
        """Process a single batch of documents for embedding and storage."""
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)

        data_to_load = []
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            material_id = doc.metadata.get("material_id", "unknown")

            data = {
                "text": doc.page_content,
                "vector": np.array(vec, dtype=np.float32).tobytes(),  # Convert to bytes
                "material_id": material_id,
                "course_id": str(doc.metadata.get("course_id", "default")),
                "filename": str(doc.metadata.get("filename", "unknown")),
                "page": str(doc.metadata.get("page", 0))  # Use actual page number from metadata
            }

            data_to_load.append(data)

        await self.index.load(data_to_load)
        rag_logger.info(f"✓ Berhasil memuat {len(data_to_load)} chunks ke Redis.")

    async def query(self, question: str, course_id: Optional[str] = None) -> Dict[str, Any]:
        """Query the RAG system (integrated vector store)"""
        try:
            await self._ensure_connection()

            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(question)

            # Search for similar documents using integrated method
            results = await self._search_knowledge_base(query_embedding, course_id, top_k=5)

            # Convert results to Document objects for should_use_context and format_context
            docs = []
            sources = []

            for i, result in enumerate(results):
                content = result.get("text", "")
                if not content.strip():
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "material_id": result.get("material_id", ""),
                        "course_id": result.get("course_id", ""),
                        "filename": result.get("filename", ""),
                        "page": result.get("page", ""),
                        "score": result.get("score", 0.0),
                        "vector_distance": 1.0 - result.get("score", 0.0)  # Convert score to distance
                    }
                )
                docs.append(doc)

                if doc.metadata.get("vector_distance", 1.0) < settings.rag_distance_threshold:
                    sources.append({
                        "content": content,
                        "metadata": doc.metadata
                    })

            # Use format_context with hybrid threshold logic
            context = format_context(docs) if docs else "Tidak ada dokumen relevan ditemukan."

            # Generate response using LLM with token tracking
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            cost_usd = 0.0

            if context.strip():
                #Use RAG prompt with direct usage_metadata extraction
                prompt_messages = self.prompt.format_messages(
                    context=context,
                    question=question
                )

                llm_response = await self.llm.ainvoke(prompt_messages)
                answer = llm_response.content

                # Extract token usage from AIMessage usage_metadata
                usage = self._extract_usage_metadata(llm_response)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Calculate cost using TokenCounter
                model_name = settings.openai_model_comprehensive
                cost_usd = self.token_counter.calculate_cost(input_tokens, output_tokens, model_name)

                rag_logger.info(f"UnifiedRAGService.query tokens: input={input_tokens}, output={output_tokens}, cost=${cost_usd:.6f}")
            else:
                # No context found
                answer = "Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."

            return {
                "answer": answer,
                "context": context,  # Add formatted context for transparency
                "sources": sources,
                "course_id": course_id,
                "context_used": bool(context.strip()),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd
            }

        except Exception as e:
            rag_logger.error(f"RAG query failed: {e}")
            return {
                "answer": "Maaf, saya mengalami kesalahan saat memproses pertanyaan Anda.",
                "context": "Tidak ada knowledge base yang tersedia.",
                "sources": [],
                "course_id": course_id,
                "context_used": False
            }

    async def search_similar(self, query: str, course_id: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents (integrated vector store)"""
        try:
            await self._ensure_connection()

            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)

            # Search documents using integrated method
            return await self._search_knowledge_base(query_embedding, course_id, top_k=k)

        except Exception as e:
            rag_logger.error(f"Similar search failed: {e}")
            return []

    async def _search_knowledge_base(
        self,
        query_vector: List[float],
        course_id: Optional[str] = None,
        material_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant documents using RedisVL VectorQuery
        (Integrated method - no external vector store dependency)
        """
        try:
            # Build filter expression for course-based and material-based filtering
            filter_expression = None
            if course_id and material_ids:
                # Filter by both course_id AND material_id (most specific)
                from redisvl.query.filter import FilterExpression as FE
                filter_expression = (Tag("course_id") == course_id) & (Tag("material_id") == material_ids[0])
                rag_logger.info("🔍 Using course+material filter: course_id=%s, material_id=%s", course_id, material_ids[0])
            elif course_id:
                # Filter by course_id only (get all materials in course)
                filter_expression = Tag("course_id") == course_id
                rag_logger.info("🔍 Using course filter: course_id=%s", course_id)
            elif material_ids:
                # Fallback to material_id filtering only
                filter_expression = Tag("material_id") == material_ids[0]
                rag_logger.info("🔍 Using material filter: material_id=%s", material_ids[0])
            else:
                rag_logger.info("🔍 No filters - searching all documents")

            # Create VectorQuery following RedisVL best practices
            vector_query = VectorQuery(
                vector=query_vector,
                vector_field_name="vector",
                return_fields=["text", "material_id", "course_id", "filename", "page"],
                num_results=top_k,
                return_score=True
            )

            # Apply filter expression if we have one (course-based or file-based)
            if filter_expression:
                vector_query.filter_expression = filter_expression
                rag_logger.info("🔍 Applied filter expression to vector query")
            elif material_ids:
                # Fallback to legacy file hash filtering
                vector_query.filter_expression = self._build_filter_expression(material_ids)
                rag_logger.info("🔍 Applied legacy material filter expression")

            # Execute search
            result = await self.index.search(vector_query.query, query_params=vector_query.params)

            rag_logger.info("🔍 VECTOR SEARCH RESULTS: found %d documents total", len(result.docs))

            # Process results - take all top_k results (no threshold filtering)
            documents = []
            for doc in result.docs:
                vector_score = getattr(doc, "vector_distance", 1.0)
                # Convert to float if it's a string
                try:
                    distance = float(vector_score) if vector_score is not None else 1.0
                except (ValueError, TypeError):
                    distance = 1.0

                rag_logger.info("   📄 Document: material_id=%s, vector_distance=%.4f", getattr(doc, "material_id", "unknown"), distance)

                # RedisVL VectorQuery returns COSINE DISTANCE
                # Take all results from top_k (no threshold filtering)
                documents.append({
                    "text": getattr(doc, "text", ""),
                    "material_id": getattr(doc, "material_id", ""),
                    "course_id": getattr(doc, "course_id", ""),
                    "filename": getattr(doc, "filename", ""),
                    "page": getattr(doc, "page", ""),
                    "score": 1.0 - distance,  # Convert distance to similarity score
                    "vector_distance": distance
                })

            return documents

        except Exception as e:
            rag_logger.error("Failed to search knowledge base: %s", str(e))
            return []

    def _build_filter_expression(self, material_ids: List[str]) -> Optional[FilterExpression]:
        """
        Build filter OR expression for multiple material IDs using RedisVL Filter API
        (Tag("material_id") == "materi_A") | (Tag("material_id") == "materi_B")
        """
        if not material_ids:
            return None

        # Create individual filters and combine with OR
        filters = [Tag("material_id") == h for h in material_ids]

        # Combine filters with OR operator
        filter_expr = filters[0]
        for f in filters[1:]:
            filter_expr = filter_expr | f

        return filter_expr

    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            await self._ensure_connection()

            # Get total entries count
            knowledge_keys = await self.client.keys(f"{self.prefix}:*")

            return {
                "rag_service": "UnifiedRAGService (integrated)",
                "knowledge_base": {
                    "type": "document_store",
                    "redis_url": self.knowledge_base_url,
                    "index_name": self.index_name,
                    "prefix": self.prefix,
                    "entries": len(knowledge_keys),
                    "purpose": "RAG knowledge retrieval"
                },
                "vector_dimension": self.vector_dimension,
                "redisvl_available": REDISVL_AVAILABLE,
                "connected": self.connected
            }

        except Exception as e:
            rag_logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def get_documents_by_course_id(self, course_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all documents for a specific course_id"""
        try:
            await self._ensure_connection()

            filter_expr = Tag("course_id") == course_id

            filter_query = FilterQuery(
                return_fields=["text", "material_id", "course_id", "filename", "page"],
                filter_expression=filter_expr,
                num_results=limit
            )

            # Execute search
            result = await self.index.search(filter_query.query, query_params=filter_query.params)

            documents = []
            for doc in result.docs:
                documents.append({
                    "text": getattr(doc, "text", ""),
                    "material_id": getattr(doc, "material_id", ""),
                    "course_id": getattr(doc, "course_id", ""),
                    "filename": getattr(doc, "filename", ""),
                    "page": getattr(doc, "page", "")
                })

            rag_logger.info(f"Retrieved {len(documents)} documents for course_id: {course_id}")
            return documents

        except Exception as e:
            rag_logger.error(f"Failed to get documents by course_id {course_id}: {e}")
            return []

    async def get_documents_by_material_id(self, material_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all documents for a specific material_id"""
        try:
            await self._ensure_connection()

            filter_expr = Tag("material_id") == material_id

            filter_query = FilterQuery(
                return_fields=["text", "material_id", "course_id", "filename", "page"],
                filter_expression=filter_expr,
                num_results=limit
            )

            # Execute search
            result = await self.index.search(filter_query.query, query_params=filter_query.params)

            documents = []
            for doc in result.docs:
                documents.append({
                    "text": getattr(doc, "text", ""),
                    "material_id": getattr(doc, "material_id", ""),
                    "course_id": getattr(doc, "course_id", ""),
                    "filename": getattr(doc, "filename", ""),
                    "page": getattr(doc, "page", "")
                })

            rag_logger.info(f"Retrieved {len(documents)} documents for material_id: {material_id}")
            return documents

        except Exception as e:
            rag_logger.error(f"Failed to get documents by material_id {material_id}: {e}")
            return []

    async def delete_documents_by_course_id(self, course_id: str) -> int:
        """Delete all documents for a specific course_id using RedisVL search, returns count of deleted documents"""
        try:
            await self._ensure_connection()

            filter_expr = Tag("course_id") == course_id

            filter_query = FilterQuery(
                return_fields=["__id"],  # Only get document IDs
                filter_expression=filter_expr,
                num_results=10000  # Large limit to get all
            )

            # Execute search to get document IDs
            result = await self.index.search(filter_query.query, query_params=filter_query.params)

            if not result.docs:
                return 0

            # Extract document keys (IDs)
            keys_to_delete = [getattr(doc, "id", None) for doc in result.docs]
            keys_to_delete = [key for key in keys_to_delete if key]  # Remove None values

            if not keys_to_delete:
                return 0

            # Delete using pipeline for better performance
            pipeline = self.client.pipeline()
            for key in keys_to_delete:
                pipeline.delete(key)

            pipeline_results = await pipeline.execute()
            deleted_count = sum(pipeline_results)  # Sum all successful deletions

            rag_logger.info(f"Deleted {deleted_count} document chunks for course_id: {course_id}")
            return deleted_count

        except Exception as e:
            rag_logger.error(f"Failed to delete documents by course_id {course_id}: {e}")
            raise RedisException(f"Delete operation failed: {str(e)}")

    async def delete_documents_by_material_id(self, material_id: str) -> int:
        """Delete all documents for a specific material_id using RedisVL search, returns count of deleted documents"""
        try:
            await self._ensure_connection()

            filter_expr = Tag("material_id") == material_id

            filter_query = FilterQuery(
                return_fields=["__id"],  # Only get document IDs
                filter_expression=filter_expr,
                num_results=10000  # Large limit to get all
            )

            # Execute search to get document IDs
            result = await self.index.search(filter_query.query, query_params=filter_query.params)

            if not result.docs:
                return 0

            # Extract document keys (IDs)
            keys_to_delete = [getattr(doc, "id", None) for doc in result.docs]
            keys_to_delete = [key for key in keys_to_delete if key]  # Remove None values

            if not keys_to_delete:
                return 0

            # Delete using pipeline for better performance
            pipeline = self.client.pipeline()
            for key in keys_to_delete:
                pipeline.delete(key)

            pipeline_results = await pipeline.execute()
            deleted_count = sum(pipeline_results)  # Sum all successful deletions

            rag_logger.info(f"Deleted {deleted_count} document chunks for material_id: {material_id}")
            return deleted_count

        except Exception as e:
            rag_logger.error(f"Failed to delete documents by material_id {material_id}: {e}")
            raise RedisException(f"Delete operation failed: {str(e)}")

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        try:
            if self.client:
                await self.client.close()
            self.connected = False
            rag_logger.info("UnifiedRAGService disconnected from Redis")

        except Exception as e:
            rag_logger.error(f"Error during disconnect: {e}")


# Global singleton instance (maintaining compatibility)
unified_rag_service = UnifiedRAGService()


# =================================================================
# LCEL PATTERN IMPLEMENTATION
# =================================================================
# LangChain Expression Language pattern with hybrid threshold logic
# =================================================================

def should_use_context(docs: List[Document], threshold: float = None) -> bool:
    """
    Check if we should use context based on hybrid logic:
    - Always use the best document (doc[0])
    - Only use full context if docs 1-4 have good scores
    """
    from ..config.settings import settings

    # Use settings threshold if not provided
    if threshold is None:
        threshold = settings.rag_distance_threshold

    if not docs:
        return False

    # If we have less than 3 docs, use what we have
    if len(docs) < 3:
        return True

    # Check if secondary docs (1-4) have good scores
    for doc in docs[1:4]:
        if doc.metadata.get("vector_distance", 1.0) < threshold:
            return True

    return False


def format_context(docs: List[Document], threshold: float = None) -> str:
    """Format documents to context string with hybrid logic"""
    from ..config.settings import settings

    # Use settings threshold if not provided
    if threshold is None:
        threshold = settings.rag_distance_threshold

    if not docs:
        return "Tidak ada dokumen relevan ditemukan."

    # Always include the best document
    best_doc = docs[0]
    context_parts = [f"Dokumen 1 (score: {best_doc.metadata.get('score', 0):.3f}):\n{best_doc.page_content}"]

    # Add secondary docs only if they have good scores
    if should_use_context(docs, threshold):
        for i, doc in enumerate(docs[1:4], 2):  # docs 1-3 -> docs 2-4
            score = doc.metadata.get("score", 0.0)
            distance = doc.metadata.get("vector_distance", 1.0)
            if distance < threshold:
                context_parts.append(f"Dokumen {i} (score: {score:.3f}):\n{doc.page_content}")

    return "\n\n".join(context_parts)


# class LCELRAGService:
#     """LCEL-enhanced RAG service with hybrid threshold logic - using existing methods directly"""

#     def __init__(self, rag_service: Optional[UnifiedRAGService] = None):
#         from ..config.settings import settings

#         self.rag_service = rag_service or unified_rag_service
#         self.rag_threshold = settings.rag_distance_threshold  # Use settings threshold

#         # Create prompt template
#         self.rag_prompt = ChatPromptTemplate.from_template("""
#         Anda adalah AI Tutor Assistant yang membantu menjawab pertanyaan user pada sebuah Learning Management System
#         berdasarkan konteks atau knowledge base pada course yang diberikan.

#         Konteks:
#         {context}

#         Pertanyaan: {question}

#         Jawab pertanyaan berdasarkan konteks yang diberikan. Jika konteks tidak mengandung
#         informasi yang cukup untuk menjawab pertanyaan, katakan "Saya tidak memiliki informasi
#         yang cukup untuk menjawab pertanyaan ini."

#         Jawab dalam bahasa yang sama dengan pertanyaan.
#         """)

#         # Output parser
#         self.output_parser = StrOutputParser()

#         rag_logger.info("LCELRAGService initialized")

#     async def _retrieve_documents(self, query: str, course_id: Optional[str] = None, top_k: int = 5) -> List[Document]:
#         """
#         Retrieve documents directly using existing UnifiedRAGService methods.
#         Hybrid logic: top_k=5 + threshold validation for docs 1-3
#         """
#         try:
#             await self.rag_service._ensure_connection()

#             # Generate query embedding using existing embeddings
#             query_embedding = await self.rag_service.embeddings.aembed_query(query)

#             # Search using existing method
#             results = await self.rag_service._search_knowledge_base(
#                 query_vector=query_embedding,
#                 course_id=course_id,
#                 top_k=top_k
#             )

#             # Convert RedisVL results to LangChain Documents
#             documents = []
#             for result in results:
#                 content = result.get("text", "")
#                 metadata = {
#                     "material_id": result.get("material_id", ""),
#                     "course_id": result.get("course_id", ""),
#                     "filename": result.get("filename", ""),
#                     "score": result.get("score", 0.0),
#                     "vector_distance": result.get("vector_distance", 1.0)
#                 }
#                 documents.append(Document(page_content=content, metadata=metadata))

#             rag_logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
#             return documents

#         except Exception as e:
#             rag_logger.error(f"Document retrieval failed: {e}")
#             return []

#     def _create_rag_chain(self, documents: List[Document], threshold: float):
#         """
#         Create LCEL RAG chain with hybrid threshold logic.
#         Documents already retrieved, now create processing chain.
#         """
#         # Format context using hybrid logic
#         context = format_context(documents, threshold)

#         # LCEL chain: context + question -> prompt -> llm -> parser
#         rag_chain = (
#             RunnableParallel({
#                 "context": lambda _: context,
#                 "question": RunnablePassthrough()
#             })
#             | self.rag_prompt
#             | self.rag_service.llm
#             | self.output_parser
#         )

#         return rag_chain

#     async def query(self, question: str, course_id: Optional[str] = None,
#                    rag_threshold: Optional[float] = None, top_k: int = 5) -> Dict[str, Any]:
#         """Query using LCEL pattern with hybrid threshold logic"""
#         try:
#             start_time = time.time()
#             threshold = rag_threshold or self.rag_threshold

#             # Step 1: Retrieve documents (top_k=5 with hybrid threshold)
#             documents = await self._retrieve_documents(question, course_id, top_k)

#             if not documents:
#                 return {
#                     "answer": "Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini.",
#                     "sources": [],
#                     "course_id": course_id,
#                     "context_used": False,
#                     "context_quality": "none",
#                     "rag_threshold": threshold,
#                     "latency_ms": (time.time() - start_time) * 1000,
#                     "method": "lcel_no_docs"
#                 }

#             # Step 2: Create and execute RAG chain
#             rag_chain = self._create_rag_chain(documents, threshold)
#             answer = await rag_chain.ainvoke(question)

#             # Step 3: Determine context quality using hybrid logic
#             used_context = should_use_context(documents, threshold)
#             context_quality = "comprehensive" if used_context else "limited"

#             # Step 4: Format sources (filtered by hybrid logic)
#             sources = []

#             # Always include the best document (doc[0])
#             if documents:
#                 sources.append({
#                     "content": documents[0].page_content,
#                     "metadata": documents[0].metadata
#                 })

#             # Add secondary docs only if they pass threshold
#             if used_context:  # This means docs[1:4] have good scores
#                 for doc in documents[1:4]:  # docs 1-3 -> docs 2-4
#                     distance = doc.metadata.get("vector_distance", 1.0)
#                     if distance < threshold:
#                         sources.append({
#                             "content": doc.page_content,
#                             "metadata": doc.metadata
#                         })

#             response_time = (time.time() - start_time) * 1000
#             rag_logger.info(f"LCEL query completed in {response_time:.2f}ms, quality={context_quality}")

#             return {
#                 "answer": answer,
#                 "sources": sources,
#                 "course_id": course_id,
#                 "context_used": True,
#                 "context_quality": context_quality,
#                 "rag_threshold": threshold,
#                 "latency_ms": response_time,
#                 "method": "lcel"
#             }

#         except Exception as e:
#             rag_logger.error(f"LCEL query failed: {e}")
#             return {
#                 "answer": "Maaf, terjadi kesalahan saat memproses pertanyaan.",
#                 "sources": [],
#                 "course_id": course_id,
#                 "context_used": False,
#                 "error": str(e),
#                 "method": "lcel_error"
#             }

#     async def stream(self, question: str, course_id: Optional[str] = None,
#                     rag_threshold: Optional[float] = None, top_k: int = 5):
#         """Streaming version of LCEL query with hybrid threshold logic"""
#         try:
#             threshold = rag_threshold or self.rag_threshold

#             # Step 1: Retrieve documents first (non-streaming part)
#             documents = await self._retrieve_documents(question, course_id, top_k)

#             if not documents:
#                 yield "Saya tidak memiliki informasi yang cukup untuk menjawab pertanyaan ini."
#                 return

#             # Step 2: Create streaming chain with pre-retrieved context
#             context = format_context(documents, threshold)
#             streaming_chain = (
#                 RunnableParallel({
#                     "context": lambda _: context,
#                     "question": RunnablePassthrough()
#                 })
#                 | self.rag_prompt
#                 | self.rag_service.llm
#                 | self.output_parser
#             )

#             # Step 3: Stream LLM response
#             async for chunk in streaming_chain.astream(question):
#                 if chunk:
#                     yield chunk

#         except Exception as e:
#             rag_logger.error(f"LCEL streaming failed: {e}")
#             yield f"Error: {str(e)}"

#     def set_threshold(self, threshold: float):
#         """Update RAG threshold"""
#         self.rag_threshold = threshold
#         rag_logger.info(f"RAG threshold updated to {threshold}")


# # Global LCEL service instance
# lcel_rag_service = LCELRAGService()


# =====================================
# INTEGRASI RAGService + Database + LCEL
# =====================================

class RAGService():
    """
    RAG Service dengan LangChain Expression Language + Database Integration
    """

    def __init__(self):
        """Initialize RAG service dengan LCEL pattern"""
        # Use existing RAG service untuk knowledge base
        self.rag_service = UnifiedRAGService()

        # Import components
        from ..core.telemetry import TokenCounter
        from ..core.database import async_db

        self.db_session = async_db.get_session
        self.rag_logger = rag_logger
        self.token_counter = TokenCounter()

        # Initialize LangChain components
        self._setup_lcel_components()

    def _setup_lcel_components(self):
        """Setup LangChain Expression Language components"""
        from langchain_core.prompts import ChatPromptTemplate

        # Main RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template("""
        Anda adalah AI Tutor Assistant yang membantu menjawab pertanyaan user pada Learning Management System
        berdasarkan konteks atau knowledge base pada course yang diberikan.

        Knowledge Base:
        {knowledge_base}

        Pertanyaan: {question}

        Jawab pertanyaan berdasarkan knowledge base yang diberikan. Jika knowledge base tidak mengandung
        informasi yang cukup untuk menjawab pertanyaan, katakan "Saya tidak memiliki informasi
        yang cukup untuk menjawab pertanyaan ini."

        Jawab dalam bahasa yang sama dengan pertanyaan.
        """)

        # Personalization prompt template
        self.personalization_prompt = ChatPromptTemplate.from_template("""
        Anda adalah personalization assistant. Response berikut telah dihasilkan untuk user query.

        Query Original: {original_query}
        Response yang Ada: {base_response}

        User Context:
        {user_context}

        Conversation History:
        {history}

        Personalisasi response tersebut sesuai dengan user context dan conversation history.
        Pertahankan esensi informasi yang sama, tapi sesuaikan dengan preferensi user.

        Response Personalisasi:
        """)

        # Output parser
        self.output_parser = StrOutputParser()

    async def generate_response(
        self,
        question: str,
        course_id: Optional[str] = None,
        chatroom_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_personalization: bool = False
    ) -> Dict[str, Any]:
        """
        Generate response menggunakan LangChain Expression Language dengan database tracking.
        """
        try:
            start_time = time.time()

            # Step 1: Get context components
            rag_result = await self._get_knowledge_base_context(question, course_id)
            user_context_text, history_text = await self._get_context_components(
                chatroom_id, user_id, course_id
            ) if use_personalization else ("", "")

            # Step 2: Build LCEL chain using LangChain ChatOpenAI
            from langchain_openai import ChatOpenAI
            chat_openai = ChatOpenAI(
                model=settings.openai_model_comprehensive,
                temperature=settings.openai_temperature,
                api_key=settings.openai_api_key
            )

            rag_chain = (
                RunnableParallel({
                    "knowledge_base": lambda _: rag_result.get("context", ""),
                    "user_context": lambda _: user_context_text,
                    "history": lambda _: history_text,
                    "question": RunnablePassthrough()
                })
                | self.rag_prompt
                | chat_openai
                | self.output_parser
            )

            # Step 3: Generate response with token tracking
            input_tokens = 0
            output_tokens = 0
            cost_usd = 0.0

            chain_without_parser = (
                RunnableParallel({
                    "knowledge_base": lambda _: rag_result.get("context", ""),
                    "user_context": lambda _: user_context_text,
                    "history": lambda _: history_text,
                    "question": RunnablePassthrough()
                })
                | self.rag_prompt
                | chat_openai
            )

            llm_response = await chain_without_parser.ainvoke(question)
            response_text = llm_response.content

            # Extract token usage from AIMessage usage_metadata
            usage = self.rag_service._extract_usage_metadata(llm_response)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Calculate cost using TokenCounter
            model_name = settings.openai_model_comprehensive
            cost_usd = self.token_counter.calculate_cost(input_tokens, output_tokens, model_name)

            rag_logger.info(f"RAGService.generate_response tokens: input={input_tokens}, output={output_tokens}, cost=${cost_usd:.6f}")

            # Step 4: Handle personalization if needed
            final_response = response_text
            model_used = settings.openai_model_comprehensive  # Use comprehensive model for RAG
            personalization_tokens = {"input": 0, "output": 0, "cost": 0.0}

            if use_personalization:
                personalization_result = await self._personalize_response(
                    response_text, user_context_text, history_text, question
                )
                final_response = personalization_result.get("response", response_text)
                model_used = personalization_result.get("model_used", model_used)

                # Add personalization tokens
                personalization_tokens = personalization_result.get("tokens", {"input": 0, "output": 0, "cost": 0.0})
                input_tokens += personalization_tokens.get("input", 0)
                output_tokens += personalization_tokens.get("output", 0)
                cost_usd += personalization_tokens.get("cost", 0.0)

            # Step 5: Calculate metrics
            latency_ms = (time.time() - start_time) * 1000

            return {
                "response": final_response,
                "model_used": model_used,
                "response_type": "rag_response",
                "source_type": "knowledge_base",
                "knowledge_base_used": bool(rag_result.get("context")),
                "user_context_used": bool(user_context_text),
                "history_used": bool(history_text),
                "source_documents": rag_result.get("sources", []),
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": cost_usd
            }

        except Exception as e:
            self.rag_logger.error(f"RAGService generation failed: {e}")
            return {
                "response": "Maaf, saya mengalami kesalahan dalam memproses permintaan Anda.",
                "error": str(e),
                "response_type": "error"
            }

    async def _get_knowledge_base_context(self, question: str, course_id: Optional[str] = None) -> Dict[str, Any]:
        """Get knowledge base context using existing RAG service"""
        return await self.rag_service.query(question, course_id)

    async def _get_context_components(
        self,
        chatroom_id: str,
        user_id: str,
        course_id: str
    ) -> tuple[str, str]:
        """Get user context and conversation history from database"""
        from ..schemas.db_models import UserContext, Chatroom

        async with self.db_session() as db:
            # Get user context
            user_context = await UserContext.aget_or_create(db, user_id, course_id)
            user_context_text = user_context.get_context()

            # Get conversation history
            chatroom = await db.get(Chatroom, chatroom_id)
            history_text = ""

            if chatroom:
                history_text = await chatroom.get_conversation_history(db, limit=5)

            return user_context_text, history_text


    async def _personalize_response(
        self,
        base_response: str,
        user_context: str,
        history: str,
        original_query: str
    ) -> Dict[str, Any]:
        """Personalize response using personalization model (gpt-4.1-nano)"""
        try:
           # Use personalization model specifically with stream_usage=True
            from langchain_openai import ChatOpenAI
            personalization_llm = ChatOpenAI(
                model=settings.openai_model_personalized,
                temperature=settings.openai_temperature,
                streaming=False,
                openai_api_key=settings.openai_api_key
            )

            # Build personalization streaming chain
            personalization_chain = (
                RunnableParallel({
                    "base_response": lambda _: base_response,
                    "original_query": lambda _: original_query,
                    "user_context": lambda _: user_context,
                    "history": lambda _: history
                })
                | self.personalization_prompt
                | personalization_llm
            )

            # Gunakan ainvoke sederhana:
            llm_response = await personalization_chain.ainvoke({})
            personalized_response = llm_response.content

            # Extract token usage from LLM response
            full_chunk = llm_response
            if full_chunk:
                usage = self.rag_service._extract_usage_metadata(full_chunk)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cost_usd = self.token_counter.calculate_cost(
                    input_tokens,
                    output_tokens,
                    settings.openai_model_personalized,
                )
                tokens = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "cost": cost_usd
                }
                rag_logger.info(f"RAGService._personalize_response tokens: input={input_tokens}, output={output_tokens}, cost=${cost_usd:.6f}")

            return {
                "response": personalized_response,
                "model_used": settings.openai_model_personalized,
                "tokens": tokens
            }

        except Exception as e:
            self.rag_logger.error(f"Personalization failed: {e}")
            return {
                "response": base_response,  # Fallback
                "model_used": settings.openai_model_comprehensive,
                "tokens": {"input": 0, "output": 0, "cost": 0.0}
            }

    async def generate_response_stream(
        self,
        question: str,
        course_id: Optional[str] = None,
        chatroom_id: Optional[str] = None,
        user_id: Optional[str] = None,
        use_personalization: bool = False
    ):
        """
        Streaming version of generate_response with consistent parameters.
        Yields dict dengan 'type' untuk membedakan content vs metadata.
        Handles RAG + optional personalization with streaming.
        """
        try:
            # Step 1: Get context components
            rag_result = await self._get_knowledge_base_context(question, course_id)

            # Step 2: Build RAG streaming chain with LangChain ChatOpenAI (stream_usage=True)
            from langchain_openai import ChatOpenAI
            chat_openai = ChatOpenAI(
                model=settings.openai_model_comprehensive,
                temperature=settings.openai_temperature,
                api_key=settings.openai_api_key,
                streaming=True,
                stream_usage=True  # Enable usage metadata in streaming
            )

            rag_streaming_chain = (
                RunnableParallel({
                    "knowledge_base": lambda _: rag_result.get("answer", ""),
                    "question": RunnablePassthrough()
                })
                | self.rag_prompt
                | chat_openai
            )

            # Step 3: Stream RAG response with token tracking
            full_response = ""
            full_chunk = None

            async for chunk in rag_streaming_chain.astream(question):
                if chunk:
                    if full_chunk is None:
                        full_chunk = chunk
                    else:
                        full_chunk += chunk

                    # Yield content chunks
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        yield {
                            "type": "content",
                            "data": chunk.content
                        }

            # Extract and yield token usage metadata
            if full_chunk:
                usage = self.rag_service._extract_usage_metadata(full_chunk)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cost_usd = self.token_counter.calculate_cost(
                    input_tokens,
                    output_tokens,
                    settings.openai_model_comprehensive,
                )
                rag_logger.info(f"RAGService.stream RAG tokens: input={input_tokens}, output={output_tokens}, cost=${cost_usd:.6f}")

                # Yield RAG metadata
                yield {
                    "type": "metadata",
                    "data": {
                        "source": "rag",
                        "model_used": settings.openai_model_comprehensive,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "cost_usd": cost_usd,
                        "response_type": "rag_response",
                        "source_type": "knowledge_base",
                        "source_documents": rag_result.get("sources", [])
                    }
                }

            # Step 4: Handle personalization if requested
            if use_personalization:
                user_context_text, history_text = await self._get_context_components(
                    chatroom_id, user_id, course_id
                )

                if user_context_text or history_text:
                    # Use dedicated streaming personalization method with token tracking
                    async for item in self._personalize_response_stream(
                        full_response, user_context_text, history_text, question
                    ):
                        # Forward content and metadata from personalization
                        yield item

        except Exception as e:
            self.rag_logger.error(f"RAGService streaming failed: {e}")
            yield {
                "type": "error",
                "data": f"Maaf, terjadi kesalahan saat memproses permintaan: {str(e)}"
            }

    async def _personalize_response_stream(
        self,
        base_response: str,
        user_context: str,
        history: str,
        original_query: str
    ):
        """Streaming version of response personalization using GPT-4.1-nano.

        Yields dict dengan 'type' untuk membedakan content vs metadata.
        """
        try:
            # Use GPT-4.1-nano specifically for personalization
            from langchain_openai import ChatOpenAI
            personalization_llm = ChatOpenAI(
                model=settings.openai_model_personalized,
                temperature=settings.openai_temperature,
                streaming=True,
                openai_api_key=settings.openai_api_key
            )

            # Build personalization chain without output parser to get AIMessage
            personalization_chain = (
                RunnableParallel({
                    "base_response": lambda _: base_response,
                    "original_query": lambda _: original_query,
                    "user_context": lambda _: user_context,
                    "history": lambda _: history
                })
                | self.personalization_prompt
                | personalization_llm
            )

            # Stream personalized response and yield chunks
            personalized_response = ""
            full_chunk = None

            async for chunk in personalization_chain.astream({
                "base_response": base_response,
                "original_query": original_query,
                "user_context": user_context,
                "history": history
            }):
                if chunk:
                    if full_chunk is None:
                        full_chunk = chunk
                    else:
                        full_chunk += chunk

                    # Yield content chunks
                    if hasattr(chunk, 'content') and chunk.content:
                        personalized_response += chunk.content
                        yield {
                            "type": "content",
                            "data": chunk.content
                        }

            # Extract and yield token usage metadata
            if full_chunk:
                usage = self.rag_service._extract_usage_metadata(full_chunk)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cost_usd = self.token_counter.calculate_cost(
                    input_tokens,
                    output_tokens,
                    settings.openai_model_personalized,
                )
                rag_logger.info(f"RAGService._personalize_response_stream tokens: input={input_tokens}, output={output_tokens}, cost=${cost_usd:.6f}")

                # Yield personalization metadata
                yield {
                    "type": "metadata",
                    "data": {
                        "source": "cache_personalized",
                        "model_used": settings.openai_model_personalized,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "cost_usd": cost_usd,
                        "response_type": "cache_hit_personalized",
                        "source_type": "redis_cache"
                    }
                }

        except Exception as e:
            self.rag_logger.error(f"Streaming personalization failed: {e}")
            # Fallback to base response without personalization
            yield {
                "type": "content",
                "data": base_response
            }
            yield {
                "type": "metadata",
                "data": {
                    "source": "cache_personalized",
                    "model_used": settings.openai_model_personalized,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "response_type": "cache_hit_personalized",
                    "source_type": "redis_cache"
                }
            }

    async def store_in_database(
        self,
        message_id: str,
        chatroom_id: str,
        user_id: str,
        response_data: Dict[str, Any]
    ) -> bool:
        """Store message and response tracking in database"""
        try:
            from ..schemas.db_models import Response

            async with self.db_session() as db:
                Response.create_response(
                    db=db,
                    message_id=message_id,
                    chatroom_id=chatroom_id,
                    user_id=user_id,
                    response_text=response_data["response"],
                    model_used=response_data["model_used"],
                    response_type=response_data["response_type"],
                    source_type=response_data["source_type"],
                    latency_ms=response_data["latency_ms"],
                    input_tokens=response_data.get("input_tokens", 0),
                    output_tokens=response_data.get("output_tokens", 0),
                    cost_usd=response_data.get("cost_usd", 0.0),
                    cache_hit=response_data.get("cache_hit", False),
                    cache_similarity_score=response_data.get("cache_similarity_score"),
                    personalized=response_data.get("user_context_used", False)
                )

            return True

        except Exception as e:
            self.rag_logger.error(f"Failed to store in database: {e}")
            return False

# Global RAG service instance with database integration
rag_service = RAGService()