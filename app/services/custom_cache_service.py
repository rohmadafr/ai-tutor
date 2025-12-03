"""
Context-Enabled Semantic Cache Service
Following the exact implementation pattern from @cesc_redis_ai.ipynb
"""
import os
import time
import uuid
import redis.asyncio as redis
from typing import List, Dict, Any
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, FilterExpression
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import OpenAITextVectorizer

from ..config.settings import settings
from ..core.logger import cache_logger
from ..core.telemetry import TokenCounter, TelemetryLogger

class CustomCacheService:
    """
    Context-Enabled Semantic Cache following the exact implementation pattern
    from @cesc_redis_ai.ipynb reference
    """

    def __init__(self):
        """Initialize custom cache service with all required components"""
        # Ensure OpenAI API key is available
        if settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        # Initialize core components
        self.vectorizer = OpenAITextVectorizer(
            model=settings.openai_embedding_model
        )

        self.token_counter = TokenCounter()
        self.telemetry = TelemetryLogger()

        # Cache configuration
        self.cache_ttl = settings.cache_ttl or 3600
        self.redis_url = settings.redis_cache_url

        # Redis and index setup
        self.client: redis.Redis = None
        self.index: AsyncSearchIndex = None
        self.index_name = "cesc_index"
        self.prefix = "cesc"  # Using same prefix as reference
        self.vector_dimension = 1536  # OpenAI embedding dimension

        # User memories for personalization
        self.user_memories: Dict[str, Dict] = {}

        # Connection flag
        self.connected = False

    def _define_cache_schema(self) -> IndexSchema:
        """Define Redis index schema following reference pattern"""
        schema_definition = {
            "index": {
                "name": self.index_name,
                "prefix": self.prefix,
                "storage_type": "hash"
            },
            "fields": [
                # Content fields
                {"name": "response", "type": "text"},
                {"name": "user_id", "type": "tag"},
                {"name": "course_id", "type": "tag"},
                {"name": "prompt", "type": "text"},
                {"name": "model", "type": "tag"},
                {"name": "created_at", "type": "numeric"},

                # Source information from RAG - stored as JSON string (array of objects)
                {"name": "sources", "type": "text"},

                # Vector field - using prompt_vector as in reference
                {
                    "name": "prompt_vector",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "HNSW",
                        "dims": self.vector_dimension,
                        "distance_metric": "cosine",
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
                pass

        try:
            # Connect to Redis
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            await self.client.ping()

            # Create or load index
            schema = self._define_cache_schema()
            self.index = AsyncSearchIndex(schema=schema)

            # Check if index exists
            try:
                await self.index.set_client(self.client)
                await self.index.create(overwrite=False)
                cache_logger.info("Context-Enabled Semantic Cache index created")
            except Exception as e:
                # Index might already exist
                if "already exists" in str(e).lower():
                    cache_logger.info("Context-Enabled Semantic Cache index already exists")
                    await self.index.set_client(self.client)
                else:
                    raise e

            self.connected = True
            cache_logger.info("CustomCacheService connected successfully")

        except Exception as e:
            cache_logger.error(f"Failed to connect custom cache: {e}")
            raise

    async def _ensure_connection(self):
        """Ensure connection to Redis"""
        if not self.connected:
            await self.connect()

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (following reference pattern)"""
        return await self.vectorizer.aembed(text)

    async def search_cache(
        self,
        embedding: List[float],
        course_id: str = None,
        distance_threshold: float = None, # Use settings.cache_threshold by default
    ):
        """
        Find the best cached match and gate it by a distance threshold.
        The score returned by RediSearch (HNSW + cosine) is a distance (lower is better).
        We accept a hit if distance <= distance_threshold.
        """
        # Use settings threshold if not provided
        if distance_threshold is None:
            distance_threshold = settings.cache_threshold

        await self._ensure_connection()

        try:
            import json

            filter_expr = Tag("course_id") == "global"
            if course_id:
                filter_expr = (Tag("course_id") == course_id) | (Tag("course_id") == "global")

            return_fields = ["response", "user_id", "course_id", "prompt", "model", "created_at", "sources"]
            query = VectorQuery(
                vector=embedding,
                vector_field_name="prompt_vector",  # Using prompt_vector as in reference
                return_fields=return_fields,
                num_results=1,
                return_score=True,
                filter_expression = filter_expr
            )
            results = await self.index.search(query.query, query_params=query.params)

            if results and len(results.docs) > 0:
                first = results.docs[0]
                # Use 'vector_distance' which is the standard score field in redisvl
                score = getattr(first, 'vector_distance', None)
                try:
                    score_float = float(score) if score is not None else None
                    # Handle floating point precision issues with negative zero
                    if score_float is not None:
                        # Convert -0.0000 to 0.0000 for proper comparison
                        if abs(score_float) < 1e-10:
                            score_float = 0.0
                        if score_float <= distance_threshold:
                            cache_logger.info(f"🎯 Cache hit with distance: {score_float:.4f}")
                            result = {field: getattr(first, field, '') for field in return_fields}
                            # Parse sources from JSON string
                            if result.get("sources"):
                                try:
                                    result["sources"] = json.loads(result["sources"])
                                except json.JSONDecodeError as e:
                                    cache_logger.error(f"Cache HIT sources JSON decode error: {e}")
                                    result["sources"] = []
                            else:
                                result["sources"] = []
                                cache_logger.warning("Cache HIT sources field is missing or empty")
                            return result
                except (ValueError, TypeError):
                    pass
            return None
        except Exception as e:
            cache_logger.error(f"Cache search failed: {e}")
            return None

    async def store_response(self, prompt: str, response: str, embedding: List[float], user_id: str, model: str, course_id: str = None, sources: List[Dict] = None):
        """Store response in cache (following reference pattern)"""
        await self._ensure_connection()

        try:
            import numpy as np
            import json
            vec_bytes = np.array(embedding, dtype=np.float32).tobytes()

            doc = {
                "response": response,
                "prompt_vector": vec_bytes,  # Using prompt_vector as in reference
                "user_id": user_id,
                "course_id": course_id or "global",
                "prompt": prompt,
                "model": model,
                "created_at": int(time.time()),
                "sources": json.dumps(sources or [])  # Always store sources (even if empty)
            }

            # Use unique key for each entry and set TTL
            key = f"{self.prefix}:{uuid.uuid4()}"
            await self.index.load([doc], keys=[key])

            # Set TTL if specified
            if self.cache_ttl > 0:
                await self.client.expire(key, self.cache_ttl)

            cache_logger.info(f"✅ Cached response: key={key[:20]}..., user_id={user_id}, course_id={course_id}, sources={len(sources or [])}")
            return True

        except Exception as e:
            cache_logger.error(f"Failed to cache response: {e}")
            return False

    async def query(self, prompt: str, user_id: str, course_id: str = None):
        """Main query method following the exact reference pattern"""
        start_time = time.time()
        embedding = await self.generate_embedding(prompt)
        cached_result = await self.search_cache(embedding, course_id)

        if cached_result:
            # Get cached response
            cached_response = cached_result["response"]

            # For now, skip personalization for cached responses
            result = {
                "response": cached_response,
                "model": cached_result.get("model", "unknown"),
                "latency_ms": int((time.time() - start_time) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "sources": cached_result.get("sources", [])
            }

            # Log cache hit
            self.telemetry.log(
                user_id=user_id,
                method="context_query",
                latency_ms=result["latency_ms"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                cache_status="hit",
                response_source=result["model"]
            )

            cache_logger.info(f"✅ Cache HIT: latency_ms={result['latency_ms']}")
            return result

        else:
            # Cache miss - return embedding to reuse
            cache_logger.info("❌ Cache MISS")
            return {
                "embedding": embedding,
                "cache_status": "miss"
            }

    async def clear_user_cache(self, user_id: str) -> int:
        """Clear cache for specific user"""
        # TODO: Implement user-specific cache clearing
        cache_logger.info(f"Clearing cache for user: {user_id}")
        return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                "cache_type": "Context-Enabled Semantic Cache",
                "redis_url": self.redis_url,
                "index_name": self.index_name,
                "prefix": self.prefix,
                "vector_dimension": self.vector_dimension,
                "distance_metric": "COSINE",
                "ttl": self.cache_ttl,
                "status": "active" if self.connected else "inactive",
                "user_memories_count": len(self.user_memories)
            }
        except Exception as e:
            cache_logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}