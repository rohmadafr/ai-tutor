from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model_comprehensive: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL_COMPREHENSIVE")
    openai_model_personalized: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL_PERSONALIZED")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.5, env="OPENAI_TEMPERATURE")

    # PostgreSQL Configuration
    postgres_db: str = Field(default="ai_db", env="POSTGRES_DB")
    postgres_user: str = Field(default="lms_ai", env="POSTGRES_USER")
    postgres_password: str = Field(default="aiASLI1234", env="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="10.101.20.220", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5434, env="POSTGRES_PORT")

    # Redis Configuration
    # Semantic Cache (port 6380)
    redis_cache_url: str = Field(default="redis://localhost:6380", env="REDIS_CACHE_URL")
    # Knowledge Base/RAG (port 6379)
    redis_knowledge_url: str = Field(default="redis://localhost:6379", env="REDIS_KNOWLEDGE_URL")

    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")

    # Vector Store Configuration
    vector_index_name: str = Field(default="semantic_cache", env="VECTOR_INDEX_NAME")
    vector_dimension: int = Field(default=1536, env="VECTOR_DIMENSION")
    vector_distance_metric: str = Field(default="cosine", env="VECTOR_DISTANCE_METRIC")
    vector_batch_size: int = Field(default=100, env="VECTOR_BATCH_SIZE")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_debug: bool = Field(default=False, env="API_DEBUG")
    api_reload: bool = Field(default=False, env="API_RELOAD")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_debug: bool = Field(default=False, env="LOG_DEBUG")
    log_dir: Path = Field(default="logs", env="LOG_DIR")

    # Semantic Cache Configuration
    cache_threshold: float = Field(default=0.3, env="CACHE_THRESHOLD")  # Vector distance <= 0.3 = good match cache -> hit
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_results: int = Field(default=5, env="CACHE_MAX_RESULTS")
    cache_context_window: int = Field(default=10, env="CACHE_CONTEXT_WINDOW")

    # RAG Configuration (lenient - for document retrieval)
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")
    rag_chunk_size: int = Field(default=1000, env="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, env="RAG_CHUNK_OVERLAP")
    rag_rerank: bool = Field(default=False, env="RAG_RERANK")
    rag_distance_threshold: float = Field(default=0.3, env="RAG_DISTANCE_THRESHOLD")  # Vector distance <= 0.3 = good match retrieval -> hit

    # Telemetry Configuration
    telemetry_enabled: bool = Field(default=True, env="TELEMETRY_ENABLED")
    telemetry_track_costs: bool = Field(default=True, env="TELEMETRY_TRACK_COSTS")
    telemetry_track_latency: bool = Field(default=True, env="TELEMETRY_TRACK_LATENCY")

    # Security Configuration
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"],
        env="CORS_ORIGINS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()