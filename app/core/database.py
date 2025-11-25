# File: app/core/database.py
"""
Async PostgreSQL Database Connection using AsyncPG
For AI-Tutor Chatbot integration with existing LMS database
"""
from typing import Optional, AsyncGenerator, Any, Dict
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from contextlib import asynccontextmanager

from ..config.settings import settings
from .logger import app_logger

# Create Base class for SQLAlchemy models
Base = declarative_base()

class AsyncDatabase:
    """
    Async PostgreSQL database manager using asyncpg driver.
    Handles connection pooling, session management, and basic CRUD operations.
    """

    def __init__(self):
        """Initialize database connection settings."""
        self.engine = None
        self.session_factory = None
        self._connected = False

        # Database connection string
        self.database_url = self._build_connection_string()

        app_logger.info("AsyncDatabase initialized with PostgreSQL connection")

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from settings."""
        try:
            # Use existing PostgreSQL settings
            postgres_config = {
                "user": settings.postgres_user,
                "password": settings.postgres_password,
                "host": settings.postgres_host,
                "port": settings.postgres_port,
                "database": settings.postgres_db
            }

            # Build connection string for asyncpg + SQLAlchemy
            connection_string = (
                f"postgresql+asyncpg://{postgres_config['user']}:{postgres_config['password']}"
                f"@{postgres_config['host']}:{postgres_config['port']}/{postgres_config['database']}"
            )

            app_logger.info(f"Database connection configured for {postgres_config['database']} at {postgres_config['host']}:{postgres_config['port']}")
            return connection_string

        except Exception as e:
            app_logger.error(f"Failed to build database connection string: {e}")
            raise

    async def connect(self) -> None:
        """Establish database connection and create session factory."""
        if self._connected:
            try:
                # Test existing connection
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                app_logger.info("Database connection already active")
                return
            except Exception:
                app_logger.warning("Existing connection failed, reconnecting...")
                await self.disconnect()

        try:
            # Create async engine with optimal settings for AI-Tutor
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.log_debug,  # Enable SQL logging in debug mode
                # Use QueuePool for SQLAlchemy-level pooling with asyncpg
                pool_size=10,  # Number of connections to maintain in pool
                max_overflow=20,  # Additional connections when pool is full
                pool_pre_ping=True,  # Test connections before use
                pool_recycle=3600,  # Recycle connections after 1 hour
                # Connection settings optimized for async operations
                connect_args={
                    "server_settings": {
                        "application_name": "ai-tutor-chatbot",
                        "jit": "off",  # Disable JIT for simpler query plans
                    },
                    "command_timeout": 60,  # Command timeout in seconds
                }
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )

            # Test connection
            await self._test_connection()

            self._connected = True
            app_logger.info("✅ Async PostgreSQL database connected successfully")

        except Exception as e:
            app_logger.error(f"❌ Failed to connect to database: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Close database connection."""
        try:
            if self.engine:
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None
                self._connected = False
                app_logger.info("Database connection closed")
        except Exception as e:
            app_logger.error(f"Error closing database connection: {e}")

    async def _test_connection(self) -> None:
        """Test database connection with simple query."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                app_logger.info(f"Database connection test successful: {version[:50]}...")
        except Exception as e:
            app_logger.error(f"Database connection test failed: {e}")
            raise

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic connection management.

        Usage:
            async with db.get_session() as session:
                # Use session for database operations
                result = await session.execute(query)
                data = result.scalars().all()
        """
        if not self._connected:
            await self.connect()

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                app_logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a raw SQL query and return results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query results
        """
        async with self.get_session() as session:
            try:
                result = await session.execute(text(query), params or {})
                if result.returns_rows:
                    return result.mappings().all()
                return result.rowcount
            except Exception as e:
                app_logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
                raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health and return status information.

        Returns:
            Database health status and metrics
        """
        try:
            if not self._connected:
                return {
                    "status": "disconnected",
                    "connected": False,
                    "error": "Database not connected"
                }

            # Test basic connectivity
            async with self.get_session() as session:
                # Get database version
                version_result = await session.execute(text("SELECT version()"))
                version = version_result.scalar()

                # Get connection info
                conn_info_result = await session.execute(text("""
                    SELECT
                        count(*) as active_connections
                    FROM pg_stat_activity
                    WHERE state = 'active' AND datname = current_database()
                """))
                active_connections = conn_info_result.scalar()

                # Get database size
                size_result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                """))
                db_size = size_result.scalar()

                # Get table counts for our schema
                table_counts = {}
                our_tables = ['users', 'courses', 'chatrooms', 'messages', 'responses', 'user_contexts']

                for table in our_tables:
                    try:
                        count_result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = count_result.scalar()
                        if count > 0:
                            table_counts[table] = count
                    except Exception:
                        # Table might not exist yet
                        continue

                return {
                    "status": "healthy",
                    "connected": True,
                    "version": str(version)[:50] + "...",
                    "active_connections": active_connections,
                    "database_size": db_size,
                    "table_counts": table_counts,
                    "connection_pool_size": self.engine.pool.size() if self.engine else 0,
                    "max_overflow": self.engine.pool.max_overflow if self.engine else 0
                }

        except Exception as e:
            return {
                "status": "error",
                "connected": self._connected,
                "error": str(e)
            }

    async def initialize_tables(self) -> bool:
        """
        Initialize database tables by creating all tables from models.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import all models to ensure they are registered with Base
            from ..schemas.db_models import (
                Base, User, Course, Document, Content, Summary, RequestTracking,
                ApiLog, QuestionGeneration, AnswerKey, QuestionOption, GeneratedQuestion,
                Chatroom, Message, Response, UserContext, CourseKnowledgeBase
            )

            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            app_logger.info("✅ Database tables initialized successfully")
            return True

        except Exception as e:
            app_logger.error(f"❌ Failed to initialize database tables: {e}")
            return False

    async def get_connection_info(self) -> Dict[str, Any]:
        """
        Get detailed connection information for debugging.

        Returns:
            Connection and pool information
        """
        if not self.engine:
            return {
                "connected": False,
                "error": "No database engine"
            }

        try:
            pool = self.engine.pool
            return {
                "connected": self._connected,
                "database_url": self.database_url.split('@')[-1],  # Hide password
                "driver": "asyncpg",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "max_overflow": pool.max_overflow(),
                "timeout": pool.timeout(),
                "invalidated": pool.invalidated(),
            }
        except Exception as e:
            return {
                "connected": self._connected,
                "error": str(e)
            }

# Global database instance
async_db = AsyncDatabase()

# Database dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session.

    Usage in FastAPI endpoint:
    @app.get("/users/")
    async def get_users(db: AsyncSession = Depends(get_db_session)):
        users = await db.execute(select(User))
        return users.scalars().all()
    """
    async with async_db.get_session() as session:
        yield session

# Convenience functions for database operations
class AsyncCRUD:
    """
    Base class for async CRUD operations.
    Provides common database operation patterns.
    """

    def __init__(self, model_class):
        """Initialize with SQLAlchemy model class."""
        self.model_class = model_class

    async def create(self, data: Dict[str, Any]) -> Any:
        """Create new record."""
        async with async_db.get_session() as session:
            instance = self.model_class(**data)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

    async def get_by_id(self, record_id: str) -> Optional[Any]:
        """Get record by ID."""
        async with async_db.get_session() as session:
            result = await session.get(self.model_class, record_id)
            return result

    async def update(self, record_id: str, data: Dict[str, Any]) -> Optional[Any]:
        """Update record by ID."""
        async with async_db.get_session() as session:
            instance = await session.get(self.model_class, record_id)
            if instance:
                for key, value in data.items():
                    setattr(instance, key, value)
                await session.commit()
                await session.refresh(instance)
            return instance

    async def delete(self, record_id: str) -> bool:
        """Delete record by ID."""
        async with async_db.get_session() as session:
            instance = await session.get(self.model_class, record_id)
            if instance:
                await session.delete(instance)
                return True
            return False

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[Any]:
        """Get all records with pagination."""
        async with async_db.get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(self.model_class)
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()

# Database initialization function
async def init_database() -> bool:
    """
    Initialize database connection and create tables.
    Should be called during application startup.

    Returns:
        True if initialization successful
    """
    try:
        # Connect to database
        await async_db.connect()

        # Create tables
        success = await async_db.initialize_tables()

        if success:
            app_logger.info("🚀 Database initialization completed successfully")

        return success

    except Exception as e:
        app_logger.error(f"❌ Database initialization failed: {e}")
        return False

# Database cleanup function
async def cleanup_database():
    """
    Cleanup database connections.
    Should be called during application shutdown.
    """
    try:
        await async_db.disconnect()
        app_logger.info("🔒 Database cleanup completed")
    except Exception as e:
        app_logger.error(f"❌ Database cleanup failed: {e}")