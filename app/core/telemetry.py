from typing import Dict, Any, List, Optional
import time
import tiktoken
import json

from ..models.telemetry import TelemetryEvent, TelemetryMetrics, CostTracker
from ..config.settings import settings
from .logger import telemetry_logger

# For backward compatibility with Document and batch processing
try:
    from langchain.schema import Document
except ImportError:
    # Fallback if langchain not available
    class Document:
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class TokenCounter:
    """Token counter for accurate cost calculation using tiktoken."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize token counter for specific model.

        Args:
            model_name: Model name for token encoding (defaults to settings.openai_model)
        """
        # Use settings as default, fallback to "gpt-4o" if settings is empty
        self.model_name = model_name or settings.openai_model_comprehensive or "gpt-4o-mini"

        # Initialize tiktoken encoding
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to cl100k_base (used by most modern models)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            telemetry_logger.warning("Unknown model for token counting, using cl100k_base: model=%s", self.model_name)

        # Latest OpenAI pricing per 1M tokens (2024 rates)
        self.pricing = {
            # GPT-4o models
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},

            # GPT-4.1 models (hypothetical pricing - adjust as needed)
            "gpt-4.1-nano": {"input": 0.100, "output": 0.400},  # Estimated pricing

            # GPT-4 models
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4-turbo-2024-04-09": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-32k": {"input": 60.00, "output": 120.00},

            # GPT-3.5 models
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
            "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},

            # Embedding models
            "text-embedding-3-small": {"input": 0.020, "output": 0},  # $0.020 per 1M tokens
            "text-embedding-3-large": {"input": 0.130, "output": 0},  # $0.130 per 1M tokens
            "text-embedding-ada-002": {"input": 0.100, "output": 0},  # $0.100 per 1M tokens
        }

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using model-specific encoding.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            telemetry_logger.error("Token counting failed: error=%s, text_length=%s", str(e), len(text))
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return max(1, len(text) // 4)

    def calculate_cost(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> float:
        """
        Calculate cost based on token usage and model pricing.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (uses self.model_name if not provided)

        Returns:
            Cost in USD
        """
        model_name = model or self.model_name

        if model_name not in self.pricing:
            telemetry_logger.warning("Unknown model for pricing, using gpt-4o rates: model=%s", model_name)
            model_name = "gpt-4o"

        pricing = self.pricing[model_name]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        total_cost = input_cost + output_cost
        return round(total_cost, 8)  # Round to 8 decimal places for precision

    def count_documents_tokens(self, documents: List[Document]) -> int:
        """
        Count total tokens in a list of Document objects.

        Args:
            documents: List of Document objects

        Returns:
            Total token count
        """
        total_tokens = 0
        for doc in documents:
            total_tokens += self.count_tokens(doc.page_content)
        return total_tokens

    def count_string_list_tokens(self, text_list: List[str]) -> int:
        """
        Count total tokens in a list of strings.

        Args:
            text_list: List of text strings

        Returns:
            Total token count
        """
        total_tokens = 0
        for text in text_list:
            total_tokens += self.count_tokens(text)
        return total_tokens

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "encoding_name": getattr(self.encoding, 'name', 'unknown'),
            "pricing_per_million_tokens": self.pricing.get(self.model_name, {"input": 0, "output": 0}),
            "supports_counting": hasattr(self.encoding, 'encode')
        }


class TelemetryLogger:
    """Logger for tracking latency, costs, and cache performance."""

    def __init__(self):
        """Initialize telemetry telemetry_logger."""
        self.logs: List[Dict[str, Any]] = []
        # Use TokenCounter with settings-based model selection
        self.token_counter = TokenCounter()  # Will use settings.openai_model by default

    def log(
        self,
        user_id: str,
        method: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cache_status: str,
        response_source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a telemetry event.

        Args:
            user_id: User identifier
            method: Method or operation name
            latency_ms: Latency in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_status: Cache status (hit_raw, hit_personalized, miss)
            response_source: Source of response (model name or 'cache')
            metadata: Additional metadata
        """
        total_tokens = input_tokens + output_tokens
        cost = self.calculate_cost(response_source, input_tokens, output_tokens)

        # Calculate actual savings based on cache status
        if cache_status == "hit":
            # For cache hits, savings = cost of fresh LLM call (gpt-4o) - cache cost (0)
            baseline_cost = self.calculate_cost("gpt-4o", input_tokens, output_tokens)
            savings_usd = baseline_cost - cost  # Should be positive for cache hits
        elif cache_status == "miss":
            # For cache misses, no savings since we had to call LLM anyway
            savings_usd = 0.0
        else:
            # For other operations (na, etc.), no meaningful savings
            savings_usd = 0.0

        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "method": method,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_status": cache_status,
            "response_source": response_source,
            "cost_usd": cost,
            "savings_usd": savings_usd,
            "metadata": metadata or {}
        }

        self.logs.append(log_entry)

        # Only log meaningful events with non-zero cost or savings
        if cost > 0 or savings_usd > 0:
            telemetry_logger.info(
                "Telemetry: user_id=%s, method=%s, cache_status=%s, cost_usd=%.4f, savings_usd=%.4f, latency_ms=%d",
                user_id, method, cache_status, cost, savings_usd, latency_ms
            )

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate API call cost based on token usage using TokenCounter.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        if model == "cache":
            return 0.0

        # Use TokenCounter for accurate cost calculation
        return self.token_counter.calculate_cost(input_tokens, output_tokens, model)

    def get_logs(self, user_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get telemetry logs with optional filtering.

        Args:
            user_id: Optional user filter
            limit: Optional limit on number of logs returned

        Returns:
            Filtered telemetry logs
        """
        logs = self.logs

        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]

        if limit:
            logs = logs[-limit:]

        return logs

    def get_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for telemetry data.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Summary statistics
        """
        if not self.logs:
            return {"error": "No telemetry data available"}

        # Filter by time window
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_logs = [log for log in self.logs if log["timestamp"] >= cutoff_time]

        if not recent_logs:
            return {"error": f"No telemetry data in the last {time_window_hours} hours"}

        # Calculate aggregates
        total_requests = len(recent_logs)
        total_cost = sum(log["cost_usd"] for log in recent_logs)
        total_baseline_cost = sum(log["baseline_cost_usd"] for log in recent_logs)
        total_savings = total_baseline_cost - total_cost
        avg_latency = sum(log["latency_ms"] for log in recent_logs) / total_requests

        # Cache statistics
        cache_hits = len([log for log in recent_logs if log["cache_status"] in ["hit_raw", "hit_personalized"]])
        cache_hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0

        # User statistics
        unique_users = len(set(log["user_id"] for log in recent_logs))

        # Cache breakdown
        raw_hits = len([log for log in recent_logs if log["cache_status"] == "hit_raw"])
        personalized_hits = len([log for log in recent_logs if log["cache_status"] == "hit_personalized"])
        misses = len([log for log in recent_logs if log["cache_status"] == "miss"])

        return {
            "time_window_hours": time_window_hours,
            "total_requests": total_requests,
            "unique_users": unique_users,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 4),
            "total_baseline_cost_usd": round(total_baseline_cost, 4),
            "total_savings_usd": round(total_savings, 4),
            "savings_percent": round((total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0, 2),
            "cache_breakdown": {
                "raw_hits": raw_hits,
                "personalized_hits": personalized_hits,
                "misses": misses
            }
        }

    def get_user_summary(self, user_id: str, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for specific user.

        Args:
            user_id: User identifier
            time_window_hours: Time window in hours

        Returns:
            User-specific summary statistics
        """
        user_logs = self.get_logs(user_id=user_id)

        if not user_logs:
            return {"error": f"No telemetry data for user {user_id}"}

        # Filter by time window
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_logs = [log for log in user_logs if log["timestamp"] >= cutoff_time]

        if not recent_logs:
            return {"error": f"No telemetry data for user {user_id} in the last {time_window_hours} hours"}

        total_requests = len(recent_logs)
        total_cost = sum(log["cost_usd"] for log in recent_logs)
        total_savings = sum(log["savings_usd"] for log in recent_logs)
        avg_latency = sum(log["latency_ms"] for log in recent_logs) / total_requests

        # Cache statistics for user
        cache_hits = len([log for log in recent_logs if log["cache_status"] in ["hit_raw", "hit_personalized"]])
        cache_hit_rate = (cache_hits / total_requests) * 100 if total_requests > 0 else 0

        return {
            "user_id": user_id,
            "time_window_hours": time_window_hours,
            "total_requests": total_requests,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "total_cost_usd": round(total_cost, 4),
            "total_savings_usd": round(total_savings, 4)
        }

    def clear_logs(self, older_than_hours: int = 168) -> int:
        """
        Clear old telemetry logs.

        Args:
            older_than_hours: Remove logs older than this many hours

        Returns:
            Number of logs cleared
        """
        cutoff_time = time.time() - (older_than_hours * 3600)
        initial_count = len(self.logs)
        self.logs = [log for log in self.logs if log["timestamp"] >= cutoff_time]
        cleared_count = initial_count - len(self.logs)

        telemetry_logger.info("Telemetry logs cleared: cleared_count=%s, remaining=%s", cleared_count, len(self.logs))
        return cleared_count

    def export_logs(self, filename: str, user_id: Optional[str] = None) -> None:
        """
        Export telemetry logs to file.

        Args:
            filename: Output filename
            user_id: Optional user filter
        """
        logs = self.get_logs(user_id=user_id)

        try:
            with open(filename, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
            telemetry_logger.info("Telemetry logs exported: filename=%s, count=%s", filename, len(logs))
        except Exception as e:
            telemetry_logger.error("Failed to export telemetry logs: filename=%s, error=%s", filename, str(e))
            raise

    def get_cost_analysis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get detailed cost analysis.

        Args:
            time_window_hours: Time window for analysis

        Returns:
            Cost analysis breakdown
        """
        summary = self.get_summary(time_window_hours)

        if "error" in summary:
            return summary

        # Additional cost analysis
        recent_logs = [log for log in self.logs if log["timestamp"] >= (time.time() - (time_window_hours * 3600))]

        # Cost by model
        cost_by_model = {}
        for log in recent_logs:
            model = log["response_source"]
            if model not in cost_by_model:
                cost_by_model[model] = {"cost": 0, "requests": 0}
            cost_by_model[model]["cost"] += log["cost_usd"]
            cost_by_model[model]["requests"] += 1

        # Cost by cache status
        cost_by_cache_status = {}
        for log in recent_logs:
            status = log["cache_status"]
            if status not in cost_by_cache_status:
                cost_by_cache_status[status] = {"cost": 0, "requests": 0}
            cost_by_cache_status[status]["cost"] += log["cost_usd"]
            cost_by_cache_status[status]["requests"] += 1

        summary["cost_by_model"] = cost_by_model
        summary["cost_by_cache_status"] = cost_by_cache_status

        return summary


class TelemetryService:
    """Service for collecting and managing telemetry data."""

    def __init__(self):
        """Initialize telemetry service with telemetry_logger and token counter."""
        self.telemetry_logger = TelemetryLogger()
        self.token_counter = TokenCounter()
        self.events: List[TelemetryEvent] = []
        self.metrics_cache: Dict[str, TelemetryMetrics] = {}

    async def track_event(self, event: TelemetryEvent) -> None:
        """
        Track a telemetry event.

        Args:
            event: Telemetry event to track
        """
        self.events.append(event)

        # Update metrics cache
        cache_key = f"{event.event_type}_{event.user_id or 'anonymous'}"
        if cache_key not in self.metrics_cache:
            self.metrics_cache[cache_key] = TelemetryMetrics(
                total_requests=0,
                cache_hit_rate=0.0,
                average_latency_ms=0.0,
                total_cost_usd=0.0,
                total_tokens=0,
                unique_users=0,
                error_rate=0.0,
                time_window_hours=24
            )

        # Update metrics
        metrics = self.metrics_cache[cache_key]
        metrics.total_requests += 1

        if hasattr(event, 'duration_ms') and event.duration_ms:
            # Update average latency
            total_latency = metrics.average_latency_ms * (metrics.total_requests - 1) + event.duration_ms
            metrics.average_latency_ms = total_latency / metrics.total_requests

        telemetry_logger.debug("Telemetry event tracked: event_type=%s, cache_key=%s", event.event_type, cache_key)

    async def track_request(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a request/performance event.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            user_id: Optional user identifier
            metadata: Additional metadata
        """
        event = TelemetryEvent(
            event_type=f"request_{operation}",
            user_id=user_id,
            duration_ms=duration_ms,
            data={"success": success, "operation": operation},
            metadata=metadata or {}
        )

        await self.track_event(event)

    async def track_cost(
        self,
        cost_tracker: CostTracker
    ) -> None:
        """
        Track API costs.

        Args:
            cost_tracker: Cost tracking information
        """
        event = TelemetryEvent(
            event_type="cost_tracking",
            user_id=None,  # Cost tracking is typically system-wide
            data={
                "model": cost_tracker.model,
                "prompt_tokens": cost_tracker.prompt_tokens,
                "completion_tokens": cost_tracker.completion_tokens,
                "total_tokens": cost_tracker.total_tokens,
                "cost_usd": cost_tracker.cost_usd
            }
        )

        await self.track_event(event)

    async def get_metrics(
        self,
        time_window_hours: int = 24,
        user_id: Optional[str] = None
    ) -> TelemetryMetrics:
        """
        Get aggregated metrics for time window.

        Args:
            time_window_hours: Time window in hours
            user_id: Optional user filter

        Returns:
            Aggregated telemetry metrics
        """
        if user_id:
            # Get user-specific metrics
            summary = self.telemetry_logger.get_user_summary(user_id, time_window_hours)
        else:
            # Get global metrics
            summary = self.telemetry_logger.get_summary(time_window_hours)

        if "error" in summary:
            return TelemetryMetrics(
                total_requests=0,
                cache_hit_rate=0.0,
                average_latency_ms=0.0,
                total_cost_usd=0.0,
                total_tokens=0,
                unique_users=0,
                error_rate=0.0,
                time_window_hours=time_window_hours
            )

        # For user-specific metrics, unique_users would be 1 (just that user)
        unique_users = 1 if user_id else summary.get("unique_users", 0)

        return TelemetryMetrics(
            total_requests=summary["total_requests"],
            cache_hit_rate=summary["cache_hit_rate_percent"],
            average_latency_ms=summary["average_latency_ms"],
            total_cost_usd=summary["total_cost_usd"],
            total_tokens=0,  # Would need to calculate from logs
            unique_users=unique_users,
            error_rate=0.0,  # Would need to track errors
            time_window_hours=time_window_hours
        )

    async def get_cache_hit_rate(self, time_window_hours: int = 24) -> float:
        """
        Calculate cache hit rate for time window.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Cache hit rate percentage
        """
        summary = self.telemetry_logger.get_summary(time_window_hours)
        return summary.get("cache_hit_rate_percent", 0.0)

    async def get_average_latency(
        self,
        operation: Optional[str] = None,
        time_window_hours: int = 24
    ) -> float:
        """
        Calculate average latency for operations.

        Args:
            operation: Optional operation filter (e.g., "chat_completion", "rag_query")
            time_window_hours: Time window in hours

        Returns:
            Average latency in milliseconds
        """
        if operation:
            # For operation-specific latency, we need to filter logs
            # This would require implementing operation-specific filtering in TelemetryLogger
            # For now, return overall average
            telemetry_logger.warning("Operation-specific latency filtering not yet implemented, using overall average: operation=%s", operation)

        summary = self.telemetry_logger.get_summary(time_window_hours)
        return summary.get("average_latency_ms", 0.0)

    async def cleanup_old_events(self, hours_to_keep: int = 168) -> int:
        """
        Clean up old telemetry events.

        Args:
            hours_to_keep: Number of hours to keep events

        Returns:
            Number of events cleaned up
        """
        # Clear old logs from telemetry telemetry_logger
        cleared_count = self.telemetry_logger.clear_logs(hours_to_keep)

        # Clear old events
        cutoff_time = time.time() - (hours_to_keep * 3600)
        initial_count = len(self.events)
        self.events = [event for event in self.events if event.timestamp >= cutoff_time]
        events_cleared = initial_count - len(self.events)

        telemetry_logger.info("Telemetry cleanup completed: logs_cleared=%s, events_cleared=%s", cleared_count, events_cleared)

        return cleared_count + events_cleared

    async def export_metrics(self, format: str = "json") -> Dict[str, Any]:
        """
        Export metrics in specified format.

        Args:
            format: Export format (json, csv, etc.)

        Returns:
            Exported metrics data
        """
        if format.lower() == "json":
            return self.telemetry_logger.get_cost_analysis()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_telemetry_logger(self) -> TelemetryLogger:
        """Get the telemetry telemetry_logger instance."""
        return self.telemetry_logger

    def get_token_counter(self) -> TokenCounter:
        """Get the token counter instance."""
        return self.token_counter

    async def track_document_processing(
        self,
        course_id: str,
        files_processed: int,
        chunks_created: int,
        tokens_processed: int,
        processing_time_ms: float,
        success_rate: float = 1.0
    ) -> None:
        """
        Track document processing metrics.

        Args:
            course_id: Course ID for the processed documents
            files_processed: Number of files processed
            chunks_created: Number of chunks created
            tokens_processed: Number of tokens processed
            processing_time_ms: Processing time in milliseconds
            success_rate: Success rate (0.0-1.0)
        """
        await self.track_request(
            operation="document_processing",
            duration_ms=processing_time_ms,
            success=success_rate > 0.5,
            user_id=None,
            metadata={
                "course_id": course_id,
                "files_processed": files_processed,
                "chunks_created": chunks_created,
                "tokens_processed": tokens_processed,
                "success_rate": success_rate
            }
        )

        # Log detailed processing metrics
        telemetry_logger.info(
            "Document processing tracked: course_id=%s, files=%d, chunks=%d, tokens=%d, time_ms=%.2f, success_rate=%.2f",
            course_id, files_processed, chunks_created, tokens_processed, processing_time_ms, success_rate
        )


class BatchTokenCounter:
    """Specialized token counter for batch processing."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.token_counter = TokenCounter(model_name)

    def analyze_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze token usage for a batch of documents.

        Args:
            documents: List of Document objects

        Returns:
            Comprehensive batch analysis
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_tokens": 0,
                "avg_tokens_per_doc": 0,
                "max_tokens_per_doc": 0,
                "min_tokens_per_doc": 0,
                "cost_estimate": {}
            }

        token_counts = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = self.token_counter.count_tokens(doc.page_content)
            token_counts.append(doc_tokens)
            total_tokens += doc_tokens

        avg_tokens = total_tokens / len(documents)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)

        # Use the token counter's calculate_cost method
        cost_estimate = {
            "input_cost_usd": self.token_counter.calculate_cost(total_tokens, 0, self.token_counter.model_name),
            "output_cost_usd": 0,
            "total_cost_usd": self.token_counter.calculate_cost(total_tokens, 0, self.token_counter.model_name),
            "model": self.token_counter.model_name,
            "input_tokens": total_tokens
        }

        return {
            "total_documents": len(documents),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "max_tokens_per_doc": max_tokens,
            "min_tokens_per_doc": min_tokens,
            "token_distribution": {
                "under_1k": sum(1 for t in token_counts if t < 1000),
                "1k_to_5k": sum(1 for t in token_counts if 1000 <= t < 5000),
                "5k_to_10k": sum(1 for t in token_counts if 5000 <= t < 10000),
                "over_10k": sum(1 for t in token_counts if t >= 10000)
            },
            "cost_estimate": cost_estimate
        }

    def suggest_batch_sizes(self, documents: List[Document], target_batch_size_tokens: int = 100000) -> List[List[Document]]:
        """
        Suggest optimal batch groupings for documents.

        Args:
            documents: List of Document objects
            target_batch_size_tokens: Target tokens per batch

        Returns:
            List of document batches
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in documents:
            doc_tokens = self.token_counter.count_tokens(doc.page_content)

            if current_tokens + doc_tokens > target_batch_size_tokens and current_batch:
                # Start new batch
                batches.append(current_batch)
                current_batch = [doc]
                current_tokens = doc_tokens
            else:
                current_batch.append(doc)
                current_tokens += doc_tokens

        # Add last batch
        if current_batch:
            batches.append(current_batch)

        return batches


# Global telemetry service instance
telemetry_tracker = TelemetryService()