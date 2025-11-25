from typing import Dict, Any, List, Optional, AsyncGenerator
from openai import AsyncOpenAI, OpenAI
import time

from ..config.settings import settings
from .exceptions import LLMException
from .telemetry import TokenCounter
from ..models.chat import ChatRequest, ChatResponse
from .logger import model_logger


class LLMClient:
    """Client for interacting with OpenAI LLM with flexible model strategy."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """
        Initialize LLM client with OpenAI connection.

        Args:
            token_counter: Optional token counter for usage tracking
        """
        self.async_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.token_counter = token_counter or TokenCounter()

        # Flexible model configuration for different use cases
        self.models = {
            "comprehensive": settings.openai_model_comprehensive or "gpt-4o-mini",  # Default comprehensive model
            "personalization": settings.openai_model_personalized or "gpt-4o-mini",  # For response personalization
        }

        # Support for custom model overrides via settings
        self.models.update({
            "custom_comprehensive": getattr(settings, 'custom_comprehensive_model', None),
            "custom_personalization": getattr(settings, 'custom_personalization_model', None)
        })

        # Filter out None values
        self.models = {k: v for k, v in self.models.items() if v is not None}

        model_logger.info("LLMClient initialized with flexible models: %s", self.models)

    # Async versions for better performance in async contexts
    async def acall_llm(self, prompt: str, model: Optional[str] = None, max_tokens: int = 200, use_case: Optional[str] = None) -> Dict[str, Any]:
        """
        Async version of call_llm for better performance in async contexts.

        Args:
            prompt: Input prompt for the model
            model: Specific model to use (overrides use_case)
            max_tokens: Maximum tokens in response
            use_case: Use case type ("comprehensive", "personalization", "efficient", "fallback", "lightweight")

        Returns:
            Dictionary with response and metadata
        """
        # Model selection priority: explicit model > use_case > default comprehensive
        if model:
            selected_model = model
        elif use_case and use_case in self.models:
            selected_model = self.models[use_case]
        else:
            selected_model = self.models["comprehensive"]
        start_time = time.time()

        try:
            response = await self.async_client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )

            latency = (time.time() - start_time) * 1000  # Convert to milliseconds

            output = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # Use TokenCounter for accurate cost calculation
            cost = self.token_counter.calculate_cost(input_tokens, output_tokens, selected_model)

            model_logger.info(
                "Async LLM call completed: model=%s, use_case=%s, latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_usd=%s",
                selected_model, use_case, round(latency, 2), input_tokens, output_tokens, cost
            )

            return {
                "response": output,
                "latency_ms": round(latency, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": selected_model,
                "use_case": use_case,
                "cost_usd": cost
            }

        except Exception as e:
            model_logger.error(
                "Async LLM call failed: model=%s, use_case=%s, error=%s",
                selected_model, use_case, str(e)
            )
            raise LLMException(f"Async LLM call failed: {str(e)}")

    async def acall_comprehensive(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """Async call comprehensive model for complex reasoning tasks."""
        return await self.acall_llm(prompt, use_case="comprehensive", max_tokens=max_tokens)

    async def acall_personalization(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """Async call efficient model for response personalization."""
        return await self.acall_llm(prompt, use_case="personalization", max_tokens=max_tokens)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get current model configuration for monitoring and debugging.

        Returns:
            Dictionary with model configuration info
        """
        return {
            "available_models": self.models,
            "default_models": {
                "comprehensive": self.models["comprehensive"],
                "personalization": self.models["personalization"]
            },
            "token_counter_model": self.token_counter.model_name,
            "client_initialized": bool(self.client),
            "async_client_initialized": bool(self.async_client)
        }

    async def health_check(self) -> bool:
        """
        Check if LLM service is healthy using cost-effective model.

        Returns:
            True if service is healthy
        """
        try:
            # Use the most cost-effective model for health check
            health_check_model = self.models.get("comprehensive", self.models.get("personalization"))

            response = await self.async_client.chat.completions.create(
                model=health_check_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            model_logger.error("LLM health check failed: %s", str(e))
            return False