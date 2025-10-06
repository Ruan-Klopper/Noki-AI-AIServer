"""
Health and metrics routes for monitoring and observability
"""
import logging
import time
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import HealthResponse, MetricsResponse
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple in-memory metrics store (in production, use proper metrics system)
metrics_store = {
    "requests_total": 0,
    "errors_total": 0,
    "latency_sum": 0.0,
    "stage_distribution": {
        "thinking": 0,
        "intent": 0,
        "response": 0,
        "complete": 0
    },
    "intent_frequency": {
        "backend_query": 0,
        "proposed_schedule": 0,
        "proposed_tasks": 0
    },
    "token_usage": {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_embedding_tokens": 0,
        "total_cost_usd": 0.0
    }
}


def update_metrics(stage: str = None, intent_type: str = None, latency: float = 0.0, 
                   error: bool = False, token_usage: Dict[str, Any] = None):
    """Update metrics counters"""
    metrics_store["requests_total"] += 1
    metrics_store["latency_sum"] += latency
    
    if error:
        metrics_store["errors_total"] += 1
    
    if stage:
        metrics_store["stage_distribution"][stage] = metrics_store["stage_distribution"].get(stage, 0) + 1
    
    if intent_type:
        metrics_store["intent_frequency"][intent_type] = metrics_store["intent_frequency"].get(intent_type, 0) + 1
    
    if token_usage:
        metrics_store["token_usage"]["total_prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        metrics_store["token_usage"]["total_completion_tokens"] += token_usage.get("completion_tokens", 0)
        metrics_store["token_usage"]["total_embedding_tokens"] += token_usage.get("embedding_tokens", 0)
        metrics_store["token_usage"]["total_cost_usd"] += token_usage.get("cost_estimate_usd", 0.0)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for liveness probes
    
    Returns basic health status and version information.
    """
    try:
        # Basic health checks
        health_status = "healthy"
        
        # Check if required services are configured
        if not settings.openai_api_key:
            health_status = "degraded"
        
        if not settings.pinecone_api_key:
            health_status = "degraded"
        
        return HealthResponse(
            status=health_status,
            timestamp=datetime.utcnow(),
            version=settings.app_version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version
        )


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with service status
    
    Returns comprehensive health information including
    external service connectivity.
    """
    try:
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.app_version,
            "services": {
                "openai": "unknown",
                "pinecone": "unknown",
                "supabase": "unknown"
            },
            "configuration": {
                "debug_mode": settings.debug,
                "log_level": settings.log_level,
                "rate_limiting_enabled": settings.rate_limit_per_user > 0
            }
        }
        
        # Check OpenAI connectivity (simplified)
        if settings.openai_api_key:
            health_info["services"]["openai"] = "configured"
        else:
            health_info["services"]["openai"] = "not_configured"
            health_info["status"] = "degraded"
        
        # Check Pinecone connectivity (simplified)
        if settings.pinecone_api_key:
            health_info["services"]["pinecone"] = "configured"
        else:
            health_info["services"]["pinecone"] = "not_configured"
            health_info["status"] = "degraded"
        
        # Check Supabase connectivity (simplified)
        if settings.supabase_url and settings.supabase_key:
            health_info["services"]["supabase"] = "configured"
        else:
            health_info["services"]["supabase"] = "not_configured"
        
        return health_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """
    Metrics endpoint for monitoring dashboards
    
    Returns basic performance metrics including
    request counts, error rates, and latency.
    """
    try:
        # Calculate average latency
        avg_latency = 0.0
        if metrics_store["requests_total"] > 0:
            avg_latency = metrics_store["latency_sum"] / metrics_store["requests_total"]
        
        return MetricsResponse(
            requests_total=metrics_store["requests_total"],
            errors_total=metrics_store["errors_total"],
            avg_latency_ms=avg_latency * 1000,  # Convert to milliseconds
            stage_distribution=metrics_store["stage_distribution"],
            intent_frequency=metrics_store["intent_frequency"],
            token_usage=metrics_store["token_usage"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return MetricsResponse(
            requests_total=0,
            errors_total=1,
            avg_latency_ms=0.0,
            stage_distribution={},
            intent_frequency={},
            token_usage={},
            timestamp=datetime.utcnow()
        )


@router.get("/metrics/prometheus")
async def prometheus_metrics() -> str:
    """
    Prometheus-formatted metrics endpoint
    
    Returns metrics in Prometheus exposition format
    for integration with Prometheus monitoring.
    """
    try:
        metrics_lines = []
        
        # Basic metrics
        metrics_lines.append(f"# HELP noki_ai_requests_total Total number of requests")
        metrics_lines.append(f"# TYPE noki_ai_requests_total counter")
        metrics_lines.append(f"noki_ai_requests_total {metrics_store['requests_total']}")
        
        metrics_lines.append(f"# HELP noki_ai_errors_total Total number of errors")
        metrics_lines.append(f"# TYPE noki_ai_errors_total counter")
        metrics_lines.append(f"noki_ai_errors_total {metrics_store['errors_total']}")
        
        # Average latency
        avg_latency = 0.0
        if metrics_store["requests_total"] > 0:
            avg_latency = metrics_store["latency_sum"] / metrics_store["requests_total"]
        
        metrics_lines.append(f"# HELP noki_ai_avg_latency_seconds Average request latency in seconds")
        metrics_lines.append(f"# TYPE noki_ai_avg_latency_seconds gauge")
        metrics_lines.append(f"noki_ai_avg_latency_seconds {avg_latency}")
        
        # Stage distribution
        metrics_lines.append(f"# HELP noki_ai_stage_total Total requests by stage")
        metrics_lines.append(f"# TYPE noki_ai_stage_total counter")
        for stage, count in metrics_store["stage_distribution"].items():
            metrics_lines.append(f'noki_ai_stage_total{{stage="{stage}"}} {count}')
        
        # Intent frequency
        metrics_lines.append(f"# HELP noki_ai_intent_total Total requests by intent type")
        metrics_lines.append(f"# TYPE noki_ai_intent_total counter")
        for intent_type, count in metrics_store["intent_frequency"].items():
            metrics_lines.append(f'noki_ai_intent_total{{intent_type="{intent_type}"}} {count}')
        
        # Token usage metrics
        token_usage = metrics_store["token_usage"]
        metrics_lines.append(f"# HELP noki_ai_tokens_total Total tokens used")
        metrics_lines.append(f"# TYPE noki_ai_tokens_total counter")
        metrics_lines.append(f'noki_ai_tokens_total{{type="prompt"}} {token_usage.get("total_prompt_tokens", 0)}')
        metrics_lines.append(f'noki_ai_tokens_total{{type="completion"}} {token_usage.get("total_completion_tokens", 0)}')
        metrics_lines.append(f'noki_ai_tokens_total{{type="embedding"}} {token_usage.get("total_embedding_tokens", 0)}')
        
        # Cost metrics
        metrics_lines.append(f"# HELP noki_ai_cost_total Total cost in USD")
        metrics_lines.append(f"# TYPE noki_ai_cost_total counter")
        metrics_lines.append(f'noki_ai_cost_total {token_usage.get("total_cost_usd", 0.0)}')
        
        return "\n".join(metrics_lines)
        
    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        return f"# Error generating metrics: {e}"


@router.post("/metrics/reset")
async def reset_metrics() -> Dict[str, Any]:
    """
    Reset metrics counters (for testing/debugging)
    
    This endpoint should be protected in production.
    """
    try:
        global metrics_store
        metrics_store = {
            "requests_total": 0,
            "errors_total": 0,
            "latency_sum": 0.0,
            "stage_distribution": {
                "thinking": 0,
                "intent": 0,
                "response": 0,
                "complete": 0
            },
            "intent_frequency": {
                "backend_query": 0,
                "proposed_schedule": 0,
                "proposed_tasks": 0
            },
            "token_usage": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_embedding_tokens": 0,
                "total_cost_usd": 0.0
            }
        }
        
        return {
            "status": "success",
            "message": "Metrics reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        return {
            "status": "error",
            "message": f"Failed to reset metrics: {e}",
            "timestamp": datetime.utcnow().isoformat()
        }
