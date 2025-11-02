from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import os

# Import settings with error handling
try:
    from config import settings
except Exception as e:
    print(f"Warning: Could not load config: {e}")
    # Create minimal settings for basic operation
    class MinimalSettings:
        app_name = "Noki AI Engine"
        app_version = "1.0.0"
        debug = False
        host = "0.0.0.0"
        port = int(os.getenv("PORT", 8000))
        log_level = "INFO"
        allowed_origins = ["*"]
        rate_limit_per_user = 0
    settings = MinimalSettings()

from app.routes import chat, embed
from app.auth import verify_bearer_token

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    description="Noki AI Engine - Intelligent Academic Assistant",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security middleware (more permissive for Railway)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allow all hosts for Railway deployment
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Railway deployment
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(embed.router, prefix="/embed", tags=["Embeddings"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with full request details"""
    try:
        body = await request.body()
        logger.error(f"=== VALIDATION ERROR ===")
        logger.error(f"Request URL: {request.url}")
        logger.error(f"Request method: {request.method}")
        logger.error(f"Raw request body: {body.decode()}")
        logger.error(f"Validation errors: {exc.errors()}")
    except Exception as e:
        logger.error(f"Error logging validation failure: {e}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.get("/")
def root(token: str = Depends(verify_bearer_token)):
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def simple_health():
    """Simple health check endpoint for Railway"""
    try:
        return {"status": "healthy", "service": "noki-ai-engine", "timestamp": "2024-01-01T00:00:00Z"}
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "healthy", "service": "noki-ai-engine"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
