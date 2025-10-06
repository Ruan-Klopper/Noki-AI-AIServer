"""
Authentication dependencies for the Noki AI Engine
"""
import logging
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()


def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify bearer token for API access
    
    This function validates the bearer token provided in the Authorization header.
    Returns the token if valid, raises HTTPException if invalid.
    """
    if not settings.bearer_token:
        logger.error("Bearer token not configured in environment variables")
        raise HTTPException(
            status_code=500,
            detail="Authentication not configured"
        )
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    if credentials.credentials != settings.bearer_token:
        logger.warning(f"Invalid bearer token attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token"
        )
    
    logger.debug("Bearer token verified successfully")
    return credentials.credentials


def verify_backend_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify backend service token (legacy function for backward compatibility)
    
    This function validates the backend service token for internal service communication.
    """
    if not settings.backend_service_token:
        logger.warning("Backend service token not configured, allowing access")
        return "no-auth"  # Development mode
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing"
        )
    
    if credentials.credentials != settings.backend_service_token:
        logger.warning(f"Invalid backend token attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid service token"
        )
    
    logger.debug("Backend service token verified successfully")
    return credentials.credentials


def optional_bearer_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """
    Optional bearer token verification
    
    This function allows endpoints to optionally require authentication.
    Returns the token if provided and valid, None if not provided.
    """
    if not credentials:
        return None
    
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token"
        )
    
    return credentials.credentials
