"""API routes package."""
from .analysis import router as analysis_router
from .upload import router as upload_router

__all__ = ["upload_router", "analysis_router"]
