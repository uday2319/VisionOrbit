"""FastAPI main application entry point for SatQuery AI backend."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import analysis_router, upload_router
from .config import get_settings
from .core.errors import AppError, ErrorCode
from .core.logging import configure_logging, get_logger
from .db import init_db

settings = get_settings()
configure_logging(level=settings.log_level, as_json=settings.log_json)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events."""
    logger.info("Initializing SatQuery AI backend service...")
    settings.ensure_dirs()
    init_db()
    logger.info("Database initialized successfully.")
    yield
    logger.info("Shutting down SatQuery AI backend service.")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing (SIH26167)",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) if settings.cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request ID tracking
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Exception Handlers
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.warning(f"AppError: {exc.message}", extra={"code": exc.code, "request_id": request_id, "context": exc.context})
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_envelope(request_id=request_id),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.error("Unhandled server exception", extra={"request_id": request_id, "error": str(exc)}, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_code": ErrorCode.INTERNAL_ERROR,
            "message": "An internal server error occurred while processing the request.",
            "request_id": request_id,
            "recoverable": False,
        },
    )


# Mount Static Files
storage_dir = settings.storage_root
storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

# Include Routers
app.include_router(upload_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")


@app.get("/")
async def root_redirect():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
