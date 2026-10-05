"""Service lifecycle, health and process-local capability-token approval API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.observability.logger import get_logger

logger = get_logger("api")


def create_app(settings: Settings | None = None, *, approval_service=None, presentation_service=None) -> FastAPI:
    settings = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting %s (environment=%s)", settings.APP_NAME, settings.APP_ENV)
        yield
        logger.info("Application stopped")

    application = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG, lifespan=lifespan)
    if approval_service is None:
        from backend.app.actions.service import ApprovalService
        from backend.app.actions.store import ApprovalStore
        from backend.app.actions.execution import lazy_write_sessions
        from backend.app.tools.registry import ToolRegistry
        approval_service = ApprovalService(ToolRegistry(None), lazy_write_sessions(settings.DATABASE_URL),
                                          store=ApprovalStore(ttl_minutes=settings.APPROVAL_TTL_MINUTES))
    from backend.app.api.actions import action_router
    application.state.approval_service = approval_service
    application.include_router(action_router(approval_service))
    from backend.app.api.presentation import PresentationService, presentation_router
    presentation_service = presentation_service or PresentationService(settings, approval_service)
    application.state.presentation_service = presentation_service
    application.include_router(presentation_router(presentation_service))
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "enterprise-salesops-agent-backend",
            "environment": settings.APP_ENV,
        }

    @application.get("/")
    def root() -> dict[str, str]:
        return {"message": "Enterprise SalesOps Agent API"}

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run("backend.app.main:app", host=config.BACKEND_HOST, port=config.BACKEND_PORT)
