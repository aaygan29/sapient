"""FastAPI application factory.

Security-relevant choices here:
  * Docs (Swagger/OpenAPI) are OFF unless SAPIENT_DOCS=1 — don't advertise the surface in prod.
  * Security headers on every response (nosniff, DENY framing, HSTS, no-referrer).
  * Every /v1 route is auth-gated (default-deny) via the route dependencies.
  * The /playground page is static HTML (asks the user for their own key); it holds no secret.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..security.auth import init_keystore
from ..security.ratelimit import init_ratelimit
from ..settings import settings
from ..storage.jobs import init_store
from . import deps
from .routes import encode, health, info, jobs

_PLAYGROUND = os.path.join(os.path.dirname(__file__), "..", "..", "playground", "index.html")
_DEMO = os.path.join(os.path.dirname(__file__), "..", "..", "demo", "index.html")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_keystore(settings.keys_file)
    init_ratelimit(settings.rate_limit_per_min, settings.daily_quota)
    init_store(settings.result_ttl_seconds)
    deps.get_engine()  # build engine now (loads weights in real mode => fail fast)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Sapient-1 Encoding API",
        version=settings.model_version,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(health.router)
    app.include_router(info.router, prefix="/v1")
    app.include_router(encode.router, prefix="/v1")
    app.include_router(jobs.router, prefix="/v1")

    @app.get("/playground", response_class=HTMLResponse)
    def playground() -> str:
        try:
            with open(_PLAYGROUND, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return "<h1>Playground not found</h1>"

    @app.get("/demo", response_class=HTMLResponse)
    def demo() -> str:
        try:
            with open(_DEMO, "r", encoding="utf-8") as fh:
                return fh.read()
        except FileNotFoundError:
            return "<h1>Demo not found</h1>"

    @app.get("/")
    def root():
        return RedirectResponse(url="/demo")

    return app


app = create_app()
