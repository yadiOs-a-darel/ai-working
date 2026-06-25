"""Application FastAPI JARVIS."""

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import chat, memory, status
from api.websocket import router as ws_router
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(f"API {settings.jarvis_name} démarrée")
    yield
    logger.info(f"API {settings.jarvis_name} arrêtée")


app = FastAPI(
    title=f"{settings.jarvis_name} API",
    version="1.0.0",
    description=f"API de l'assistant IA personnel {settings.jarvis_name}",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:7860",
        f"http://localhost:{settings.ui_port}",
        f"http://127.0.0.1:{settings.ui_port}",
        f"http://127.0.0.1:{settings.api_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    logger.debug(f"{request.method} {request.url.path} — {duration:.0f}ms")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Erreur API : {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


app.include_router(chat.router)
app.include_router(status.router)
app.include_router(memory.router)
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "name": f"{settings.jarvis_name} API",
        "version": "1.0.0",
        "docs": "/docs",
    }
