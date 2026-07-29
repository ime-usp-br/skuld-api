# app/api/routes.py
"""Agregador da aplicação FastAPI da Skuld API.

Monta as versões v1 e v2 sob seus respectivos prefixos. O roteador de cache
(extração histórica da "Máquina do Tempo") é compartilhado entre as versões,
pois só divergem os modelos/regras -- nunca a extração (AGENTS.md §3).

Layout:
- ``GET  /health``
- ``POST /api/v{1,2}/cache/initialize``  -- extração histórica (background)
- ``POST /api/v{1,2}/train``              -- retreino da versão (background)
- ``GET  /api/v{1,2}/predict``            -- inferência síncrona ultra-rápida
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routers import cache, v1, v2

app = FastAPI(
    title="Skuld API",
    description="Microserviço preditivo para alocação de salas da USP",
    version="0.2.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Skuld API está viva e respirando!"}


# Cache compartilhado sob ambos os prefixos (mesmo job de extração).
app.include_router(cache.router, prefix="/api/v1")
app.include_router(cache.router, prefix="/api/v2")

app.include_router(v1.router, prefix="/api/v1")
app.include_router(v2.router, prefix="/api/v2")