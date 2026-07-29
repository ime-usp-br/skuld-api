# app/api/routers/v2.py
"""Rotas de inferência e retreino da Skuld API v2 (próxima geração de modelo).

Mesmo contrato Pydantic do v1 (AGENTS.md §4 -- contrato com o Laravel). Só
divergem os artefatos: ``app.ml.v2.predictor``/``trainer`` e os ``.pkl`` em
``MODELS_DIR_V2`` (env própria, ver docker-compose.yml).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.api.schemas import BaseResponse, PredicaoResponse
from app.ml.v2.predictor import gerar_predicoes
from app.ml.v2.trainer import pipeline_treinamento

CACHE_DIR = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))

router = APIRouter(tags=["v2"])


@router.post("/train", status_code=202, response_model=BaseResponse)
def treinar(background_tasks: BackgroundTasks) -> BaseResponse:
    """Dispara em background o retreino do v2 e responde 202 Accepted."""
    background_tasks.add_task(pipeline_treinamento)
    return BaseResponse(status="accepted", message="Retreino v2 iniciado em background.")


@router.get("/predict", response_model=PredicaoResponse)
def prever(
    ano_sem: int = Query(..., description="Semestre alvo no formato YYYYX (ex.: 20262)."),
) -> PredicaoResponse:
    """Inferência síncrona e ultra-rápida do v2."""
    if not os.path.exists(CACHE_DIR):
        raise HTTPException(
            status_code=428,
            detail=(
                "Cache da 'Máquina do Tempo' inexistente. Inicialize-o via "
                "POST /api/v2/cache/initialize antes de inferir."
            ),
        )
    try:
        predicoes = gerar_predicoes(ano_sem)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- rota de produção: envelope 500
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na inferência v2 para ano_sem={ano_sem}: {exc}",
        ) from exc
    return PredicaoResponse(
        semestre_alvo=ano_sem,
        total_turmas=len(predicoes),
        predicoes=predicoes,
    )