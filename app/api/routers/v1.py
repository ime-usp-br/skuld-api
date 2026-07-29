# app/api/routers/v1.py
"""Rotas de inferência e retreino da Skuld API v1 (modelo ARGUS v2.2).

Contrato mantido para o Laravel (/predict). O modelo é relido do disco a cada
chamada (Regra de Ouro, AGENTS.md §4) via ``app.ml.predictor``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.api.schemas import BaseResponse, PredicaoResponse
from app.ml.predictor import gerar_predicoes
from app.ml.trainer import pipeline_treinamento

CACHE_DIR = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))

router = APIRouter(tags=["v1"])


@router.post("/train", status_code=202, response_model=BaseResponse)
def treinar(background_tasks: BackgroundTasks) -> BaseResponse:
    """Dispara em background o retreino SOTA do v1 e responde 202 Accepted.
    Ao final sobrescreve os ``.pkl`` em ``MODELS_DIR/{codundclg}/`` -- a
    próxima ``/predict`` os relê do disco."""
    background_tasks.add_task(pipeline_treinamento)
    return BaseResponse(status="accepted", message="Retreino v1 iniciado em background.")


@router.get("/predict", response_model=PredicaoResponse)
def prever(
    ano_sem: int = Query(..., description="Semestre alvo no formato YYYYX (ex.: 20262)."),
) -> PredicaoResponse:
    """Inferência síncrona e ultra-rápida do v1 (matemática pura)."""
    if not os.path.exists(CACHE_DIR):
        raise HTTPException(
            status_code=428,
            detail=(
                "Cache da 'Máquina do Tempo' inexistente. Inicialize-o via "
                "POST /api/v1/cache/initialize antes de inferir."
            ),
        )
    try:
        predicoes = gerar_predicoes(ano_sem)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 -- rota de produção: envelope 500
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na inferência v1 para ano_sem={ano_sem}: {exc}",
        ) from exc
    return PredicaoResponse(
        semestre_alvo=ano_sem,
        total_turmas=len(predicoes),
        predicoes=predicoes,
    )