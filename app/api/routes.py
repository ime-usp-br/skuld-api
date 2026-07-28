# app/api/routes.py
"""Rotas HTTP da Skuld API (FastAPI).

Layout (AGENTS.md §3):
- ``POST /api/v1/cache/initialize``: extração histórica total (2010 → corrente)
  em background -- "Máquina do Tempo".
- ``POST /api/v1/train``: retreino SOTA em background (refresh incremental +
  fit direto); sobrescreve os ``.pkl`` no disco.
- ``GET /api/v1/predict``: inferência síncrona ultra-rápida (matemática pura).

Agnóstico à unidade USP: todo escopo (codundclg, prefixos, ano_min/ano_max)
vem do ``.env`` via ``DatasetConfig.from_env`` -- nada hardcoded de IME/MAC.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from replicado.dataset_alocacao import DatasetConfig, montar_dataset

from app.api.schemas import BaseResponse, PredicaoResponse
from app.ml.predictor import gerar_predicoes
from app.ml.trainer import pipeline_treinamento

# Diretórios centralizados em ``tmp/`` (gitignored), alinhados ao ``predictor``,
# ``trainer`` e aos volumes Docker (``skuld_cache_data`` / ``skuld_models``).
# Override em produção via ``SKULD_DIR_SAIDA`` / ``SKULD_DIR_CACHE``.
DIR_SAIDA = Path(os.getenv("SKULD_DIR_SAIDA", "tmp/dataset_alocacao.csv"))
CACHE_DIR = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))

app = FastAPI(
    title="Skuld API",
    description="Microserviço preditivo para alocação de salas da USP",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Skuld API está viva e respirando!"}


def extrair_historico_completo() -> None:
    """Extração histórica total (2010 → ano corrente) que semeia o cache da
    "Máquina do Tempo".

    ``forcar_extracao=True`` com ``atualizar_anos`` vazio re-extraí do banco
    TURMAGR, todas as tabelas auxiliares e a **totalidade** da HISTESCOLARGR
    via ``replicado-python`` (o fix de honrar ``forcar`` no histórico). O
    escopo (unidade, prefixos, ``ano_min``/``ano_max``) é lido do ``.env`` em
    ``DatasetConfig.from_env`` -- o skuld permanece agnóstico à unidade. O CSV
    mestre é serializado em ``DIR_SAIDA`` e os pickles em ``CACHE_DIR`` (ambos
    gitignored / volumes Docker).
    """
    montar_dataset(
        DatasetConfig.from_env(saida=DIR_SAIDA, cache_dir=CACHE_DIR),
        forcar_extracao=True,
    )


@app.post("/api/v1/cache/initialize", status_code=202, response_model=BaseResponse)
def inicializar_cache(background_tasks: BackgroundTasks) -> BaseResponse:
    """Dispara em background a extração histórica completa e responde 202
    Accepted imediatamente (carga pesada -- não bloqueia o cliente)."""
    background_tasks.add_task(extrair_historico_completo)
    return BaseResponse(
        status="accepted",
        message="Extração histórica iniciada em background (2010 → corrente).",
    )


@app.post("/api/v1/train", status_code=202, response_model=BaseResponse)
def treinar(background_tasks: BackgroundTasks) -> BaseResponse:
    """Dispara em background o retreino SOTA (refresh incremental dos 2 últimos
    anos + fit direto) e responde 202 Accepted. Ao final sobrescreve os
    ``.pkl`` no ``MODELS_DIR/{codundclg}/`` -- a próxima ``/predict`` os relê."""
    background_tasks.add_task(pipeline_treinamento)
    return BaseResponse(
        status="accepted",
        message="Retreino iniciado em background.",
    )


@app.get("/api/v1/predict", response_model=PredicaoResponse)
def prever(
    ano_sem: int = Query(..., description="Semestre alvo no formato YYYYX (ex.: 20262)."),
) -> PredicaoResponse:
    """Inferência síncrona e ultra-rápida (matemática pura, sem reextrair).

    Requer que o cache da "Máquina do Tempo" já tenha sido semeado. Caso
    ausente, responde ``428 Precondition Required`` indicando a necessidade de
    ``POST /api/v1/cache/initialize``. Erros inesperados viram ``500``.
    """
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
    except Exception as exc:  # noqa: BLE001 -- rota de produção: envelope 500 agnóstico
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno na inferência para ano_sem={ano_sem}: {exc}",
        ) from exc
    return PredicaoResponse(
        semestre_alvo=ano_sem,
        total_turmas=len(predicoes),
        predicoes=predicoes,
    )