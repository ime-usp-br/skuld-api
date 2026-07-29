# app/api/routers/cache.py
"""Rota de inicialização do cache da "Máquina do Tempo".

Compartilhada entre v1 e v2: a extração histórica (replicado-python) não
diverge entre versões -- só os modelos/regras divergem. Ambas as versões
montam o mesmo ``APIRouter`` e o agregador ``routes.py`` monta-o sob cada
prefixo de versão.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from replicado.dataset_alocacao import DatasetConfig, montar_dataset

from app.api.schemas import BaseResponse

DIR_SAIDA = Path(os.getenv("SKULD_DIR_SAIDA", "tmp/dataset_alocacao.csv"))
CACHE_DIR = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))

router = APIRouter(tags=["cache"])


def extrair_historico_completo() -> None:
    """Extração histórica total (2010 -> ano corrente) que semeia o cache da
    "Máquina do Tempo". ``forcar_extracao=True`` re-extraí tudo via
    ``replicado-python``. Escopo lido do ``.env`` via ``DatasetConfig.from_env``.
    """
    montar_dataset(
        DatasetConfig.from_env(saida=DIR_SAIDA, cache_dir=CACHE_DIR),
        forcar_extracao=True,
    )


@router.post("/cache/initialize", status_code=202, response_model=BaseResponse)
def inicializar_cache(background_tasks: BackgroundTasks) -> BaseResponse:
    """Dispara em background a extração histórica completa e responde 202
    Accepted imediatamente (carga pesada -- não bloqueia o cliente)."""
    background_tasks.add_task(extrair_historico_completo)
    return BaseResponse(
        status="accepted",
        message="Extração histórica iniciada em background (2010 -> corrente).",
    )