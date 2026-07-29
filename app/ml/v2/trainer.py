"""Placeholder do pipeline de treinamento v2.

Versão em desenvolvimento. Mantém o mesmo contrato do v1
(``pipeline_treinamento() -> None``) para que o roteador
``app/api/routers/v2.py`` dispare em background sem alterações.

Ao final de uma implementação real, sobrescreve os ``.pkl`` em
``MODELS_DIR_V2 / {codundclg}/`` -- a próxima ``/api/v2/predict`` os relê do
disco (Regra de Ouro, AGENTS.md §4).
"""

from __future__ import annotations


def pipeline_treinamento() -> None:
    """Stub: no-op para validar o fluxo da rota ``/api/v2/train``.

    FIXME: implementar o retreino real do v2 quando a nova geração do modelo
    e os hiperparâmetros (Optuna) estiverem definidos. Lembrar de:
      - fazer o refresh incremental dos 2 últimos anos antes do fit;
      - sobrescrever os ``.pkl`` em ``MODELS_DIR_V2 / {codundclg}/``;
      - serializar as regras de negócio em ``app/ml/v2/configs/{codundclg}.json``.
    """
    return None