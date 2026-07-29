"""Placeholder do pipeline de inferência v2.

Versão em desenvolvimento. Mantém o mesmo contrato do v1
(``gerar_predicoes(ano_sem) -> list[PredicaoTurma]``) para que o roteador
``app/api/routers/v2.py`` funcione sem alterações. Substituir pela
implementação real à medida que a nova geração do modelo for desenvolvida.

Regras de Ouro (AGENTS.md §4) já respeitadas aqui:
- Nada de modelo carregado globalmente.
- ``capacidade_sugerida`` sempre ``int`` (via ``np.ceil`` -> ``int``).
"""

from __future__ import annotations

from app.api.schemas import PredicaoTurma


def gerar_predicoes(ano_sem: int) -> list[PredicaoTurma]:
    """Stub: devolve lista vazia para validar o fluxo da rota ``/api/v2/predict``.

    FIXME: implementar a inferência real do v2 quando o ``trainer`` e os
    ``.pkl`` estiverem prontos. Lembrar de:
      - reler os modelos de ``MODELS_DIR_V2 / {codundclg}`` via ``joblib.load``
        a cada chamada;
      - reaplicar o cast categórico (§6.3) e o coerce defensivo object -> numeric
        (§6.7);
      - ler as regras de negócio de ``app/ml/v2/configs/{codundclg}.json``.
    """
    _ = ano_sem  # placeholder -- ainda não há modelo
    return []