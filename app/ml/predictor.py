"""Pipeline de inferência ARGUS ML v2.1 (tradução da Célula 3 do notebook).

Regras de Ouro (ver AGENTS.md):
- Nenhum modelo é carregado em escopo global. A cada chamada de
  ``gerar_predicoes`` os ``.pkl`` são relidos do disco via ``joblib.load``,
  garantindo que um retreino na rota ``/train`` seja refletido na próxima
  predição sem reiniciar o container.
- O dataset vem do ``replicado.dataset_alocacao.montar_dataset`` (agnóstico
  à unidade USP; escopo via ``DatasetConfig.from_env``).
- ``capacidade_sugerida`` é sempre ``np.ceil`` -> ``int`` (contrato Pydantic).
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from replicado.dataset_alocacao import DatasetConfig, montar_dataset

# Diretório base dos artefatos serializados pela rota /train. No container
# Docker o volume persistente é /app; localmente, espelha app/ml/models.
_DIRETORIO_MODELOS = Path(os.getenv("SKULD_DIR_MODELOS", Path(__file__).parent / "models"))

NOME_MODELO_CALOUROS = "modelo_calouros.pkl"
NOME_MODELO_VETERANOS = "modelo_veteranos.pkl"

# Parâmetros de negócio (Célula 3). Defaults do notebook, overridables por env
# para manter a Skuld agnóstica à unidade (regra de ouro do AGENTS.md).
BUFFER_CALOUROS = float(os.getenv("SKULD_BUFFER_CALOUROS", "1.23"))
CORTE_HIBRIDO = int(os.getenv("SKULD_CORTE_HIBRIDO", "80"))

# O ``replicado.montar_dataset`` aceita ``saida`` e ``cache_dir`` no
# ``DatasetConfig``, então centralizamos ambos em ``tmp/`` (gitignored): o CSV
# mestre (``saida``) e os pickles da "Máquina do Tempo" (``cache_dir``). Em
# produção, ambos vêm do bind mount ``.:/app`` → host ``tmp/`` (caches
# extraídos uma vez por ``/cache/initialize`` e servidos em DevOps); o volume
# nomeado ``skuld_cache_data`` (se montado em ``temp/...``) é ignorado por
# padrão para evitar criar ``temp/`` no host. Override via env se necessário.
_DIRETORIO_SAIDA = Path(os.getenv("SKULD_DIR_SAIDA", "tmp/dataset_alocacao.csv"))
_DIRETORIO_CACHE = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))

# Colunas que o notebook (Célula 1) converte para ``category`` antes do
# treino. O ``montar_dataset`` devolve-nas como ``object``; sem este cast o
# LightGBM levanta "categorical_feature do not match" na inferência. Mantém-se
# a ordem do notebook para alinhar com ``Booster.pandas_categorical``.
_COLS_CAT = ["coddis", "sufixo", "departamento", "tiptur", "statur", "sem_tipo"]


def _features_do_modelo(modelo) -> list[str]:
    """Resgata a lista exata de features que o modelo viu no treino.

    O ``LGBMRegressor`` (sklearn API) expõe ``feature_name_`` após o ``fit``;
    isso evita hardcode e tolera qualquer mutação no esquema de features
    feita por um retreino. Cai para ``n_features_in_`` genérico só em
    fallback (não esperado para LightGBM).
    """
    feat = getattr(modelo, "feature_name_", None)
    if feat:
        return list(feat)
    n = getattr(modelo, "n_features_in_", 0)
    return [f"f{i}" for i in range(int(n))]


def _carregar_modelos() -> tuple[object, list[str], object, list[str]]:
    """Lê os dois ``.pkl`` do disco a cada chamada (Regra de Ouro do AGENTS.md).

    Levanta FileNotFoundError com mensagem útil se a rota /train ainda não
    populou os artefatos.
    """
    pcal = _DIRETORIO_MODELOS / NOME_MODELO_CALOUROS
    pvet = _DIRETORIO_MODELOS / NOME_MODELO_VETERANOS
    for p in (pcal, pvet):
        if not p.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {p}. Rode a rota /train antes da inferência."
            )

    mod_cal = joblib.load(pcal)
    mod_vet = joblib.load(pvet)
    feat_cal = _features_do_modelo(mod_cal)
    feat_vet = _features_do_modelo(mod_vet)
    return mod_cal, feat_cal, mod_vet, feat_vet


def _aplicar_categoricas(df: pd.DataFrame, feat: list[str]) -> None:
    """Reaplica o cast ``object -> category`` das colunas categóricas que o
    modelo viu no treino (espelha a Célula 1 do notebook).

    O ``montar_dataset`` devolve ``coddis/sufixo/departamento/tiptur`` como
    ``object``. O LightGBM valida em inferência que a *quantidade* de colunas
    com dtype ``category`` bata com o treino (``Booster.pandas_categorical``),
    sob pena de ``categorical_feature do not match``. Fazemos cast exatamente
    das ``_COLS_CAT`` presentes nas features do modelo, sem hardcode de valores
    (categorias inferidas do próprio dado; valores inéditos viram "unknown",
    tratados nativamente pelo LightGBM).
    """
    feat_set = set(feat)
    for col in _COLS_CAT:
        if col in feat_set and col in df.columns:
            df[col] = df[col].astype("category")
    # Feature não-categórica porém ``object``: qualquer coluna numérica que o
    # gerador de dataset venha a emitir como ``object`` (bug histórico do
    # ``replicado.dataset_macrosensores._macro_gap_calendario``, corrigido em
    # upstream 3da52ef, ou regressões futuras) quebraria o LightGBM. Aqui
    # coercemos ``object -> numeric`` (``pd.to_numeric``) preservando NaN,
    # como defesa para a rota de produção ``/predict``.
    for col in feat:
        if col in _COLS_CAT or col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _inferir_capacidade(
    df_target: pd.DataFrame,
    mod_cal,
    feat_cal: list[str],
    mod_vet,
    feat_vet: list[str],
) -> pd.DataFrame:
    """Pipeline puro de inferência: devolve a demanda teto contínua/arredondada.

    Matemática transcrita da Célula 3:
      1. ``mask_cal = flag_turma_ingressantes == 1``.
      2. ``delta_pred`` (0 por default) é a predição do modelo quantílico,
         separada por população (calouros / veteranos).
      3. ``cap_ml_pura = estmtr + delta_pred``.
      4. Corte Híbrido: se ``estmtr >= CORTE_HIBRIDO`` impede downgrade abaixo
         do ``estmtr`` -> ``max(estmtr, cap_ml_pura)``.
      5. Escudo de Calouros: ``cap_final[mask_cal] *= BUFFER_CALOUROS``.
      6. ``capacidade_sugerida = np.ceil(cap_final)`` (cadeiras inteiras).
    """
    df_out = df_target.copy()
    _aplicar_categoricas(df_out, feat_cal)
    mask_cal = df_out["flag_turma_ingressantes"] == 1

    # 1. Predição do Delta
    df_out["delta_pred"] = 0.0
    if mask_cal.sum() > 0:
        df_out.loc[mask_cal, "delta_pred"] = mod_cal.predict(
            df_out.loc[mask_cal, feat_cal]
        )
    # O modelo de veteranos pode ter um conjunto categórico distinto (features
    # diferentes); aplicamos seu cast sobre a fatia de veteranos.
    df_vet = df_out.loc[~mask_cal].copy()
    if len(df_vet) > 0:
        _aplicar_categoricas(df_vet, feat_vet)
        df_out.loc[~mask_cal, "delta_pred"] = mod_vet.predict(df_vet[feat_vet]).astype(float)

    # 2. Capacidade pura (estimativa institucional + correção predita)
    df_out["cap_ml_pura"] = df_out["estmtr"] + df_out["delta_pred"]

    # 3. Corte Híbrido (proibição de downgrade em auditórios >= CORTE_HIBRIDO)
    df_out["cap_hibrida"] = np.where(
        df_out["estmtr"] >= CORTE_HIBRIDO,
        np.maximum(df_out["estmtr"], df_out["cap_ml_pura"]),
        df_out["cap_ml_pura"],
    )

    # 4. Escudo isolado de calouros (multiplicador estrito)
    cap_final = df_out["cap_hibrida"].copy()
    cap_final[mask_cal] = cap_final[mask_cal] * BUFFER_CALOUROS

    # 5. Arredondamento contínuo -> inteiro (contrato Pydantic: nunca float)
    df_out["capacidade_sugerida"] = np.ceil(cap_final)
    return df_out


def gerar_predicoes(ano_sem: int) -> list[dict]:
    """Monta o dataset (cache, sem reextrair), filtra o semestre alvo e
    aplica o pipeline quantílico ARGUS v2.1.

    Retorna uma lista de dicionários chaveados pelo schema ``PredicaoTurma``
    (``coddis``, ``codtur``, ``estmtr``, ``capacidade_sugerida``,
    ``is_calouros``). ``capacidade_sugerida`` é sempre ``int`` via ``np.ceil``.
    """
    df_full = montar_dataset(
        DatasetConfig.from_env(
            saida=_DIRETORIO_SAIDA, cache_dir=_DIRETORIO_CACHE
        ),
        forcar_extracao=False,
    )
    df_target = df_full[df_full["ano_sem"] == ano_sem].copy().reset_index(drop=True)

    if df_target.empty:
        raise ValueError(
            f"Nenhuma turma encontrada para ano_sem={ano_sem} no dataset."
        )

    mod_cal, feat_cal, mod_vet, feat_vet = _carregar_modelos()

    df_out = _inferir_capacidade(df_target, mod_cal, feat_cal, mod_vet, feat_vet)

    predicoes: list[dict] = []
    for row in df_out.itertuples(index=False):
        is_cal = bool(getattr(row, "flag_turma_ingressantes", 0) == 1)
        predicoes.append(
            {
                "coddis": str(row.coddis),
                "codtur": str(row.codtur),
                "estmtr": int(row.estmtr),
                "capacidade_sugerida": int(row.capacidade_sugerida),
                "is_calouros": is_cal,
            }
        )
    return predicoes