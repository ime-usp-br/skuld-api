"""Pipeline de retreino RÁPIDO ARGUS ML (fit direto, sem Optuna).

Regras de Ouro (ver AGENTS.md):
- Antes de treinar, refaz SELETIVAMENTE os 2 últimos anos letivos de
  HISTESCOLARGR via ``montar_dataset(forcar_extracao=True,
  atualizar_anos=[ano-1, ano])`` — a lib agora honra ``forcar`` no histórico
  e oferece refresh cirúrgico por ano (replicado 5c26c29 em diante). Os anos
  antigos (≤ ano−3) são imutáveis e vêm do cache existente; só os 2 anos
  quentes são re-extraídos, garantindo frescor sem re-pecar todo o histórico.
- Hiperparâmetros NÃO são calibrados aqui: são lidos do JSON de config da
  unidade (``app/ml/configs/{codundclg}.json``), já fixados em valores SOTA
  encontrados por Optuna offline. O retreino reduz-se a ``lgb.fit()`` sobre
  todo o histórico disponível → segundos, não horas.
- Modelos serializados em ``MODELS_DIR / {codundclg} /`` via ``joblib.dump``,
  sobrescrevendo os ``.pkl`` anteriores. O ``/predict`` relê-os a cada
  chamada (Regra de Ouro: nenhum modelo carregado em escopo global).
- Agnóstico à unidade: ``codundclg`` vem do ``.env``; nada hardcoded.

Tradução da Célula 2 do notebook (separação de features por população) +
Célula 4-final (fit com hiperparams congelados), sem a Célula 4 (Optuna).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from replicado.dataset_alocacao import DatasetConfig, montar_dataset

# Diretórios alinhados ao AGENTS.md / docker-compose. ``saida`` e ``cache_dir``
# são os mesmos lidos pelo ``predictor.py`` e pelo volume ``skuld_cache_data``.
# Modelos vão para ``MODELS_DIR`` (volume ``skuld_models``), subpasta por
# unidade; fallback é o diretório deste módulo + ``models``.
_DIRETORIO_SAIDA = Path(os.getenv("SKULD_DIR_SAIDA", "tmp/dataset_alocacao.csv"))
_DIRETORIO_CACHE = Path(os.getenv("SKULD_DIR_CACHE", "tmp/cache_maquina_tempo"))
_DIRETORIO_MODELOS = Path(os.getenv("MODELS_DIR", Path(__file__).parent / "models"))

# Diretório dos arquivos JSON de config por unidade (regras + hiperparams).
_DIRETORIO_CONFIGS = Path(__file__).parent / "configs"

NOME_MODELO_CALOUROS = "modelo_calouros.pkl"
NOME_MODELO_VETERANOS = "modelo_veteranos.pkl"

TARGET_COL = "delta"
# Identifiers descartados do leque de features (Célula 2 do notebook).
IDENTIFIERS = ["verdis", "codtur", "ano", "ano_sem", "sufixo", "coddis"]

# Cast categórico (espelha Célula 1 do notebook e predictor._COLS_CAT). Sem
# ele o LightGBM levanta "categorical_feature do not match" no fit/inferência.
_COLUNAS_CATEGORICAS = [
    "coddis",
    "sufixo",
    "departamento",
    "tiptur",
    "statur",
    "sem_tipo",
]


def _carregar_hyperparams(codundclg: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lê ``app/ml/configs/{codundclg}.json`` e devolve os dois dicts de
    hiperparâmetros (calouros, veteranos) com os sufixos fixos adicionados.

    Adiciona ``random_state=42``, ``n_jobs=-1`` e ``verbosity=-1`` sobre o
    que vier do JSON (garante reprodutibilidade e silência o LightGBM, mesmo
    que o JSON não traga essas chaves). Mantém a Skuld agnóstica à unidade.
    """
    caminho = _DIRETORIO_CONFIGS / f"{codundclg}.json"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de config não encontrado: {caminho}. Crie "
            f"app/ml/configs/{codundclg}.json com as chaves "
            f"'hyperparametros_calouros' e 'hyperparametros_veteranos'."
        )
    with caminho.open(encoding="utf-8") as f:
        cfg = json.load(f)
    sufixo = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
    params_cal = {**cfg["hyperparametros_calouros"], **sufixo}
    params_vet = {**cfg["hyperparametros_veteranos"], **sufixo}
    return params_cal, params_vet


def _preparar_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Aplica o cast categórico e separa os leques de features por população.

    Tradução literal da Célula 2 do notebook:
    - Cast ``object -> category`` das colunas categóricas operacionais.
    - Calouros: features operacionais puras + todos os sensores macro_*
      (sensibilidade a semestres atípicos pós-crise).
    - Veteranos: leque completo de features complexas (exclui apenas
      identificadores, o alvo e o sinal cru de pico ``nummtr_max`` que
      vaza o alvo).
    """
    for col in _COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    features_macro = [c for c in df.columns if c.startswith("macro_")]

    features_calouros_base = [
        c
        for c in [
            "coddis",
            "sufixo",
            "departamento",
            "tiptur",
            "estmtr",
            "vagas_reais",
            "flag_vagas_baixas",
            "creaul",
            "cretrb",
            "carga_total_creditos",
            "qtd_professores",
            "qtd_turmas_abertas",
            "flag_noturno",
            "flag_sexta",
            "flag_turma_ingressantes",
        ]
        if c in df.columns
    ]

    # Operacional + macro, sem duplicatas (preserva a ordem).
    features_calouros = list(dict.fromkeys(features_calouros_base + features_macro))

    # Veteranos: todas exceto identificadores, alvo e nummtr_max (vazamento).
    features_veteranos = [
        c
        for c in df.columns
        if c not in IDENTIFIERS and c != TARGET_COL and c != "nummtr_max"
    ]

    return df, features_calouros, features_veteranos


def _salvar_modelo(modelo: object, codundclg: int, nome: str) -> Path:
    """Serializa o modelo em ``MODELS_DIR / {codundclg} / nome``,
    criando a pasta da unidade se não existir. Sobrescreve o ``.pkl`` anterior.
    """
    base = _DIRETORIO_MODELOS / str(codundclg)
    base.mkdir(parents=True, exist_ok=True)
    caminho = base / nome
    joblib.dump(modelo, caminho)
    return caminho


def pipeline_treinamento() -> dict[str, Any]:
    """Retreino RÁPIDO ARGUS ML: refresh incremental + fit direto.

    Fluxo (AGENTS.md §3):
      1. ``REPLICADO_CODUNDCLG`` define a unidade.
      2. Hiperparâmetros SOTA são lidos do JSON da unidade.
      3. ``montar_dataset(forcar_extracao=True, atualizar_anos=[ano-1,
         ano])``: refaz TURMAGR + auxiliares e re-extrai SÓ os 2 anos
         quentes de HISTESCOLARGR; demais anos vêm do cache (imutáveis na
         prática). Treina-se sobre todo o histórico 2010 → corrente.
      4. Separação por população (Modelo Dual — segregação de risco).
      5. ``lgb.fit()`` direto sobre 100% dos dados, sem Optuna (segundos).
      6. Sobrescreve os ``.pkl`` em ``MODELS_DIR / {codundclg} /``.

    Retorna um dict de status (caminhos, contagens de amostras, parâmetros)
    para logging na rota ``/train``.
    """
    codundclg = int(os.getenv("REPLICADO_CODUNDCLG", "0"))
    if codundclg == 0:
        raise ValueError(
            "REPLICADO_CODUNDCLG não definido no .env — o treino requer a "
            "unidade para localizar hiperparâmetros (config JSON) e gravar "
            "modelos no subdiretório correto."
        )

    params_cal, params_vet = _carregar_hyperparams(codundclg)

    # 1. Refresh incremental: só os 2 últimos anos quentes de HISTESCOLARGR
    #    são re-extraídos do banco; todo o histórico (2010 → ano) é montado
    #    a partir do cache atualizado. ``forcar_extracao=True`` refaz
    #    TURMAGR + auxiliares (rápido) e, com ``atualizar_anos``, SÓ as
    #    fatias listadas do histórico.
    ano = datetime.now().year
    df_full = montar_dataset(
        DatasetConfig.from_env(saida=_DIRETORIO_SAIDA, cache_dir=_DIRETORIO_CACHE),
        forcar_extracao=True,
        atualizar_anos=[ano - 1, ano],
    )
    if df_full.empty:
        raise ValueError("Dataset de treino vazio após montar_dataset.")

    # 2. Separação de features + cast categórico (Célula 2 do notebook).
    df_full, feat_cal, feat_vet = _preparar_features(df_full)

    # 3. Separação por população (Modelo Dual — segregação de risco).
    df_cal = df_full[df_full["flag_turma_ingressantes"] == 1].reset_index(drop=True)
    df_vet = df_full[df_full["flag_turma_ingressantes"] == 0].reset_index(drop=True)

    if df_cal.empty:
        raise ValueError(
            "Nenhuma turma de calouros (flag_turma_ingressantes == 1) no "
            "dataset; verifique o cache e o REPLICADO_PREFIXOS_DISC."
        )
    if df_vet.empty:
        raise ValueError(
            "Nenhuma turma de veteranos (flag_turma_ingressantes == 0) no "
            "dataset; verifique o cache e o REPLICADO_PREFIXOS_DISC."
        )

    # 4. Fit direto sobre 100% dos dados com hiperparâmetros SOTA do JSON.
    modelo_calouros = LGBMRegressor(**params_cal)
    modelo_calouros.fit(df_cal[feat_cal], df_cal[TARGET_COL])

    modelo_veteranos = LGBMRegressor(**params_vet)
    modelo_veteranos.fit(df_vet[feat_vet], df_vet[TARGET_COL])

    # 5. Serialização sobrescrevendo os .pkl anteriores (subpasta da unidade).
    caminho_cal = _salvar_modelo(modelo_calouros, codundclg, NOME_MODELO_CALOUROS)
    caminho_vet = _salvar_modelo(modelo_veteranos, codundclg, NOME_MODELO_VETERANOS)

    return {
        "versao_arquitetura": "SOTA (fit direto, hiperparams do JSON)",
        "codundclg": codundclg,
        "anos_atualizados": [ano - 1, ano],
        "n_turmas_total": len(df_full),
        "n_turmas_calouros": len(df_cal),
        "n_turmas_veteranos": len(df_vet),
        "n_features_calouros": len(feat_cal),
        "n_features_veteranos": len(feat_vet),
        "hiperparametros_calouros": params_cal,
        "hiperparametros_veteranos": params_vet,
        "caminho_modelo_calouros": str(caminho_cal),
        "caminho_modelo_veteranos": str(caminho_vet),
        "dir_modelos": str(_DIRETORIO_MODELOS / str(codundclg)),
    }