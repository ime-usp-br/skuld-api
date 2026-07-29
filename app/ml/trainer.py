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
- Leques de features também vêm do JSON (``features_calouros`` /
  ``features_veteranos``): esse arquivo é o metadado SOTA exportado pela
  Célula 8 do notebook de pesquisa. Usar os leques explícitos (em vez de
  derivá-los dinamicamente) garante fidelidade entre a pesquisa e a produção
  e impede divergência de skema. Uma guarda (``_validar_features``) falha
  cedo se alguma feature do metadado não existir no dataset montado.
- Modelos serializados em ``MODELS_DIR / {codundclg} /`` via ``joblib.dump``,
  sobrescrevendo os ``.pkl`` anteriores. O ``/predict`` relê-os a cada
  chamada (Regra de Ouro: nenhum modelo carregado em escopo global).
- Agnóstico à unidade: ``codundclg`` vem do ``.env``; nada hardcoded.

Tradução da Célula 2 do notebook (separação por população) +
Célula 4-final (fit com hiperparams congelados), sem a Célula 4 (Optuna),
consumindo os leques de features do metadado exportado pela Célula 8.
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


def _carregar_config(
    codundclg: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Lê ``app/ml/configs/{codundclg}.json`` e devolve ``(params_cal,
    params_vet, feat_cal, feat_vet)``.

    O arquivo JSON é o próprio metadado SOTA exportado pelo notebook de
    pesquisa (Célula 8), com chaves normalizadas para o runtime:
    ``regras_negocio``, ``hyperparametros_*`` e ``features_*``. A unidade é
    inferida do nome do arquivo (ex.: ``45.json`` -> unidade 45), sem depender
    de uma chave ``unidade`` interna.

    Os leques ``features_*`` são a **fonte canônica** do que o modelo viu no
    treino da pesquisa: ao usá-los explicitamente no retreino garantimos
    fidelidade entre o notebook e a produção, em vez de derivar as features
    dinamicamente (o que já provocou divergência de skema no passado).

    Adiciona ``random_state=42``, ``n_jobs=-1`` e ``verbosity=-1`` sobre o que
    vier do JSON (garante reprodutibilidade e silência o LightGBM, mesmo que
    o JSON não traga essas chaves). Mantém a Skuld agnóstica à unidade.
    """
    caminho = _DIRETORIO_CONFIGS / f"{codundclg}.json"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de config não encontrado: {caminho}. Crie "
            f"app/ml/configs/{codundclg}.json (metadado SOTA exportado pelo "
            f"notebook) com as chaves 'hyperparametros_calouros', "
            f"'hyperparametros_veteranos', 'features_calouros' e "
            f"'features_veteranos'."
        )
    with caminho.open(encoding="utf-8") as f:
        cfg = json.load(f)

    for chave in ("hyperparametros_calouros", "hyperparametros_veteranos",
                  "features_calouros", "features_veteranos"):
        if chave not in cfg:
            raise KeyError(
                f"Chave '{chave}' ausente em {caminho}. O metadado SOTA "
                f"precisa declarar hiperparâmetros e leques de features."
            )

    sufixo = {"random_state": 42, "n_jobs": -1, "verbosity": -1}
    params_cal = {**cfg["hyperparametros_calouros"], **sufixo}
    params_vet = {**cfg["hyperparametros_veteranos"], **sufixo}
    feat_cal = list(cfg["features_calouros"])
    feat_vet = list(cfg["features_veteranos"])
    return params_cal, params_vet, feat_cal, feat_vet


def _aplicar_categoricas(df: pd.DataFrame) -> None:
    """Reaplica o cast ``object -> category`` das colunas categóricas
    operacionais (espelha a Célula 1 do notebook). Sem este cast o LightGBM
    levanta "categorical_feature do not match" no ``fit``.
    """
    for col in _COLUNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _validar_features(
    df: pd.DataFrame, feat_cal: list[str], feat_vet: list[str]
) -> None:
    """Guarda de fidelidade: garante que toda feature declarada no metadado
    SOTA esteja presente no dataset montado.

    Se faltar alguma coluna, é sinal de divergência de skema entre o metadado
    (treinado na pesquisa) e o ``replicado.montar_dataset`` atual — tipicamente
    um bump de pin do ``replicado-python`` que mutou o esquema, ou variável de
    ambiente (``REPLICADO_PREFIXOS_DISC``) que alterou o leque de cursos.
    Falhar cedo aqui evita treinar um modelo sobre features parcialmente
    erradas e, depois, quebrar a inferência (``feature_name_`` não bateria).
    """
    colunas = set(df.columns)
    faltam_cal = [c for c in feat_cal if c not in colunas]
    faltam_vet = [c for c in feat_vet if c not in colunas]
    if faltam_cal or faltam_vet:
        raise ValueError(
            "Divergência de skema entre o metadado SOTA e o dataset montado. "
            f"Features de calouros ausentes: {faltam_cal}. "
            f"Features de veteranos ausentes: {faltam_vet}. "
            f"Verifique o pin do replicado-python no pyproject.toml e as envs "
            f"REPLICADO_* em relação ao ambiente usado na pesquisa."
        )


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

    # 0. Carrega hiperparâmetros + leques de features do metadado SOTA
    #    exportado pelo notebook (app/ml/configs/{codundclg}.json). Os leques
    #    ``features_*`` são a fonte canônica do que o modelo viu na pesquisa;
    #    usá-los explicitamente garante fidelidade notebook <-> produção.
    params_cal, params_vet, feat_cal, feat_vet = _carregar_config(codundclg)

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

    # 2. Cast categórico (Célula 1 do notebook) + guarda de fidelidade: garante
    #    que todo o leque do metadado SOTA existe no dataset montado. Falha
    #    cedo em vez de treinar com features divergentes da pesquisa.
    _aplicar_categoricas(df_full)
    _validar_features(df_full, feat_cal, feat_vet)

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