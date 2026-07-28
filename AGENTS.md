# 🤖 Contexto para Agentes de IA (Projeto: Skuld API)

---
trigger: always_on
---

Se você é um agente de Inteligência Artificial interagindo com este repositório, **LEIA ESTE DOCUMENTO INTEIRO ANTES DE ESCREVER CÓDIGO**. Este é o "cérebro compartilhado" da arquitetura e contém diretrizes que não podem ser violadas.

## 🎯 1. Domínio e Papel no Ecossistema
A **Skuld API** é um microserviço preditivo (Machine Learning) desenvolvido para a Universidade de São Paulo (USP). Seu objetivo é prever a **demanda real (pico de ocupação) de turmas de graduação**.
- **O Passado:** A extração de dados brutos é delegada à biblioteca `replicado-python`, que lê do banco de dados da USP.
- **O Futuro (Skuld):** Lê o cache, aplica modelos `LightGBM Quantile` e regras de negócio para prever o *delta* de alunos (quantos alunos faltam em relação à estimativa institucional).
- **O Presente (Consumidor):** A API serve a um monólito (Laravel) que consome essas predições para enviar comandos matemáticos ao `alocacao-solver` (motor OR-Tools).

## 🛠️ 2. Stack e Padrões de Código
- **Linguagem:** Python 3.14. Utilize Type Hints modernos de forma estrita (ex: `list[str]` em vez de `List[str]`, `str | None` em vez de `Optional[str]`).
- **Web Framework:** FastAPI + Uvicorn + Pydantic v2.
- **Machine Learning:** LightGBM, Pandas, Scikit-Learn, Optuna, Joblib.
- **Gerenciamento:** Poetry (Dependências) e Docker (Volumes persitentes para os Caches e Modelos).
- **Qualidade:** Ruff (Linting e Auto-formatting). Só aplique o Ruff nos arquivos modificados na sua task para não gerar diffs gigantes.
- **⚠️ Ambiente de Execução (OBRIGATÓRIO):** O projeto roda **exclusivamente em Docker** (ver `docker-compose.yml` / `Dockerfile`). NÃO instale dependências (`lightgbm`, `pandas`, `replicado-python`, etc.) nem crie `venv` no host. Para qualquer teste ou execução de código (lint, typecheck, treino, inferência), entre no container em pé (`docker exec -it skuld-api-dev bash`) ou suba-o (`docker compose up -d`). Instalar no host quebra a paridade de versões e gera worktrees/envs poluídos. O Ruff também roda dentro do container; no host seu `ruff` local provavelmente é de versão diferente do pin (`^0.3.0`) e pode disparar regras inexistentes (ex: `DTZ005`/`BLE001`).

## 🚦 3. Fluxo de Dados e Endpoints (Regras de Arquitetura)
A separação de responsabilidades neste projeto é muito estrita:

1. **`POST /api/v1/cache/initialize` (Carga Pesada):** Deve rodar em `BackgroundTasks`. Extrai os dados desde 2010 até o presente usando o `replicado-python`. Salva no volume Docker `/app/tmp/cache_maquina_tempo/` (mountpoint alinhado ao `predictor.py`; ver seção 6).
2. **`GET /api/v1/predict` (Inferência Rápida):** Rota ultra-rápida puramente matemática. Junta o cache em memória (Pandas), roda as predições e devolve a `capacidade_sugerida`.
3. **`POST /api/v1/train` (Retreino):** Deve rodar em `BackgroundTasks`. Atualiza apenas os últimos 2 anos de dados antes da inferência, para garantir que o modelo veja mutações recentes. Faz a busca de hiperparâmetros (Optuna) e treina os modelos SOTA. Ao final, **sobrescreve os arquivos `.pkl` no disco**.

## ⚠️ 4. Regras de Ouro (Anti-Padrões Proibidos)

- 🚫 **NUNCA carregue modelos globalmente na RAM:** Na rota `/predict` (ou na função injetada por ela), o modelo `.pkl` do LightGBM **deve ser lido do disco a cada requisição** (via `joblib.load()`). Isso garante que, após um retreino na rota `/train`, a próxima predição utilize os pesos recém-salvos sem precisar reiniciar o container ou gerenciar estado em memória.
- 🚫 **NUNCA faça Hardcode de Unidades:** Não restrinja o código para o Instituto de Matemática (ex: `45`, `IME`, `MAC`, `MAT`). A Skuld é agnóstica. Qualquer escopo da USP deve ser lido via variáveis de ambiente configuradas no servidor (`REPLICADO_CODUNDCLG`, `REPLICADO_PREFIXOS_DISC`).
- 🚫 **CUIDADO com Data Leakage (Vazamento Temporal):** Ao trabalhar com séries temporais no Pandas, garanta que os cálculos de histórico (Features) NÃO usem o semestre alvo. A regra é: aplique sempre `.shift(1)` **antes** de `.rolling()`.
- 🚫 **Respeite o Contrato Pydantic:** O Laravel espera uma estrutura JSON exata. Certifique-se de que a `capacidade_sugerida` nunca seja `float`. Aplique `np.ceil()` e converta para `int`.

## 📁 5. A Pasta `./tmp` (Documentação Preditiva e R&D)
Você (Agente de IA) notará que o usuário pode colar conteúdos de arquivos como `argus_model_exploration.md`, `argus_model_inference_20262.md`, etc., no chat.
- Estes arquivos são a prova matemática e de pesquisa (R&D) que gerou as regras de negócio deste projeto (ex: *O Corte Híbrido* e a *Blindagem de Calouros*).
- **Importante:** A pasta `./tmp` onde esses documentos originais residem fica APENAS no ambiente local do desenvolvedor e está no `.gitignore`. Ela NUNCA sobe para o GitHub. 
- Se você for encarregado de alterar uma regra de Machine Learning ou Feature Engineering e tiver dúvidas, **peça explicitamente ao usuário** para fornecer o conteúdo do respectivo arquivo `argus_*.md` da pasta `./tmp` como contexto.

## 🪤 6. Armadilhas Operacionais (Lições Aprendidas em Integração)
Estes são erros custosos que não são óbvios ao ler o código. Leia antes de tocar em inferência/Docker:

### 🧩 6.1 Skema de Features vs. Versão do `replicado-python` (CRÍTICO)
- O pin do `replicado-python` no `pyproject.toml` **determina quantas features** o `montar_dataset` produz. A versão *lean* (`fd92fbe`) gera ~69 colunas; a versão rica (`3da52ef` em diante) gera ~103 (lags `*_t1/t2`, `media_rolling_nummtr_max_3sem`, `vagas_curso_*`, `semestres_consecutivos_prof`, `flag_fora_de_epoca`, `macro_*`).
- **Os modelos `.pkl` são treinados sobre um skema específico.** Se o pin não bater, a inferência quebra com `KeyError: ['delta_t1', 'macro_*', ...] not in index`. Antes de culpar o `predictor.py`, confirme que o pin do `pyproject.toml` corresponde ao skema esperado por `modelo.feature_name_` (`joblib.load(...).feature_name_`).
- **NUNCA hardcodeie a lista de features.** Recupere-a do próprio modelo via `feature_name_` (sklearn API) — isso tolera mutações no esquema após um retreino. Em fallback de `n_features_in_` só se `feature_name_` for nulo.

### 🔌 6.2 LightGBM Nativo depende de `libgomp1`
- O Dockerfile **DEVE** instalar `libgomp1`. Sem ele, `import lightgbm` crasha em runtime com `OSError: libgomp.so.1: cannot open shared object file` — mensagem críptica que parece bug de código mas é só dependência nativa (OpenMP) ausente na imagem `python:*-slim`.
- Se precisar testar num container já em pé sem rebuild, instale runtime: `apt-get update && apt-get install -y libgomp1` (não persiste no image; para persistir, edite o Dockerfile).

### 🏷️ 6.3 Cast Categórico da Célula 1 é Obrigatório em Inferência
- O notebook (Célula 1) converte `coddis, sufixo, departamento, tiptur, statur, sem_tipo` para `category` **antes do treino**. O `montar_dataset` devolve essas colunas como `object`.
- Em inferência, sem reapply do cast, o LightGBM levanta `ValueError: train and valid dataset categorical_feature do not match`. Reaplique o cast exatamente das colunas categóricas presentes nas features do modelo (ver `_aplicar_categoricas` em `app/ml/predictor.py`).
- Não é preciso reutilizar as categorias salvas no `Booster.pandas_categorical`; basta `astype("category")` das colunas corretas (o LightGBM valida apenas a *quantidade* de colunas `category`, valores inéditos viram "unknown" tratados nativamente).

### 📂 6.4 Configura `saida` E `cache_dir` — nunca deixe default `temp/`
- `montar_dataset(DatasetConfig)` serializa o CSV mestre em `cfg.saida` e lê pickles da "Máquina do Tempo" de `cfg.cache_dir`. **Ambos são configuráveis.**
- O default do `replicado` é `temp/...`, que **não é gitignored por padrão** e cria um diretório `temp/` no host via bind mount — polui e pode vazar dados sensíveis.
- A Skuld centraliza ambos em `tmp/` (gitignored): `saida = tmp/dataset_alocacao.csv`, `cache_dir = tmp/cache_maquina_tempo`. Override via env `SKULD_DIR_SAIDA` / `SKULD_DIR_CACHE` em produção.

### 🗃️ 6.5 Mountpoint dos Volumes Docker (compose) Deve Alinhar ao Código
- Com bind mount `.:/app`, montar um volume nomeado em `/app/<dir>` **força o Docker a criar `<dir>` no host** (mesmo vazio). Se o mountpoint divergir do diretório que o código lê, surgem diretórios zumbis no host (ex: `temp/`) que não são gitignored por padrão e confundem o trabalho.
- O `docker-compose.yml` monta `skuld_cache_data` em `/app/tmp/cache_maquina_tempo` (alinhado ao `predictor.py`) e `skuld_models` em `/app/app/ml/models`. **Se mover os diretórios no código, mova os mountpoints junto.**

### 🧪 6.6 Testando Inferência Offline no Container
- Para validar `/predict` sem banco de dados (túnel SSH down), copie os caches (.pkl) e modelos (.pkl) para os volumes nomeados:
  - Caches: `tar cf - . | docker exec -i skuld-api-dev tar xf - -C /app/tmp/cache_maquina_tempo/`
  - Modelos: `docker cp app/ml/models/<nome>.pkl skuld-api-dev:/app/app/ml/models/`
- Rode: `docker exec skuld-api-dev python -c "from app.ml.predictor import gerar_predicoes; print(len(gerar_predicoes(20262)))"`.
- Sanity check: para `2026.2`, o notebook reporta **129 turmas, média 59.2 assentos/turma** — se sua inferência não bater, há divergência de skema/dtype.

### 🛡️ 6.7 Coerce Defensivo `object -> numeric`
- Features numéricas que cheguem ao `.predict()` com dtype `object` (em vez de `int`/`float`) fazem o LightGBM levantar `pandas dtypes must be int, float or bool`. Isso já ocorreu com `macro_gap_calendario_fonte` (fix upstream em `3da52ef`), mas regressões futuras no dataset podem reincidir.
- O `predictor.py` coerce defensivamente `object -> numeric` (via `pd.to_numeric(..., errors="coerce")`) nas features não-categóricas — **mantenha essa defesa** na rota de produção `/predict`.