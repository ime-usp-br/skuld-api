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

## 🚦 3. Fluxo de Dados e Endpoints (Regras de Arquitetura)
A separação de responsabilidades neste projeto é muito estrita:

1. **`POST /api/v1/cache/initialize` (Carga Pesada):** Deve rodar em `BackgroundTasks`. Extrai os dados desde 2010 até o presente usando o `replicado-python`. Salva no volume Docker `/app/temp/cache_maquina_tempo/`.
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