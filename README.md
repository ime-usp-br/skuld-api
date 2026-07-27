# 🔮 Skuld API (Predictive Engine)

Microserviço de Inteligência Artificial responsável por prever a demanda real (pico de ocupação) das turmas de graduação do Instituto de Matemática e Estatística da USP (IME-USP).

Na mitologia nórdica, **Skuld** é uma das Três Nornas, responsável por tecer o fio do **Futuro** ("Aquilo que deve acontecer"). No nosso ecossistema, a Skuld lê o passado histórico da universidade e prevê a lotação física das salas, evitando que alunos fiquem em pé ou que auditórios sejam desperdiçados.

---

## 🏛️ Arquitetura do Ecossistema

A **Skuld** atua como a mente preditiva de uma arquitetura baseada em microserviços:

1. **Extração (Passado):** Importa a biblioteca base `replicado-python` para ler o banco de dados legado da USP (Sybase) via SQLAlchemy.
2. **Predição (Futuro):** Aplica modelos em árvore treinados (`LightGBM Quantile`) em memória RAM para prever o "delta" de erro da estimativa de alunos matriculados.
3. **Pós-processamento:** Aplica as regras de negócio de *Blindagem de Calouros* e o *Corte Híbrido* de Auditórios.
4. **Entrega:** Retorna um payload JSON síncrono para o Monólito (Laravel), que posteriormente enviará as restrições matemáticas para o `alocacao-solver` (Google OR-Tools).

## 🚀 Tecnologias

- **Linguagem:** Python 3.14
- **API Framework:** FastAPI + Uvicorn
- **Machine Learning:** LightGBM, Pandas, Scikit-Learn
- **Integração de Dados:** Importação nativa do `replicado-python`
- **Infraestrutura:** Docker (Volumes persistentes para Feature Store) + Poetry

---

## 📂 Estrutura do Projeto

```text
skuld-api/
├── app/
│   ├── api/
│   │   ├── routes.py          # Endpoints do FastAPI (/predict, /cache/initialize)
│   │   └── schemas.py         # Contratos de entrada e saída via Pydantic
│   └── ml/
│       ├── predictor.py       # Pipeline de inferência (DataPrep + Model.predict)
│       └── models/            # Binários (.pkl) dos modelos ARGUS ML treinados
├── scripts/
│   └── train_argus.py         # Script MLOps para retreinar o modelo semestralmente via Optuna
├── seed_cache/                # (No deploy) Arquivos .pkl pré-processados da última década
├── docker-compose.yml         # Orquestração da API e mapeamento de volumes de cache
├── Dockerfile                 
└── pyproject.toml             # Gerenciamento de dependências via Poetry
