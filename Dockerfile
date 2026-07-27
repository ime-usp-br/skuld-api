FROM python:3.14-slim-trixie

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala os drivers do Sybase/SQL Server e o Git (para baixar o replicado-python)
RUN apt-get update && apt-get install -y \
    freetds-dev \
    freetds-bin \
    tdsodbc \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

# Copia os arquivos de dependência (Se o lock ainda não existir, crie ele rodando 'poetry lock' depois)
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]