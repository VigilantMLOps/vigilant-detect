FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry==2.3.4
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root 2>/dev/null || \
       pip install polars fastapi uvicorn pydantic httpx numpy scipy scikit-learn \
           xgboost shap pyyaml psycopg2-binary clickhouse-connect redis apscheduler \
           typer faker joblib pytz loguru

COPY . .

RUN mkdir -p models core/logs data/raw \
    && chmod +x entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
