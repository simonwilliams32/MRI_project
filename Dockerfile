
#Docker file
FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY saved_models ./saved_models

CMD ["sh", "-c", "uvicorn api.app_transfer_learning:app --host 0.0.0.0 --port ${PORT:-8000}"]
