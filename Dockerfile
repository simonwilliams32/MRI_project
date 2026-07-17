
#Docker file
FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api ./api
COPY saved_models ./saved_models

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
