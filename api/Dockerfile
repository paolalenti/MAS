FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN hf download sentence-transformers/all-MiniLM-L6-v2 --include "*.safetensors"

COPY . .

CMD python ingest.py && python main.py