FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY env /app/env
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml

EXPOSE 7860

CMD ["uvicorn", "env.environment:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
