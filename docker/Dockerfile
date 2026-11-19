FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements/runtime.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "hotel_mlops.app:app", "--host", "0.0.0.0", "--port", "8000"]
