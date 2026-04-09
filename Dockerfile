FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .
RUN mkdir -p data models exports logs
RUN useradd -m -s /bin/bash trader
USER trader
ENTRYPOINT ["python", "scripts/paper_trade.py"]
CMD ["--config", "configs/deployment/production.yaml", "--synthetic"]
