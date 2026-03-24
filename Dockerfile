# ── Stage 1: builder ─────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-docker.txt


# ── Stage 2: runtime ─────────────────────────
FROM python:3.11-slim

WORKDIR /app

# виртуальное окружение
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# код проекта
COPY app/      ./app/
COPY model/    ./model/
COPY frontend/ ./frontend/

# пользователь (без root)
RUN useradd -m appuser
USER appuser

# запуск (ВАЖНО: порт от Render)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]