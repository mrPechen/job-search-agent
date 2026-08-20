# ==========================
# Builder stage
# ==========================
FROM python:3.12-slim AS builder

WORKDIR /app

# gcc и libpq-dev нужны для сборки asyncpg; curl — для установки Playwright
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./

# Собираем только продакшн-зависимости (без black/pytest/testcontainers)
RUN uv sync --no-dev

# Установка Chromium для Playwright (нужен сервису browser): браузер и системные
# зависимости ставятся в builder-стадии, чтобы не тянуть компилятор в runtime
RUN uv run playwright install --with-deps chromium


# ==========================
# Runtime stage
# ==========================
FROM python:3.12-slim AS runtime

WORKDIR /app

# libpq5 — рантайм-библиотека для asyncpg; curl нужен для playwright install-deps
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache/ms-playwright /root/.cache/ms-playwright
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright

# Системные зависимости Chromium в runtime-образе: без них браузер не запустится
# (в builder они ставились в отдельном слое и в runtime не переносятся)
RUN .venv/bin/playwright install-deps chromium

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
