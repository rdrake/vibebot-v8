# Stage 1: Builder
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy

# Install dependencies first (incomplete workspace - use --frozen)
COPY pyproject.toml uv.lock ./
COPY plugins/llm/pyproject.toml plugins/llm/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev

# Install complete project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Stage 2: Runtime
FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Volumes for persistent data
VOLUME ["/app/conf", "/app/data", "/app/logs"]

ENTRYPOINT ["limnoria", "bot.conf"]
