# Stage 1: Builder
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy

# Install dependencies first (incomplete workspace - use --frozen)
COPY pyproject.toml uv.lock ./
COPY plugins/llm/pyproject.toml plugins/llm/
COPY plugins/rpg/pyproject.toml plugins/rpg/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev

# Install complete project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Stage 2: Runtime
FROM python:3.14-slim

# Create non-root user for running the bot
RUN groupadd -r vibebot && useradd -r -g vibebot -d /app vibebot

WORKDIR /app
COPY --from=builder --chown=vibebot:vibebot /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Volumes for persistent data
VOLUME ["/app/conf", "/app/data", "/app/logs"]

# Default to non-root user; callers can override with --user if needed for volume permissions
USER vibebot
ENTRYPOINT ["limnoria", "bot.conf"]
