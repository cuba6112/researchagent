# Docker Integration

Best practices for using uv in Docker containers for fast, reproducible builds.

## Table of Contents
- [Official Images](#official-images)
- [Installing uv](#installing-uv)
- [Basic Dockerfile](#basic-dockerfile)
- [Multi-Stage Builds](#multi-stage-builds)
- [Optimizations](#optimizations)
- [Development Workflow](#development-workflow)
- [CI/CD Integration](#cicd-integration)

## Official Images

### Distroless (Smallest)

```dockerfile
FROM ghcr.io/astral-sh/uv:latest
# Contains only uv binary, no shell
```

### Python Images

```dockerfile
# Debian-based with Python
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Alpine-based with Python
FROM ghcr.io/astral-sh/uv:python3.12-alpine
```

### Version Pinning

```dockerfile
# Pin to specific version (recommended)
FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim

# Pin to SHA (most reproducible)
FROM ghcr.io/astral-sh/uv@sha256:2381d6aa60c326b71fd40023f921a0a3b8f91b14d5db6b90402e65a635053709
```

## Installing uv

### Copy from Official Image

```dockerfile
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/
```

### Install via Script

```dockerfile
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"
```

## Basic Dockerfile

### Simple Project

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Install project
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "myapp"]
```

### With .dockerignore

```dockerignore
.venv/
__pycache__/
*.pyc
.git/
.env
```

## Multi-Stage Builds

### Production Build (Recommended)

```dockerfile
# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Use copy mode (required for multi-stage)
ENV UV_LINK_MODE=copy

# Install dependencies first (cached)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime stage (no uv needed)
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "myapp"]
```

### With Managed Python

```dockerfile
# Build stage with uv-managed Python
FROM ghcr.io/astral-sh/uv:bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python (cached before project)
RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Minimal runtime (no system Python)
FROM debian:bookworm-slim

WORKDIR /app
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "myapp"]
```

## Optimizations

### Bytecode Compilation

Improves startup time at cost of build time:

```dockerfile
ENV UV_COMPILE_BYTECODE=1
```

Or per-command:

```dockerfile
RUN uv sync --compile-bytecode
```

### Cache Mounts

Speed up rebuilds with BuildKit cache:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
```

### Link Mode

```dockerfile
# Copy mode required for cache mounts and multi-stage
ENV UV_LINK_MODE=copy
```

### Disable Cache (Smaller Images)

When not using cache mounts:

```dockerfile
RUN uv sync --frozen --no-cache
```

Or:

```dockerfile
ENV UV_NO_CACHE=1
```

### Non-Root User

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN groupadd --system --gid 999 app \
    && useradd --system --gid 999 --uid 999 --create-home app

WORKDIR /app
COPY --chown=app:app . .

USER app
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "myapp"]
```

## Development Workflow

### Bind Mount Development

```bash
#!/bin/bash
docker run -it --rm \
    -v $(pwd):/app \
    -v /app/.venv \
    -p 8000:8000 \
    myapp:dev \
    uv run uvicorn main:app --reload --host 0.0.0.0
```

### Docker Compose with Watch

```yaml
# compose.yml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    develop:
      watch:
        - action: sync
          path: .
          target: /app
          ignore:
            - .venv/
        - action: rebuild
          path: pyproject.toml
        - action: rebuild
          path: uv.lock
```

```bash
docker compose watch
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4
  with:
    version: "0.9.18"

- name: Install dependencies
  run: uv sync --frozen

- name: Run tests
  run: uv run pytest
```

### GitLab CI

```yaml
image: ghcr.io/astral-sh/uv:python3.12-bookworm-slim

variables:
  UV_CACHE_DIR: .uv-cache

cache:
  paths:
    - .uv-cache/

test:
  script:
    - uv sync --frozen
    - uv run pytest
```

### AWS Lambda

```dockerfile
FROM ghcr.io/astral-sh/uv:0.9.18 AS uv

FROM public.ecr.aws/lambda/python:3.13 AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_INSTALLER_METADATA=1
ENV UV_LINK_MODE=copy

RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv export --frozen --no-dev -o requirements.txt && \
    uv pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

FROM public.ecr.aws/lambda/python:3.13
COPY --from=builder ${LAMBDA_TASK_ROOT} ${LAMBDA_TASK_ROOT}
COPY ./app ${LAMBDA_TASK_ROOT}/app
CMD ["app.main.handler"]
```

## Environment Variables Reference

| Variable | Purpose |
|----------|---------|
| `UV_COMPILE_BYTECODE=1` | Compile .pyc files |
| `UV_LINK_MODE=copy` | Copy files instead of hard links |
| `UV_NO_CACHE=1` | Disable caching |
| `UV_SYSTEM_PYTHON=1` | Use system Python |
| `UV_PYTHON_INSTALL_DIR` | Where to install managed Python |
| `UV_PYTHON_PREFERENCE=only-managed` | Only use uv-managed Python |
| `UV_NO_DEV=1` | Skip dev dependencies |
| `UV_TOOL_BIN_DIR` | Where to install tool binaries |
