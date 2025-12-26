# Projects

## Table of Contents
- [Creating Projects](#creating-projects)
- [Managing Dependencies](#managing-dependencies)
- [Running Commands](#running-commands)
- [Locking and Syncing](#locking-and-syncing)
- [Building and Publishing](#building-and-publishing)
- [Dependency Sources](#dependency-sources)

## Creating Projects

### Initialize New Project

```bash
# Application (no build system)
uv init myapp

# Library (with build system)
uv init --lib mylib

# Package with src layout
uv init --package mypackage

# Specify Python version
uv init --python 3.11 myproject

# Initialize in current directory
uv init
```

### Project Types

**Application** (default):
```toml
[project]
name = "myapp"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
```

**Library** (`--lib`):
```toml
[project]
name = "mylib"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Managing Dependencies

### Adding Dependencies

```bash
# Add runtime dependency
uv add requests

# Add with version constraint
uv add "fastapi>=0.100"

# Add multiple
uv add requests rich httpx

# Add development dependency
uv add --dev pytest ruff mypy

# Add to optional extras
uv add --optional test pytest pytest-cov

# Add to dependency group
uv add --group lint ruff

# Add from git
uv add git+https://github.com/org/repo

# Add from git with ref
uv add git+https://github.com/org/repo@v1.0.0
uv add git+https://github.com/org/repo@main

# Add editable local package
uv add --editable ../mylib

# Add from requirements.txt
uv add -r requirements.txt
```

### Removing Dependencies

```bash
uv remove requests
uv remove --dev pytest
```

### Upgrading Dependencies

```bash
# Upgrade specific package
uv lock --upgrade-package requests

# Upgrade all dependencies
uv lock --upgrade

# Sync after upgrade
uv sync
```

## Running Commands

### uv run

```bash
# Run Python script
uv run python main.py

# Run module
uv run -m pytest

# Run with specific Python
uv run --python 3.11 python -c "import sys; print(sys.version)"

# Run without project dependencies (isolated)
uv run --isolated script.py

# Run with extra dependencies
uv run --with rich python -c "from rich import print; print('Hello')"

# Run in specific package (workspace)
uv run --package mylib pytest
```

### Environment Variables

```bash
# Skip project discovery
UV_PROJECT_ENVIRONMENT=/path/to/.venv uv run python

# Use system Python
UV_SYSTEM_PYTHON=1 uv run python
```

## Locking and Syncing

### uv lock

```bash
# Create/update lockfile
uv lock

# Check lockfile is up-to-date (CI)
uv lock --check

# Upgrade all packages
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package requests

# Lock with reproducibility timestamp
uv lock --exclude-newer "2025-01-01T00:00:00Z"

# Lock with cooldown period (security)
uv lock --exclude-newer "7 days"
```

### uv sync

```bash
# Sync environment with lockfile
uv sync

# Sync without dev dependencies
uv sync --no-dev

# Sync with specific extras
uv sync --extra test --extra docs

# Sync all extras
uv sync --all-extras

# Sync specific groups
uv sync --group test --group lint

# Fail if lockfile outdated
uv sync --frozen

# Sync specific package (workspace)
uv sync --package mylib
```

## Building and Publishing

### Build Distribution

```bash
# Build wheel and sdist
uv build

# Build only wheel
uv build --wheel

# Build only sdist
uv build --sdist

# Build specific package (workspace)
uv build --package mylib
```

### Publish to PyPI

```bash
# Publish to PyPI
uv publish

# Publish to TestPyPI
uv publish --index-url https://test.pypi.org/legacy/

# With token
uv publish --token $PYPI_TOKEN
```

### Export Lockfile

```bash
# Export to requirements.txt
uv export > requirements.txt

# Export without hashes
uv export --no-hashes > requirements.txt

# Export with extras
uv export --extra dev > requirements-dev.txt

# Export frozen (exact versions)
uv export --frozen > requirements.txt
```

## Dependency Sources

Configure alternate sources for dependencies during development in `tool.uv.sources`:

```toml
[tool.uv.sources]
# Git repository
requests = { git = "https://github.com/psf/requests" }

# Git with specific ref
requests = { git = "https://github.com/psf/requests", tag = "v2.31.0" }
requests = { git = "https://github.com/psf/requests", branch = "main" }
requests = { git = "https://github.com/psf/requests", rev = "abc123" }

# Local path
mylib = { path = "../mylib" }

# Editable local path
mylib = { path = "../mylib", editable = true }

# Workspace member
shared = { workspace = true }

# URL
package = { url = "https://example.com/package-1.0.0.tar.gz" }

# Platform-specific source
torch = [
  { index = "pytorch-cpu", marker = "sys_platform != 'darwin'" },
  { index = "pytorch", marker = "sys_platform == 'darwin'" },
]
```

### Index Configuration

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true  # Only use when explicitly referenced

[[tool.uv.index]]
name = "private"
url = "https://pypi.mycompany.com/simple"
```
