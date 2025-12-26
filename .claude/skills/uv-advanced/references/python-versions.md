# Python Versions

uv can install and manage multiple Python versions without pyenv, asdf, or system package managers.

## Table of Contents
- [Installing Python](#installing-python)
- [Version Selection](#version-selection)
- [Pinning Versions](#pinning-versions)
- [Python Implementations](#python-implementations)
- [Discovery Behavior](#discovery-behavior)
- [Environment Variables](#environment-variables)

## Installing Python

### Install Specific Versions

```bash
# Install latest 3.12
uv python install 3.12

# Install specific patch version
uv python install 3.12.4

# Install multiple versions
uv python install 3.10 3.11 3.12

# Install latest stable
uv python install
```

### List Available Versions

```bash
# Show all available versions
uv python list

# Show installed versions
uv python list --installed

# Show only installed
uv python list --only-installed
```

### Uninstall Python

```bash
uv python uninstall 3.10
```

### Reinstall (Update Distributions)

```bash
uv python install --reinstall 3.12
```

## Version Selection

### Command-Specific

```bash
# Use specific version for command
uv run --python 3.11 python -c "import sys; print(sys.version)"

# Create venv with specific version
uv venv --python 3.10

# Sync with specific version
uv sync --python 3.12
```

### Version Specifiers

```bash
# Exact version
uv run --python 3.12.4 script.py

# Minimum version
uv run --python ">=3.10" script.py

# Range
uv run --python ">=3.10,<3.12" script.py
```

## Pinning Versions

### Project-Level (.python-version)

```bash
# Create .python-version file
uv python pin 3.12

# Pin specific patch
uv python pin 3.12.4

# Contents of .python-version
cat .python-version
# 3.12
```

### In pyproject.toml

```toml
[project]
requires-python = ">=3.10"
```

### Priority Order

1. `--python` flag
2. `UV_PYTHON` environment variable
3. `.python-version` file (searched up to root)
4. `requires-python` in pyproject.toml
5. First compatible Python found

## Python Implementations

### CPython (Default)

```bash
uv python install 3.12
uv python install cpython-3.12
```

### PyPy

```bash
# Install PyPy
uv python install pypy3.10
uv python install pypy@3.10

# Run with PyPy
uv run --python pypy@3.10 script.py
```

### GraalPy

```bash
uv python install graalpy-24
```

### Free-Threaded CPython

```bash
# Python 3.13+ with free-threading
uv python install 3.13t
uv run --python 3.13t script.py
```

## Discovery Behavior

### Search Order

uv searches for Python in this order:

1. uv-managed installations (`~/.local/share/uv/python/`)
2. Active virtual environment (`VIRTUAL_ENV`)
3. Conda environment (`CONDA_PREFIX`)
4. System PATH

### Control Discovery

```bash
# Only use uv-managed Python
UV_PYTHON_PREFERENCE=only-managed uv run script.py

# Prefer system Python
UV_PYTHON_PREFERENCE=only-system uv run script.py

# Allow uv to install if needed
UV_PYTHON_DOWNLOADS=automatic uv run --python 3.13 script.py
```

### Virtual Environment Handling

```bash
# Use specific venv
uv run --python /path/to/.venv/bin/python script.py

# Use system Python (bypass venv)
UV_SYSTEM_PYTHON=1 uv pip install requests
```

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `UV_PYTHON` | Version string | Default Python version |
| `UV_PYTHON_PREFERENCE` | `only-managed`, `managed`, `system`, `only-system` | Where to find Python |
| `UV_PYTHON_DOWNLOADS` | `automatic`, `manual`, `never` | Auto-download behavior |
| `UV_PYTHON_INSTALL_DIR` | Path | Where to install Python |

### Preference Values

| Value | Behavior |
|-------|----------|
| `only-managed` | Only use uv-installed Python |
| `managed` | Prefer uv Python, fall back to system |
| `system` | Prefer system Python, fall back to uv (default) |
| `only-system` | Only use system Python |

### Download Values

| Value | Behavior |
|-------|----------|
| `automatic` | Download if needed (default in projects) |
| `manual` | Download only with `uv python install` |
| `never` | Never download, fail if not found |

## Storage Locations

```bash
# uv-managed Python installations
~/.local/share/uv/python/

# Example structure
~/.local/share/uv/python/
├── cpython-3.10.14-linux-x86_64-gnu/
├── cpython-3.11.9-linux-x86_64-gnu/
├── cpython-3.12.4-linux-x86_64-gnu/
└── pypy-3.10-linux-x86_64-gnu/

# Cache for Python downloads
~/.cache/uv/
```

## Cross-Platform Resolution

uv can resolve for different platforms:

```bash
# Compile requirements for different Python version
uv pip compile --python-version 3.10 requirements.in

# Compile for different platform
uv pip compile --python-platform linux --python-version 3.10 requirements.in
```

Supported platforms:
- `linux`
- `windows`
- `macos` / `darwin`

## Docker Integration

```dockerfile
FROM ghcr.io/astral-sh/uv:bookworm-slim

# Configure Python installation
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

# Install Python (separate layer for caching)
RUN uv python install 3.12

# Now use for project
WORKDIR /app
COPY . .
RUN uv sync --frozen
```

## Project Python Constraints

### Single Version

```toml
[project]
requires-python = "==3.12.*"
```

### Version Range

```toml
[project]
requires-python = ">=3.10,<3.13"
```

### Testing Multiple Versions

```bash
# Test with different versions
uv run --python 3.10 pytest
uv run --python 3.11 pytest
uv run --python 3.12 pytest
```

### CI Matrix Example

```yaml
# .github/workflows/test.yml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]

steps:
  - uses: astral-sh/setup-uv@v4
  - run: uv python install ${{ matrix.python-version }}
  - run: uv sync --python ${{ matrix.python-version }}
  - run: uv run pytest
```
