# Scripts & Tools

uv supports running single-file scripts with inline dependencies (PEP 723) and executing tools from Python packages without permanent installation.

## Table of Contents
- [Running Scripts](#running-scripts)
- [Inline Script Metadata](#inline-script-metadata)
- [Script Lockfiles](#script-lockfiles)
- [Tool Execution](#tool-execution)
- [Tool Installation](#tool-installation)
- [Shebang Scripts](#shebang-scripts)

## Running Scripts

### Basic Execution

```bash
# Run script in project context
uv run script.py

# Run without project dependencies
uv run --no-project script.py

# Run in complete isolation
uv run --isolated script.py
```

### Ad-hoc Dependencies

```bash
# Run with temporary dependency
uv run --with requests script.py

# Multiple dependencies
uv run --with requests --with rich script.py

# With version constraints
uv run --with "requests>=2.28,<3" script.py
```

### Specify Python Version

```bash
uv run --python 3.11 script.py
uv run --python pypy script.py
```

## Inline Script Metadata

PEP 723 allows declaring dependencies directly in the script file.

### Basic Metadata

```python
# /// script
# dependencies = [
#   "requests>=2.28",
#   "rich",
# ]
# ///

import requests
from rich import print

response = requests.get("https://api.github.com")
print(response.json())
```

### With Python Version

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "pydantic>=2.0",
# ]
# ///

import httpx
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
```

### With Custom Index

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///

import torch
print(torch.__version__)
```

### With Exclude-Newer (Reproducibility)

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
#
# [tool.uv]
# exclude-newer = "2025-01-01T00:00:00Z"
# ///
```

### Adding Dependencies via CLI

```bash
# Add dependency to script
uv add --script script.py requests

# Add with Python version
uv add --script script.py --python ">=3.10" httpx

# Add multiple
uv add --script script.py requests rich pydantic

# Remove dependency
uv remove --script script.py requests
```

## Script Lockfiles

Lock script dependencies for reproducibility:

```bash
# Create lockfile (script.py.lock)
uv lock --script script.py

# Run uses lockfile automatically
uv run script.py

# Export dependencies
uv export --script script.py > requirements.txt

# View dependency tree
uv tree --script script.py
```

## Tool Execution

### uvx (Ephemeral Execution)

Run tools without installation:

```bash
# Run tool from PyPI
uvx ruff check .

# Equivalent to
uv tool run ruff check .

# Specific version
uvx ruff@0.1.0 check .

# From specific package
uvx --from 'ruff>=0.1' ruff check .

# With extra dependencies
uvx --with rich ruff check .
```

### Common Tools

```bash
# Code quality
uvx ruff check .
uvx ruff format .
uvx black .
uvx isort .
uvx mypy .

# Testing
uvx pytest
uvx pytest-cov

# Documentation
uvx mkdocs serve
uvx sphinx-build docs build

# Utilities
uvx httpie GET https://api.github.com
uvx pipx list
uvx cowsay "Hello uv!"
```

### Specify Python Version

```bash
uvx --python 3.11 pytest
```

## Tool Installation

### Install Globally

```bash
# Install tool
uv tool install ruff

# Specific version
uv tool install ruff@0.1.0

# With extras
uv tool install 'mkdocs[material]'

# From git
uv tool install git+https://github.com/astral-sh/ruff
```

### Manage Installed Tools

```bash
# List installed tools
uv tool list

# Upgrade tool
uv tool upgrade ruff

# Upgrade all tools
uv tool upgrade --all

# Uninstall tool
uv tool uninstall ruff

# Show tool directory
uv tool dir
uv tool dir --bin  # Binary directory
```

### Tool Environment

Each tool gets an isolated virtual environment:

```bash
# Locations
~/.local/share/uv/tools/           # Tool environments
~/.local/bin/                      # Tool binaries (symlinks)

# Override binary location
UV_TOOL_BIN_DIR=/custom/bin uv tool install ruff
```

## Shebang Scripts

Make scripts directly executable:

### Basic Shebang

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["requests"]
# ///

import requests
print(requests.get("https://httpbin.org/get").json())
```

```bash
chmod +x script.py
./script.py
```

### Quiet Shebang

Suppress uv output:

```python
#!/usr/bin/env -S uv run --quiet
# /// script
# dependencies = ["rich"]
# ///

from rich import print
print("[bold green]Hello![/]")
```

### With Python Version

```python
#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
```

## Script Best Practices

### 1. Always Specify Python Version

```python
# /// script
# requires-python = ">=3.10"  # Be explicit
# dependencies = ["requests"]
# ///
```

### 2. Pin Major Versions

```python
# /// script
# dependencies = [
#   "requests>=2.28,<3",
#   "pydantic>=2.0,<3",
# ]
# ///
```

### 3. Use Lockfiles for Critical Scripts

```bash
uv lock --script critical_script.py
# Commit both script.py and script.py.lock
```

### 4. Include Type Hints

```python
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "pydantic>=2"]
# ///

from pydantic import BaseModel
import httpx

class Response(BaseModel):
    status: int
    data: dict

def fetch(url: str) -> Response:
    r = httpx.get(url)
    return Response(status=r.status_code, data=r.json())
```

## Comparison: Script vs Project

| Feature | Script (PEP 723) | Project |
|---------|-----------------|---------|
| File structure | Single file | Directory with pyproject.toml |
| Dependencies | Inline metadata | pyproject.toml |
| Lockfile | Optional (.py.lock) | Required (uv.lock) |
| Virtual env | Ephemeral/cached | Persistent (.venv) |
| Best for | One-off scripts, utilities | Applications, libraries |
| Sharing | Copy single file | Clone repo |
