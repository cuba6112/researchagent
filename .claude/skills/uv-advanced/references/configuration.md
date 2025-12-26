# Configuration

uv configuration via pyproject.toml, uv.toml, and environment variables.

## Table of Contents
- [Configuration Files](#configuration-files)
- [pyproject.toml Settings](#pyprojecttoml-settings)
- [uv.toml Settings](#uvtoml-settings)
- [Environment Variables](#environment-variables)
- [Package Indexes](#package-indexes)
- [Authentication](#authentication)

## Configuration Files

### Priority Order (Highest to Lowest)

1. Command-line flags
2. Environment variables
3. Project-level `pyproject.toml` (`[tool.uv]`)
4. User-level `uv.toml` (`~/.config/uv/uv.toml`)

### File Locations

```
~/.config/uv/uv.toml     # User-level (Linux/macOS)
~/Library/Application Support/uv/uv.toml  # macOS alternative
%APPDATA%\uv\uv.toml     # Windows

./pyproject.toml         # Project-level [tool.uv] section
./uv.toml                # Project-level alternative
```

## pyproject.toml Settings

### Complete Example

```toml
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff"]
docs = ["mkdocs", "mkdocs-material"]

[dependency-groups]
test = ["pytest", "pytest-cov"]
lint = ["ruff", "mypy"]

[tool.uv]
# Development dependencies (alternative to dependency-groups)
dev-dependencies = ["pytest", "ruff"]

# Dependency sources for development
[tool.uv.sources]
mylib = { git = "https://github.com/org/mylib", branch = "main" }
localutil = { path = "../utils", editable = true }

# Dependency overrides (fix incorrect metadata)
override-dependencies = [
    "pydantic>=1.0,<3.0",
]

# Constraints (narrow versions)
constraint-dependencies = [
    "urllib3<2.0",
]

# Resolution settings
resolution = "highest"  # or "lowest", "lowest-direct"
prerelease = "disallow"  # or "allow", "if-necessary"
exclude-newer = "2025-01-01T00:00:00Z"

# Limit resolved environments
environments = [
    "sys_platform == 'linux'",
    "sys_platform == 'darwin'",
]

# Package indexes
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "private"
url = "https://pypi.internal.com/simple"

# Build settings
no-build-isolation-package = ["flash-attn"]

# Workspace configuration
[tool.uv.workspace]
members = ["packages/*"]
exclude = ["packages/deprecated"]

# Conflicting extras/groups
conflicts = [
    [
        { extra = "cuda11" },
        { extra = "cuda12" },
    ],
]

# Provide metadata for packages that don't declare it
[[tool.uv.dependency-metadata]]
name = "custom-package"
version = "1.0.0"
requires-dist = ["numpy", "scipy"]
```

## uv.toml Settings

User-level configuration (without `[tool.uv]` prefix):

```toml
# ~/.config/uv/uv.toml

# Default Python version
python = "3.12"

# Index configuration
[[index]]
name = "private"
url = "https://pypi.internal.com/simple"

# Authentication
[index.private]
username = "user"
# Password via keyring or UV_INDEX_PRIVATE_PASSWORD

# Cache settings
cache-dir = "/custom/cache/path"

# Compile bytecode by default
compile-bytecode = true

# Link mode
link-mode = "copy"  # or "hardlink", "symlink"

# Concurrent downloads
concurrent-downloads = 50

# Build isolation
no-build-isolation = false
```

## Environment Variables

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `UV_CACHE_DIR` | Cache directory | `~/.cache/uv` |
| `UV_PYTHON` | Default Python version | - |
| `UV_PYTHON_PREFERENCE` | Python discovery preference | `system` |
| `UV_PYTHON_DOWNLOADS` | Auto-download Python | `automatic` |
| `UV_SYSTEM_PYTHON` | Use system Python | `0` |
| `UV_PROJECT_ENVIRONMENT` | Project venv path | `.venv` |

### Build & Install

| Variable | Description |
|----------|-------------|
| `UV_COMPILE_BYTECODE` | Compile .pyc files |
| `UV_LINK_MODE` | `copy`, `hardlink`, `symlink` |
| `UV_NO_CACHE` | Disable caching |
| `UV_NO_BUILD_ISOLATION` | Disable build isolation |
| `UV_CONCURRENT_DOWNLOADS` | Parallel downloads |

### Resolution

| Variable | Description |
|----------|-------------|
| `UV_RESOLUTION` | `highest`, `lowest`, `lowest-direct` |
| `UV_PRERELEASE` | `allow`, `disallow`, `if-necessary` |
| `UV_EXCLUDE_NEWER` | Date/duration for reproducibility |

### Indexes & Auth

| Variable | Description |
|----------|-------------|
| `UV_INDEX_URL` | Primary index URL |
| `UV_EXTRA_INDEX_URL` | Additional indexes |
| `UV_INDEX_{NAME}_USERNAME` | Index username |
| `UV_INDEX_{NAME}_PASSWORD` | Index password |
| `UV_KEYRING_PROVIDER` | Keyring backend |

### Output

| Variable | Description |
|----------|-------------|
| `UV_NO_PROGRESS` | Disable progress bars |
| `UV_QUIET` | Suppress output |
| `UV_VERBOSE` | Verbose output |
| `NO_COLOR` | Disable colored output |

## Package Indexes

### Default Index (PyPI)

```toml
[[tool.uv.index]]
url = "https://pypi.org/simple"
default = true
```

### Additional Indexes

```toml
# Private index (checked before PyPI)
[[tool.uv.index]]
name = "private"
url = "https://pypi.internal.com/simple"

# PyTorch (explicit only)
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true  # Only for packages that reference it

# Use in sources
[tool.uv.sources]
torch = { index = "pytorch" }
```

### Index Strategy

```toml
[tool.uv]
# How to handle multiple indexes
index-strategy = "first-index"  # Stop at first match (default)
# index-strategy = "unsafe-best-match"  # Check all, pick best
```

### Local Directory Index

```toml
[[tool.uv.index]]
name = "local"
url = "file:///path/to/packages"
```

## Authentication

### Environment Variables

```bash
# Basic auth
export UV_INDEX_PRIVATE_USERNAME=user
export UV_INDEX_PRIVATE_PASSWORD=secret

# Or single variable
export UV_INDEX_URL=https://user:pass@pypi.internal.com/simple
```

### Keyring

```bash
# Use system keyring
export UV_KEYRING_PROVIDER=subprocess

# Store credentials
keyring set https://pypi.internal.com/simple username
```

### .netrc

```
# ~/.netrc
machine pypi.internal.com
login username
password secret
```

### Git Authentication

```bash
# SSH (uses SSH agent)
git+ssh://git@github.com/org/repo

# HTTPS with token
git+https://token@github.com/org/repo
```

## Cache Management

### Cache Location

```bash
# View cache directory
uv cache dir

# Default locations:
# Linux: ~/.cache/uv
# macOS: ~/Library/Caches/uv
# Windows: %LOCALAPPDATA%\uv\cache
```

### Cache Commands

```bash
# Clear all cache
uv cache clean

# Clear specific package
uv cache clean requests

# View cache size
uv cache prune --dry-run
```

### Disable Cache

```bash
# Per-command
uv sync --no-cache

# Environment variable
UV_NO_CACHE=1 uv sync
```

## Project-Specific Config

### .python-version

```
3.12
```

### .env Support

uv doesn't read .env files directly, but works with shell:

```bash
# Load .env then run
set -a && source .env && set +a && uv run python app.py

# Or use python-dotenv in your code
```

### Tool-Specific Config

```toml
# pyproject.toml

[tool.ruff]
line-length = 100

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```
