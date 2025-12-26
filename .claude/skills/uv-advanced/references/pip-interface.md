# pip Interface

uv provides a drop-in replacement for pip, pip-tools, and virtualenv with 10-100x faster performance.

## Table of Contents
- [Quick Migration](#quick-migration)
- [Virtual Environments](#virtual-environments)
- [Installing Packages](#installing-packages)
- [Compiling Requirements](#compiling-requirements)
- [Syncing Environments](#syncing-environments)
- [Inspecting Environments](#inspecting-environments)
- [Advanced Features](#advanced-features)
- [Compatibility Notes](#compatibility-notes)

## Quick Migration

### Drop-in Replacement

| pip Command | uv Command |
|-------------|------------|
| `pip install requests` | `uv pip install requests` |
| `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| `pip uninstall requests` | `uv pip uninstall requests` |
| `pip freeze` | `uv pip freeze` |
| `pip list` | `uv pip list` |
| `pip-compile requirements.in` | `uv pip compile requirements.in` |
| `pip-sync requirements.txt` | `uv pip sync requirements.txt` |
| `python -m venv .venv` | `uv venv` |

### Alias Setup

```bash
# Add to ~/.bashrc or ~/.zshrc
alias pip="uv pip"
alias pip-compile="uv pip compile"
alias pip-sync="uv pip sync"
```

## Virtual Environments

### Create Environment

```bash
# Default (.venv)
uv venv

# Custom path
uv venv myenv

# Specific Python version
uv venv --python 3.11

# With seed packages (pip, setuptools)
uv venv --seed

# System site-packages access
uv venv --system-site-packages
```

### Environment Discovery

uv automatically uses:
1. Active venv (`VIRTUAL_ENV`)
2. `.venv` in current or parent directories

```bash
# Force system Python
uv pip install --system requests

# Or via environment variable
UV_SYSTEM_PYTHON=1 uv pip install requests
```

## Installing Packages

### Basic Installation

```bash
# Single package
uv pip install requests

# Multiple packages
uv pip install requests rich httpx

# From requirements file
uv pip install -r requirements.txt

# Editable install
uv pip install -e .
uv pip install -e ./mylib
```

### Version Constraints

```bash
uv pip install "requests>=2.28"
uv pip install "requests>=2.28,<3"
uv pip install "requests==2.31.0"
uv pip install "requests~=2.28"  # >=2.28,<2.29
```

### Install Extras

```bash
uv pip install "fastapi[standard]"
uv pip install ".[dev]"
```

### Install from Sources

```bash
# Git
uv pip install git+https://github.com/psf/requests
uv pip install git+https://github.com/psf/requests@v2.31.0
uv pip install git+ssh://git@github.com/org/private-repo

# URL
uv pip install https://files.pythonhosted.org/packages/.../requests-2.31.0.tar.gz

# Local
uv pip install ./downloads/requests-2.31.0.tar.gz
```

### Reinstall

```bash
# Force reinstall
uv pip install --reinstall requests

# Reinstall all
uv pip install --reinstall-package requests -r requirements.txt
```

### Uninstall

```bash
uv pip uninstall requests
uv pip uninstall requests rich httpx
```

## Compiling Requirements

### Basic Compilation

```bash
# From requirements.in
uv pip compile requirements.in -o requirements.txt

# From pyproject.toml
uv pip compile pyproject.toml -o requirements.txt

# From setup.py
uv pip compile setup.py -o requirements.txt
```

### Universal (Cross-Platform)

```bash
# Universal lockfile with markers
uv pip compile --universal requirements.in -o requirements.txt
```

Output includes platform markers:

```
numpy==1.24.0 ; python_version < "3.9"
numpy==2.0.0 ; python_version >= "3.9"
```

### Resolution Strategies

```bash
# Lowest compatible versions
uv pip compile --resolution lowest requirements.in

# Lowest for direct deps only
uv pip compile --resolution lowest-direct requirements.in

# Upgrade all
uv pip compile --upgrade requirements.in

# Upgrade specific package
uv pip compile --upgrade-package requests requirements.in
```

### Platform-Specific

```bash
# For different Python version
uv pip compile --python-version 3.10 requirements.in

# For different platform
uv pip compile --python-platform linux requirements.in
```

### With Constraints/Overrides

```bash
# Apply constraints
uv pip compile --constraint constraints.txt requirements.in

# Apply overrides
uv pip compile --override overrides.txt requirements.in

# Build constraints
uv pip compile --build-constraint build-constraints.txt requirements.in
```

### Compile Options

```bash
# Include hashes (for verification)
uv pip compile --generate-hashes requirements.in

# Exclude annotations (cleaner output)
uv pip compile --no-annotate requirements.in

# Strip extras from output
uv pip compile --no-strip-extras requirements.in

# Custom index
uv pip compile --index-url https://pypi.internal.com/simple requirements.in

# Reproducible (exclude newer)
uv pip compile --exclude-newer "2025-01-01" requirements.in
```

### With Extras

```bash
# Include all extras
uv pip compile --all-extras pyproject.toml

# Specific extras
uv pip compile --extra dev --extra test pyproject.toml
```

## Syncing Environments

### Basic Sync

```bash
# Sync to match requirements exactly
uv pip sync requirements.txt

# From compiled lockfile
uv pip compile requirements.in -o requirements.txt
uv pip sync requirements.txt
```

### Sync Behavior

- Installs missing packages
- Removes extra packages
- Upgrades/downgrades to match versions

### With Constraints

```bash
uv pip sync --constraint constraints.txt requirements.txt
```

## Inspecting Environments

### List Packages

```bash
# List installed packages
uv pip list

# With versions
uv pip list --format freeze

# Outdated packages
uv pip list --outdated
```

### Freeze

```bash
# Output installed packages
uv pip freeze

# To file
uv pip freeze > requirements.txt

# Exclude specific packages
uv pip freeze --exclude pip --exclude setuptools
```

### Show Package Info

```bash
uv pip show requests
uv pip show --files requests  # Include file locations
```

### Dependency Tree

```bash
uv pip tree
uv pip tree --package requests
uv pip tree --invert  # Reverse dependencies
```

### Check Consistency

```bash
uv pip check
```

## Advanced Features

### Index Configuration

```bash
# Custom primary index
uv pip install --index-url https://pypi.internal.com/simple requests

# Extra indexes
uv pip install --extra-index-url https://pypi.internal.com/simple requests

# Find links (local directory)
uv pip install --find-links ./wheelhouse requests
```

### Pre-releases

```bash
# Allow pre-releases
uv pip install --prerelease allow requests

# Only for specific packages
uv pip install "flask>=2.0.0a1"
```

### Build Options

```bash
# No binary (build from source)
uv pip install --no-binary :all: numpy

# Only binary (no source builds)
uv pip install --only-binary :all: numpy

# Specific package from source
uv pip install --no-binary numpy numpy
```

### Offline Mode

```bash
# Install from cache only
uv pip install --offline requests
```

### Dry Run

```bash
# Show what would be installed
uv pip install --dry-run requests
```

## Compatibility Notes

### Differences from pip

1. **Virtual environment required by default**
   - Use `--system` for system Python
   - Use `UV_SYSTEM_PYTHON=1` environment variable

2. **Build constraints separate**
   - pip: `PIP_CONSTRAINT` applies to builds
   - uv: Use `--build-constraint` explicitly

3. **Pre-release handling**
   - uv requires explicit opt-in
   - More predictable behavior

4. **Legacy formats**
   - `.egg` distributions not created
   - Still recognized for compatibility

### Environment Variables

| pip Variable | uv Variable |
|--------------|-------------|
| `PIP_INDEX_URL` | `UV_INDEX_URL` |
| `PIP_EXTRA_INDEX_URL` | `UV_EXTRA_INDEX_URL` |
| `PIP_NO_CACHE_DIR` | `UV_NO_CACHE` |
| `PIP_CONSTRAINT` | `UV_CONSTRAINT` |

### Migration Checklist

1. ✅ Replace `pip install` with `uv pip install`
2. ✅ Replace `pip-compile` with `uv pip compile`
3. ✅ Replace `pip-sync` with `uv pip sync`
4. ✅ Update CI/CD scripts
5. ✅ Update Dockerfiles
6. ✅ Update documentation
7. ⚠️ Review pre-release handling
8. ⚠️ Review build constraint usage
