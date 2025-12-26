# Workspaces

Workspaces organize large codebases by splitting them into multiple packages with a shared lockfile. Inspired by Cargo workspaces.

## Table of Contents
- [Creating Workspaces](#creating-workspaces)
- [Workspace Configuration](#workspace-configuration)
- [Inter-Package Dependencies](#inter-package-dependencies)
- [Working with Members](#working-with-members)
- [Workspace Layouts](#workspace-layouts)
- [When to Use Workspaces](#when-to-use-workspaces)

## Creating Workspaces

### Initialize Workspace

```bash
# Create workspace root
uv init myworkspace
cd myworkspace

# Add workspace members
uv init --lib packages/core
uv init --lib packages/api
uv init packages/cli
```

Running `uv init` inside an existing project automatically adds it as a workspace member.

## Workspace Configuration

### Root pyproject.toml

```toml
[project]
name = "myworkspace"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["core", "api"]

[tool.uv.sources]
core = { workspace = true }
api = { workspace = true }

[tool.uv.workspace]
members = ["packages/*"]
exclude = ["packages/deprecated"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Member pyproject.toml

```toml
# packages/core/pyproject.toml
[project]
name = "core"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic>=2.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```toml
# packages/api/pyproject.toml
[project]
name = "api"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["core", "fastapi>=0.100"]

[tool.uv.sources]
core = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Inter-Package Dependencies

### Workspace Sources

Workspace members reference each other via `tool.uv.sources`:

```toml
[tool.uv.sources]
# Reference workspace member
core = { workspace = true }

# Inheritance: sources in root apply to all members
# Members can override specific sources
```

**Key points:**
- Dependencies between workspace members are always editable
- Root `tool.uv.sources` apply to all members (unless overridden)
- Member sources override root sources for the same package

## Working with Members

### Running Commands

```bash
# Run in workspace root (default)
uv run pytest

# Run in specific package
uv run --package core pytest
uv run --package api python -m api.main

# Sync specific package
uv sync --package api

# Build specific package
uv build --package core
```

### Locking

```bash
# Lock entire workspace (always operates on whole workspace)
uv lock

# Lockfile is at workspace root: uv.lock
```

## Workspace Layouts

### Standard Layout

```
myworkspace/
├── pyproject.toml          # Workspace root
├── uv.lock                 # Shared lockfile
├── packages/
│   ├── core/
│   │   ├── pyproject.toml
│   │   └── src/core/
│   ├── api/
│   │   ├── pyproject.toml
│   │   └── src/api/
│   └── cli/
│       ├── pyproject.toml
│       └── src/cli/
└── src/
    └── myworkspace/        # Root package (optional)
```

### Virtual Workspace (No Root Package)

For workspaces where the root is only a container:

```toml
# pyproject.toml
[tool.uv.workspace]
members = ["packages/*"]

# No [project] section - root is not a package
```

### Flat Layout

```
myworkspace/
├── pyproject.toml
├── uv.lock
├── core/
│   └── pyproject.toml
├── api/
│   └── pyproject.toml
└── cli/
    └── pyproject.toml
```

```toml
[tool.uv.workspace]
members = ["core", "api", "cli"]
```

## When to Use Workspaces

### Good Use Cases

✅ **Monorepo with shared dependencies**
- Multiple packages that share common dependencies
- Want consistent versions across all packages

✅ **Library with plugins**
- Core library with optional plugin packages
- Plugins depend on core

✅ **Application with internal libraries**
- Web app with shared utilities
- Microservices sharing common code

✅ **Extension modules**
- Python library with Rust/C extension in separate package

### When NOT to Use Workspaces

❌ **Conflicting requirements**
- Members need different versions of same dependency
- Use path dependencies instead

❌ **Separate environments needed**
- Each package needs isolated testing
- Different Python version requirements

### Alternative: Path Dependencies

For packages with conflicting dependencies:

```toml
# Instead of workspace, use path dependencies
[project]
name = "myapp"
dependencies = ["mylib"]

[tool.uv.sources]
mylib = { path = "../mylib" }

# Each project has its own lockfile
```

## Conflicting Extras

If workspace members have conflicting optional dependencies:

```toml
# Declare conflicts explicitly
[tool.uv]
conflicts = [
  [
    { package = "member1", extra = "cuda11" },
    { package = "member2", extra = "cuda12" },
  ],
]
```

Now `uv lock` resolves them separately, but they can't be installed together:

```bash
# This will error
uv sync --package member1 --extra cuda11 --package member2 --extra cuda12
```

## Workspace-Wide Settings

Settings in the workspace root apply to all members:

```toml
# Root pyproject.toml
[tool.uv]
# Shared Python requirement
requires-python = ">=3.10"

# Shared index configuration
[[tool.uv.index]]
name = "private"
url = "https://pypi.internal.com/simple"

# Shared overrides
override-dependencies = ["numpy>=1.24"]

# Limit resolution environments
environments = [
  "sys_platform == 'linux'",
  "sys_platform == 'darwin'",
]
```
