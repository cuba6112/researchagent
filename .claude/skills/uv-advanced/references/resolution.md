# Resolution

uv's resolver converts dependency requirements into concrete package versions. It supports both platform-specific and universal (cross-platform) resolution.

## Table of Contents
- [Universal vs Platform-Specific](#universal-vs-platform-specific)
- [Resolution Strategies](#resolution-strategies)
- [Constraints](#constraints)
- [Overrides](#overrides)
- [Pre-releases](#pre-releases)
- [Conflicting Dependencies](#conflicting-dependencies)
- [Reproducible Resolutions](#reproducible-resolutions)
- [Environment Limits](#environment-limits)
- [Dependency Metadata](#dependency-metadata)

## Universal vs Platform-Specific

### Universal Resolution (Default for Projects)

`uv.lock` is **universal** — works across all platforms and Python versions:

```bash
# Creates universal lockfile
uv lock
```

- Same lockfile for Linux, macOS, Windows
- Same lockfile for Python 3.10, 3.11, 3.12
- Uses markers to specify platform-specific packages
- May include multiple versions of same package for different platforms

### Platform-Specific Resolution (pip Interface)

```bash
# Platform-specific (like pip-tools)
uv pip compile requirements.in -o requirements.txt

# Universal resolution in pip interface
uv pip compile --universal requirements.in -o requirements.txt
```

### Cross-Platform Compilation

Compile for a different platform:

```bash
# On macOS, compile for Linux Python 3.10
uv pip compile --python-platform linux --python-version 3.10 requirements.in
```

## Resolution Strategies

### Default (Highest)

Prefers latest compatible version of each package:

```bash
uv lock  # Uses highest by default
```

### Lowest

Uses lowest compatible version — useful for testing library bounds:

```bash
# Lock with lowest versions
uv lock --resolution lowest

# Run tests with lowest versions
uv run --resolution lowest pytest

# Or in pip interface
uv pip compile --resolution lowest requirements.in
```

### Lowest-Direct

Lowest for direct dependencies, highest for transitive:

```bash
uv lock --resolution lowest-direct
```

### Fork Strategy

Controls version selection across Python versions:

```bash
# Default: optimize for latest per Python version
uv lock --fork-strategy requires-python

# Minimize total versions (prefer consistency)
uv lock --fork-strategy fewest
```

Example with `--fork-strategy requires-python` (default):
```
numpy==1.24.4 ; python_version == "3.8"
numpy==2.0.2 ; python_version == "3.9"
numpy==2.2.0 ; python_version >= "3.10"
```

With `--fork-strategy fewest`:
```
numpy==1.24.4  # Single version for all Python versions
```

## Constraints

Constraints narrow acceptable versions without adding dependencies:

### Constraints File

```bash
# constraints.txt
requests>=2.28,<3.0
urllib3<2.0
```

```bash
# Apply constraints
uv pip compile --constraint constraints.txt requirements.in
uv pip install --constraint constraints.txt -r requirements.txt
```

### In pyproject.toml

```toml
[tool.uv]
constraint-dependencies = [
  "requests>=2.28,<3.0",
  "urllib3<2.0",
]
```

### Build Constraints

Constraints for build-time dependencies (e.g., setuptools):

```bash
# build-constraints.txt
setuptools>=60,<70

uv pip compile --build-constraint build-constraints.txt requirements.in
```

```toml
[tool.uv]
build-constraint-dependencies = ["setuptools>=60,<70"]
```

## Overrides

Overrides **replace** declared dependencies — escape hatch for incorrect metadata:

### Use Case: Remove Incorrect Upper Bounds

Package declares `pydantic>=1.0,<2.0` but actually works with 2.x:

```toml
[tool.uv]
override-dependencies = [
  "pydantic>=1.0,<3.0",
]
```

### Override File (pip Interface)

```bash
# overrides.txt
pydantic>=1.0,<3.0

uv pip compile --override overrides.txt requirements.in
```

### Key Differences from Constraints

| Constraints | Overrides |
|------------|-----------|
| Narrow acceptable versions | Replace declared requirements |
| Additive (combined with package requirements) | Absolute (replaces package requirements) |
| Can only restrict | Can expand allowed versions |

## Pre-releases

### Default Behavior

Pre-releases included only when:
1. Direct dependency specifies pre-release: `flask>=2.0.0rc1`
2. All published versions are pre-releases

### Allow Pre-releases

```bash
# Allow for all packages
uv lock --prerelease allow

# Or add direct dependency with pre-release specifier
uv add "flask>=2.0.0rc1"
```

```toml
[tool.uv]
prerelease = "allow"
```

## Conflicting Dependencies

### Extras with Conflicts

When extras have incompatible requirements:

```toml
[project.optional-dependencies]
cuda11 = ["torch==2.0.0+cu118"]
cuda12 = ["torch==2.0.0+cu121"]

[tool.uv]
conflicts = [
  [
    { extra = "cuda11" },
    { extra = "cuda12" },
  ],
]
```

### Dependency Groups with Conflicts

```toml
[dependency-groups]
test-old = ["pytest<7"]
test-new = ["pytest>=8"]

[tool.uv]
conflicts = [
  [
    { group = "test-old" },
    { group = "test-new" },
  ],
]
```

### Workspace Member Conflicts

```toml
[tool.uv]
conflicts = [
  [
    { package = "member1", extra = "v1" },
    { package = "member2", extra = "v2" },
  ],
]
```

## Reproducible Resolutions

### Exclude Newer

Lock using only packages published before a date:

```bash
# Absolute timestamp
uv lock --exclude-newer "2025-01-01T00:00:00Z"

# Local date
uv lock --exclude-newer "2025-01-01"
```

```toml
[tool.uv]
exclude-newer = "2025-01-01T00:00:00Z"
```

### Dependency Cooldowns (Security)

Delay package updates for community vetting:

```bash
# Only packages older than 7 days
uv lock --exclude-newer "7 days"

# 30 day cooldown
uv lock --exclude-newer "30 days"
```

```toml
[tool.uv]
exclude-newer = "1 week"
```

### Per-Package Exclude Newer

```toml
[tool.uv]
exclude-newer = "1 week"
exclude-newer-package = { setuptools = "30 days" }
```

## Environment Limits

### Limit Resolution Platforms

Only solve for specific platforms:

```toml
[tool.uv]
environments = [
  "sys_platform == 'darwin'",
  "sys_platform == 'linux'",
]
# Windows packages won't be in lockfile
```

### Required Environments

Require wheels for specific platforms:

```toml
[tool.uv]
required-environments = [
  "sys_platform == 'darwin' and platform_machine == 'x86_64'"
]
# Fails if any package lacks Intel macOS wheel
```

## Dependency Metadata

Provide metadata for packages that don't declare it (avoids building from source):

```toml
[[tool.uv.dependency-metadata]]
name = "chumpy"
version = "0.70"
requires-dist = ["numpy>=1.8.1", "scipy>=0.13.0"]

[[tool.uv.dependency-metadata]]
name = "flash-attn"
version = "2.6.3"
requires-dist = ["torch", "einops"]
```

Useful for:
- Packages without static metadata
- Packages that fail to build on some platforms
- Git dependencies that require complex build environments

## Troubleshooting Resolution

### View Dependency Tree

```bash
uv tree
uv tree --package requests
uv tree --invert  # Show reverse dependencies
```

### Explain Why Package Included

```bash
uv tree --package urllib3
```

### Force Re-resolution

```bash
uv lock --refresh
```
