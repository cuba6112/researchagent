# Repository Guidelines

## Project Structure & Module Organization
- `ika_agent/` holds the core agent implementation, including `agent.py` (pipeline), `playwright_computer.py` (browser automation), and the package entrypoint in `__init__.py`.
- `outputs/` stores persistent research data like `memory.json` and generated papers under `outputs/reports/`.
- `main.py` wires the package entrypoint, while `pyproject.toml` defines dependencies and the required Python version.
- `CLAUDE.md` documents architecture and agent behavior; treat it as source-of-truth guidance for this repo.

## Build, Test, and Development Commands
- `uv sync` installs dependencies into the local environment.
- Python 3.12+ is required (see `pyproject.toml`).
- `uv run adk run ika_agent` starts the agent via the ADK CLI.
- `uv run adk web ika_agent` launches the web UI.
- `uv run playwright install chromium` installs browser binaries for optional computer-use research.

## Coding Style & Naming Conventions
- Follow existing Python style in `ika_agent/`: clear function names, descriptive class names, and module-level constants in `UPPER_SNAKE_CASE`.
- Keep agent names consistent with the pipeline (e.g., `Theorist`, `Critic`, `Gatekeeper`) and prefer explicit `output_key` naming.
- No formatter or linter is configured; avoid introducing tooling changes without aligning with maintainers.

## Testing Guidelines
- There is no dedicated test suite yet. Validate changes by running the agent locally and confirming expected outputs in `outputs/`.
- If you add tests, keep them under a new `tests/` directory and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Recent commits use short, imperative subjects (e.g., "Add ...", "Initial commit: ..."). Keep messages concise and action-oriented.
- PRs should include a brief summary, how you tested (or why not), and link related issues. Include screenshots if you touch the web UI.

## Security & Configuration Tips
- Store secrets in `ika_agent/.env` (e.g., `GEMINI_API_KEY`) and do not commit them.
- Do not change model IDs without explicit approval, per `CLAUDE.md`.
