# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Google ADK (Agent Development Kit) project implementing "Forgetting Lab" - an autonomous AI scientist for researching catastrophic forgetting in neural networks. The agent runs iterative research cycles: hypothesis generation, critique, experimentation, and analysis.

## Development Commands

```bash
# Install dependencies
uv sync

# Run the agent via ADK CLI
adk run ika_agent

# Run with web interface
adk web ika_agent

# Install Playwright browsers (required for computer use)
playwright install chromium
```

## Codebase Structure

```
agent1/
├── ika_agent/                    # Main agent package
│   ├── __init__.py              # Package init (exports root_agent)
│   ├── agent.py                 # Core agent definitions (~750 lines)
│   ├── playwright_computer.py   # Browser automation for computer use
│   └── .env                     # API keys (GEMINI_API_KEY)
├── outputs/                      # Persistent storage (outside agent package)
│   ├── memory.json              # Research history, insights, failures
│   └── reports/                 # Generated papers and summaries
├── .claude/
│   ├── settings.local.json      # Tool permissions
│   └── skills/                  # Project-local skills
│       ├── google-adk/
│       ├── gemini-genai/
│       ├── uv-advanced/
│       ├── prompt-engineering/
│       ├── mcp-builder/
│       └── headless-cli-agents/
├── pyproject.toml               # Dependencies: google-adk, playwright
├── main.py                      # Entry point (imports ika_agent)
├── CLAUDE.md                    # This file
├── .python-version              # Python version (uv)
└── .venv/                       # Virtual environment (uv managed)
```

## Architecture

### Agent Pipeline (`ika_agent/agent.py`)

The system uses Google ADK's SequentialAgent to chain specialized agents in a research cycle:

1. **Theorist** - Generates falsifiable hypotheses about catastrophic forgetting
2. **Critic** - Attacks hypotheses ruthlessly to find flaws
3. **Gatekeeper** - Makes PROCEED/REVISE/REJECT decisions
4. **Architect** - Designs minimal runnable experiments (has code executor)
5. **Analyst** - Provides honest numerical analysis of results
6. **Teacher** - Explains findings in simple terms
7. **Editor** - Reviews for scientific rigor
8. **Memory Saver** - Persists findings to disk

Additional standalone agents:
- **Reporter** - Generates scientific papers from accumulated research
- **Synthesizer** - Analyzes patterns across multiple research cycles
- **Browser Researcher** - Uses Playwright for web research (optional)

### Memory System

Persistent memory stored in `ika_agent/outputs/`:
- `memory.json` - Research history, insights, and failures
- `reports/` - Generated papers and summaries

The Memory dataclass tracks cycle history and provides context to agents.

### Computer Use (`ika_agent/playwright_computer.py`)

Custom `PlaywrightComputer` implementation that wraps Playwright's sync API in a ThreadPoolExecutor to avoid Windows asyncio subprocess issues. Implements the ADK `BaseComputer` interface for browser automation.

### Models

**DO NOT CHANGE THE MODEL without user permission.**

- Standard agents: `gemini-3-flash-preview`
- Computer use agent: `gemini-2.5-computer-use-preview-10-2025` (no Gemini 3 version yet)

## Key Patterns

- All agents use `output_key` to pass data through the pipeline via session state
- Custom `FunctionTool` wrappers expose memory operations to agents
- The root `forgetting_lab` agent orchestrates cycles and handles user commands
- Research cycles auto-save via the `memory_saver` agent at the end of each cycle

## Available Skills

Claude Code has access to these skills for working on this project:

| Skill | Use For |
|-------|---------|
| `google-adk` | ADK architecture, SequentialAgent, transfer_to_agent, output_key/session.state flow, FunctionTool patterns, common pitfalls |
| `gemini-genai` | Gemini API, model IDs, thinking levels, structured outputs, function calling, streaming, multimodal |
| `uv-advanced` | Python environment management, uv sync/run/add, dependency resolution |
| `prompt-engineering` | Agent instruction crafting, chain of thought, XML structuring |
| `mcp-builder` | Building MCP servers for tool integrations |
| `headless-cli-agents` | CLI automation, multi-agent orchestration patterns |
