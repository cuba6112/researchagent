# Forgetting Lab

An autonomous AI scientist built with Google ADK that researches catastrophic forgetting in neural networks through iterative hypothesis-critique-experiment cycles.

## Features

- **Autonomous Research Cycles**: Generates hypotheses, critiques them, runs experiments, and analyzes results
- **Multi-Agent Pipeline**: Specialized agents (Theorist, Critic, Gatekeeper, Architect, Analyst, etc.)
- **Persistent Memory**: Tracks insights, failures, and research history across sessions
- **Report Generation**: Produces scientific papers from accumulated research
- **Browser Research**: Optional Playwright integration for web research

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Google Gemini API key

## Installation

```bash
# Clone the repository
git clone https://github.com/cuba6112/researchagent.git
cd researchagent

# Install dependencies with uv
uv sync

# Set up your Gemini API key
cp ika_agent/.env.example ika_agent/.env
# Edit ika_agent/.env and add your key from https://aistudio.google.com/apikey
```

## Usage

```bash
# Run with ADK CLI
uv run adk run ika_agent

# Run with web interface
uv run adk web ika_agent
```

### Optional: Browser Research

For computer use capabilities:

```bash
# Install Playwright browsers
uv run playwright install chromium
```

## Project Structure

```
researchagent/
├── ika_agent/
│   ├── agent.py                 # Core agent definitions
│   ├── playwright_computer.py   # Browser automation
│   └── outputs/                 # Research history and reports
├── pyproject.toml
└── CLAUDE.md                    # Claude Code guidance
```

## Models

- Standard agents: `gemini-3-flash-preview`
- Computer use: `gemini-2.5-computer-use-preview-10-2025`

## Author

Jose Munoz (cuba6112@gmail.com)

## License

MIT
