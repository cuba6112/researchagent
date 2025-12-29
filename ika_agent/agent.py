"""
Forgetting Lab - Autonomous Continual Learning Researcher

An autonomous AI scientist that runs iterative research cycles
on catastrophic forgetting in neural networks.
"""

import logging

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools import google_search, FunctionTool, ToolContext, transfer_to_agent
from google.adk.code_executors import UnsafeLocalCodeExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Computer use imports (optional - requires playwright)
try:
    from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
    from .playwright_computer import PlaywrightComputer
    COMPUTER_USE_AVAILABLE = True
    logger.info("Computer use (Playwright) is available")
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    ComputerUseToolset = None
    PlaywrightComputer = None
    logger.info("Computer use (Playwright) is not available - install playwright to enable")
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
import time

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_FILE = OUTPUT_DIR / "memory.json"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory system behavior."""
    # Context display limits
    recent_insights_limit: int = 5
    recent_failures_limit: int = 5
    recent_surprises_limit: int = 5
    recent_cycles_limit: int = 3

    # Compaction settings
    keep_full_cycles: int = 3
    max_insights: int = 20
    max_failures: int = 10
    max_surprises: int = 20

    # Truncation lengths
    hypothesis_preview_length: int = 100
    hypothesis_compact_length: int = 200
    summary_compact_length: int = 300

    # File locking
    lock_timeout_seconds: float = 30.0
    stale_lock_age_seconds: float = 60.0


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent behavior."""
    # Model settings
    default_model: str = "gemini-3-flash-preview"
    computer_use_model: str = "gemini-2.5-computer-use-preview-10-2025"

    # Browser settings
    browser_screen_size: tuple = (1280, 936)


# Default configuration instances
MEMORY_CONFIG = MemoryConfig()
AGENT_CONFIG = AgentConfig()


# ============================================================================
# MEMORY SYSTEM
# ============================================================================

@dataclass
class Memory:
    """Persistent memory across research cycles."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    surprises: List[str] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)

    def add_cycle(self, cycle_id: int, result: Dict[str, Any]) -> None:
        self.history.append({
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            **result
        })

    def add_insight(self, insight: str) -> None:
        self.insights.append(insight)

    def add_failure(self, failure: str) -> None:
        self.failures.append(failure)

    def add_surprise(self, surprise: str) -> None:
        self.surprises.append(surprise)

    def get_context(self, config: Optional[MemoryConfig] = None) -> str:
        """Build context string for agents."""
        cfg = config or MEMORY_CONFIG
        context = []
        if self.insights:
            recent_insights = self.insights[-cfg.recent_insights_limit:]
            context.append("KEY INSIGHTS:\n" + "\n".join(f"- {i}" for i in recent_insights))
        if self.failures:
            recent_failures = self.failures[-cfg.recent_failures_limit:]
            context.append("KNOWN FAILURES:\n" + "\n".join(f"- {f}" for f in recent_failures))
        if self.surprises:
            recent_surprises = self.surprises[-cfg.recent_surprises_limit:]
            context.append("SURPRISE FINDINGS:\n" + "\n".join(f"- {s}" for s in recent_surprises))
        if self.history:
            recent = self.history[-cfg.recent_cycles_limit:]
            cycle_summaries = []
            for h in recent:
                # Safe slicing: handle None values
                hypothesis = h.get('hypothesis') or 'N/A'
                preview = hypothesis[:cfg.hypothesis_preview_length]
                cycle_summaries.append(f"- Cycle {h['cycle_id']}: {preview}...")
            context.append("RECENT CYCLES:\n" + "\n".join(cycle_summaries))
        return "\n\n".join(context) if context else "No prior context."

    def compact(self, config: Optional[MemoryConfig] = None) -> "Memory":
        """Compact memory to reduce token usage. Keeps recent cycles in full, summarizes older ones."""
        cfg = config or MEMORY_CONFIG

        # Compact history: keep only summary for old cycles
        if len(self.history) > cfg.keep_full_cycles:
            old_cycles = self.history[:-cfg.keep_full_cycles]
            recent_cycles = self.history[-cfg.keep_full_cycles:]

            # Summarize old cycles to minimal format
            compacted: List[Dict[str, Any]] = []
            for h in old_cycles:
                hypothesis = (h.get("hypothesis") or "")[:cfg.hypothesis_compact_length]
                summary = (h.get("key_learning") or h.get("analysis") or "")[:cfg.summary_compact_length]
                compacted.append({
                    "cycle_id": h.get("cycle_id"),
                    "timestamp": h.get("timestamp"),
                    "hypothesis": hypothesis,
                    "verdict": h.get("verdict", "unknown"),
                    "summary": summary
                })
            self.history = compacted + recent_cycles

        # Keep only recent insights/failures, deduplicate
        self.insights = list(dict.fromkeys(self.insights[-cfg.max_insights:]))
        self.failures = list(dict.fromkeys(self.failures[-cfg.max_failures:]))
        self.surprises = list(dict.fromkeys(self.surprises[-cfg.max_surprises:]))

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "history": self.history,
            "insights": self.insights,
            "failures": self.failures,
            "surprises": self.surprises,
            "knowledge_base": self.knowledge_base
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create Memory from dictionary."""
        data.setdefault("history", [])
        data.setdefault("insights", [])
        data.setdefault("failures", [])
        data.setdefault("surprises", [])
        data.setdefault("knowledge_base", {})
        return cls(**data)


class MemoryManager:
    """
    Singleton manager for persistent memory with file-level locking.

    Provides thread-safe and process-safe access to the memory file.
    Use MemoryManager.get() to access the singleton instance.
    """
    _instance: Optional["MemoryManager"] = None
    _init_lock = threading.Lock()

    def __init__(self, memory_path: Path):
        """Initialize the memory manager. Use MemoryManager.get() instead."""
        self._memory_path = memory_path
        self._lock_path = memory_path.with_suffix(".lock")
        self._thread_lock = threading.Lock()
        self._memory: Optional[Memory] = None
        logger.debug(f"MemoryManager initialized with path: {memory_path}")

    @classmethod
    def get(cls, memory_path: Optional[Path] = None) -> "MemoryManager":
        """Get the singleton MemoryManager instance."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    path = memory_path or MEMORY_FILE
                    cls._instance = cls(path)
                    logger.info(f"MemoryManager singleton created for: {path}")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        with cls._init_lock:
            cls._instance = None
            logger.debug("MemoryManager singleton reset")

    def _acquire_file_lock(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a file-level lock for cross-process safety.
        Uses a .lock file with atomic creation.
        """
        timeout = timeout if timeout is not None else MEMORY_CONFIG.lock_timeout_seconds
        stale_age = MEMORY_CONFIG.stale_lock_age_seconds
        start_time = time.time()

        while True:
            try:
                # Try to create lock file exclusively
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                logger.debug(f"File lock acquired: {self._lock_path}")
                return True
            except FileExistsError:
                # Lock file exists, check if stale
                try:
                    lock_age = time.time() - os.path.getmtime(self._lock_path)
                    if lock_age > stale_age:
                        logger.warning(f"Removing stale lock file (age: {lock_age:.1f}s)")
                        os.remove(self._lock_path)
                        continue
                except OSError:
                    pass  # Lock file was removed by another process

                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for file lock after {timeout}s")
                    return False

                time.sleep(0.1)  # Wait and retry

    def _release_file_lock(self) -> None:
        """Release the file-level lock."""
        try:
            os.remove(self._lock_path)
            logger.debug(f"File lock released: {self._lock_path}")
        except OSError as e:
            logger.warning(f"Error releasing file lock: {e}")

    def load(self) -> Memory:
        """Load memory from disk with file locking."""
        with self._thread_lock:
            if not self._acquire_file_lock():
                raise RuntimeError("Could not acquire memory file lock")
            try:
                if os.path.exists(self._memory_path):
                    with open(self._memory_path, 'r') as f:
                        data = json.load(f)
                        self._memory = Memory.from_dict(data)
                        logger.info(f"Memory loaded (cycles={len(self._memory.history)}, insights={len(self._memory.insights)})")
                else:
                    self._memory = Memory()
                    logger.info("No existing memory, starting fresh")
                return self._memory
            finally:
                self._release_file_lock()

    def save(self, memory: Optional[Memory] = None, auto_compact: bool = True) -> None:
        """Save memory to disk with file locking."""
        with self._thread_lock:
            mem = memory or self._memory
            if mem is None:
                raise ValueError("No memory to save")

            if auto_compact:
                mem.compact()

            if not self._acquire_file_lock():
                raise RuntimeError("Could not acquire memory file lock")
            try:
                with open(self._memory_path, 'w') as f:
                    json.dump(mem.to_dict(), f, indent=2)
                self._memory = mem
                logger.info(f"Memory saved (cycles={len(mem.history)}, insights={len(mem.insights)})")
            finally:
                self._release_file_lock()

    def with_memory(self, func):
        """
        Context manager decorator for atomic memory operations.

        Usage:
            @memory_manager.with_memory
            def my_operation(memory: Memory) -> str:
                memory.add_insight("New insight")
                return "done"
        """
        def wrapper(*args, **kwargs):
            with self._thread_lock:
                if not self._acquire_file_lock():
                    raise RuntimeError("Could not acquire memory file lock")
                try:
                    # Load latest
                    if os.path.exists(self._memory_path):
                        with open(self._memory_path, 'r') as f:
                            self._memory = Memory.from_dict(json.load(f))
                    else:
                        self._memory = Memory()

                    # Execute function
                    result = func(self._memory, *args, **kwargs)

                    # Save changes
                    self._memory.compact()
                    with open(self._memory_path, 'w') as f:
                        json.dump(self._memory.to_dict(), f, indent=2)

                    return result
                finally:
                    self._release_file_lock()
        return wrapper

    @property
    def memory(self) -> Memory:
        """Get the current memory, loading if necessary."""
        if self._memory is None:
            self.load()
        return self._memory

    @property
    def cycle_count(self) -> int:
        """Get the number of research cycles."""
        return len(self.memory.history)


# Create singleton instance
memory_manager = MemoryManager.get()


# ============================================================================
# FILE-WRITING TOOLS
# ============================================================================

def save_cycle_to_memory(
    hypothesis: str,
    critique: str,
    gate_decision: str,
    analysis: str,
    verdict: str,
    key_learning: str,
    surprise: str = "",
    tool_context: ToolContext = None
) -> str:
    """
    Save a completed research cycle to persistent memory.

    Args:
        hypothesis: The hypothesis that was tested
        critique: The critique of the hypothesis
        gate_decision: PROCEED/REVISE/REJECT decision
        analysis: The analysis results
        verdict: SUPPORTED/REFUTED/INCONCLUSIVE
        key_learning: One sentence summary of what was learned

    Returns:
        Confirmation message with cycle ID
    """
    logger.debug(f"Saving cycle to memory (verdict={verdict})")

    # Use memory manager for atomic load-modify-save
    memory = memory_manager.load()

    cycle_id = len(memory.history) + 1
    logger.info(f"Creating research cycle {cycle_id} with verdict: {verdict}")

    result = {
        "hypothesis": hypothesis,
        "critique": critique,
        "gate_decision": gate_decision,
        "analysis": analysis,
        "verdict": verdict,
        "key_learning": key_learning,
        "surprise": surprise
    }

    memory.add_cycle(cycle_id, result)

    # Add to insights or failures based on verdict
    verdict_upper = verdict.upper()
    if verdict_upper == "SUPPORTED":
        memory.add_insight(key_learning)
        logger.info(f"Cycle {cycle_id}: Added insight - {key_learning[:50]}...")
    elif verdict_upper == "REFUTED":
        memory.add_failure(f"Cycle {cycle_id}: {key_learning}")
        logger.info(f"Cycle {cycle_id}: Recorded failure")
    # INCONCLUSIVE: neither insight nor failure - valuable data but not conclusive
    if surprise:
        cleaned = surprise.strip()
        if cleaned and cleaned.lower() not in {"none", "n/a", "na", "no", "null"}:
            memory.add_surprise(cleaned)
            logger.info(f"Cycle {cycle_id}: Recorded surprise finding")

    # Persist to disk with file locking
    memory_manager.save(memory)

    return f"Cycle {cycle_id} saved to memory. Total cycles: {len(memory.history)}"


def clear_cycle_state(
    keys: Optional[List[str]] = None,
    tool_context: ToolContext = None
) -> str:
    """
    Clear transient cycle state to avoid stale data when a cycle is rejected or revised.

    Args:
        keys: Specific state keys to clear; defaults to experiment/analysis outputs.
    """
    if tool_context is None:
        return "No tool_context available; state not cleared."

    keys = keys or [
        "experiment_code",
        "analysis",
        "simple_explanation",
        "editorial_review",
    ]
    tool_context.actions.state_delta = {key: "" for key in keys}
    return f"Cleared state keys: {', '.join(keys)}"


def save_report(
    title: str,
    content: str,
    report_type: str = "summary",
    tool_context: ToolContext = None
) -> str:
    """
    Save a report or paper to disk as a markdown file.

    Args:
        title: Title of the report (used for filename)
        content: Full markdown content of the report
        report_type: Type of report (summary, paper, findings)

    Returns:
        Path to the saved report
    """
    logger.info(f"Generating {report_type} report: {title[:50]}...")

    # Create safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.replace(" ", "_")[:50]

    filename = f"{timestamp}_{report_type}_{safe_title}.md"
    filepath = REPORTS_DIR / filename

    # Get accurate cycle count from memory manager
    total_cycles = memory_manager.cycle_count

    # Add metadata header
    full_content = f"""---
title: {title}
type: {report_type}
generated: {datetime.now().isoformat()}
total_cycles: {total_cycles}
---

{content}
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)

    logger.info(f"Report saved to: {filepath} ({len(content)} chars)")
    return f"Report saved to: {filepath}"


def get_memory_context(tool_context: ToolContext = None) -> str:
    """
    Get the current memory context including past insights, failures, and recent cycles.

    Returns:
        Formatted string with memory context for the agent
    """
    # Load latest from disk via memory manager
    memory = memory_manager.load()
    return memory.get_context()


def get_all_cycles(tool_context: ToolContext = None) -> str:
    """
    Get detailed information about all past research cycles.

    Returns:
        JSON string with all cycle data
    """
    # Load latest from disk via memory manager
    memory = memory_manager.load()

    if not memory.history:
        return "No research cycles have been recorded yet."

    output = []
    for cycle in memory.history:
        # Safe slicing: handle None values
        hypothesis = cycle.get('hypothesis') or 'N/A'
        hypothesis_preview = hypothesis[:MEMORY_CONFIG.hypothesis_compact_length]
        output.append(f"""
## Cycle {cycle.get('cycle_id', 'N/A')} - {cycle.get('timestamp', 'N/A')}
**Hypothesis:** {hypothesis_preview}...
**Verdict:** {cycle.get('verdict', 'N/A')}
**Key Learning:** {cycle.get('key_learning', 'N/A')}
""")

    return "\n".join(output)


# Create FunctionTool wrappers
save_cycle_tool = FunctionTool(func=save_cycle_to_memory)
clear_cycle_state_tool = FunctionTool(func=clear_cycle_state)
save_report_tool = FunctionTool(func=save_report)
get_memory_tool = FunctionTool(func=get_memory_context)
get_cycles_tool = FunctionTool(func=get_all_cycles)


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

# Model configuration (from AGENT_CONFIG)
MODEL = AGENT_CONFIG.default_model
COMPUTER_USE_MODEL = AGENT_CONFIG.computer_use_model  # No Gemini 3 computer use model yet

# Theorist: Generate falsifiable hypothesis
theorist = Agent(
    name="theorist",
    model=MODEL,
    description="Generates falsifiable research hypotheses",
    instruction="""You are the THEORIST - a hypothesis generator for continual learning research.

YOUR TASK:
Generate ONE falsifiable hypothesis about catastrophic forgetting in neural networks.

FIRST: Use get_memory_context and get_all_cycles tools to review past research.
- Check what hypotheses have already been tested
- Avoid repeating failed approaches
- Build on successful insights

REQUIREMENTS:
1. Must be FALSIFIABLE - specify exact condition that would refute it
2. Must specify EXPECTED EFFECT SIZE vs baseline (e.g., "5% improvement")
3. Must use a STANDARD BENCHMARK (PermutedMNIST, SplitMNIST, etc.)
4. Must be NOVEL - do NOT repeat hypotheses from past cycles

OUTPUT FORMAT:
HYPOTHESIS: [One sentence]
FALSIFICATION CONDITION: [What result would prove this wrong]
EXPECTED EFFECT: [X% improvement/change vs baseline]
BENCHMARK: [Which standard benchmark]
RATIONALE: [Why this might work, cite papers if relevant]

Focus on REAL techniques: EWC, SAM, PackNet, memory replay, regularization.
Do NOT propose vague "novel architectures" - be specific and runnable.""",
    tools=[get_memory_tool, get_cycles_tool],
    output_key="hypothesis"
)

# Critic: Attack hypothesis ruthlessly
critic = Agent(
    name="critic",
    model=MODEL,
    description="Ruthlessly critiques hypotheses",
    instruction="""You are the CRITIC - your job is to ATTACK the hypothesis ruthlessly.

HYPOTHESIS TO CRITIQUE:
{hypothesis}

Find every weakness, flaw, and potential failure mode.

CONSIDER:
1. Is it actually falsifiable or will any result be "explained away"?
2. Is the expected effect size realistic?
3. Has this been tried before? What happened?
4. Are there confounding variables?
5. Is the benchmark appropriate?
6. Can this actually be implemented in reasonable code?

OUTPUT FORMAT:
WEAKNESSES:
1. [Major flaw 1]
2. [Major flaw 2]

STRENGTHS (if any):
1. [Strength]

FATAL FLAWS: [Yes/No - list any that make this untestable]
RECOMMENDATION: [Specific improvements needed]

Be HARSH. Failed hypotheses waste compute.""",
    output_key="critique"
)

# Gatekeeper: PROCEED/REVISE/REJECT decision
gatekeeper = Agent(
    name="gatekeeper",
    model=MODEL,
    description="Makes go/no-go decisions on hypotheses",
    instruction="""You are the GATEKEEPER - you make the final decision.

HYPOTHESIS:
{hypothesis}

CRITIQUE:
{critique}

Decide: PROCEED, REVISE, or REJECT

DECISION CRITERIA:
- PROCEED: Hypothesis is sound, testable, and worth running. Continue to experiment.
- REVISE: Good core idea but needs specific fixes. Use transfer_to_agent to go back to theorist.
- REJECT: Fundamentally flawed, untestable, or already tried. Use transfer_to_agent to skip to memory_saver.

OUTPUT FORMAT:
DECISION: [PROCEED/REVISE/REJECT]
REASONING: [2-3 sentences]
ACTION: [If REVISE: specific changes. If REJECT: why]

IMPORTANT:
- If REJECT: You MUST call clear_cycle_state, then transfer_to_agent("memory_saver") to skip the experiment phase.
- If REVISE: You MUST call clear_cycle_state, then transfer_to_agent("theorist") to generate a new hypothesis.
- If PROCEED: Do NOT call transfer_to_agent, just output your decision and the pipeline continues.

Default to REJECT if uncertain.""",
    tools=[transfer_to_agent, clear_cycle_state_tool],
    output_key="gate_decision"
)

# Architect: Design minimal runnable experiment
architect = Agent(
    name="architect",
    model=MODEL,
    description="Designs minimal runnable experiments",
    instruction="""You are the ARCHITECT - design MINIMAL runnable experiments.

HYPOTHESIS: {hypothesis}
GATE DECISION: {gate_decision?}

Design the SIMPLEST possible experiment to test this hypothesis.

REQUIREMENTS:
1. Must run in < 5 minutes on CPU
2. Must use standard libraries (torch, numpy, matplotlib)
3. Must have CLEAR success/failure metrics defined BEFORE running
4. Must include baseline comparison
5. Must output numerical results
6. Must run at least 3 random seeds (or explain why not)
7. Must include at least one ablation that removes the key proposed component
8. Must log the full experiment config (seed, dataset, model, hyperparams)

OUTPUT FORMAT:
```python
# EXPERIMENT: [Title]
# HYPOTHESIS: [One line]
# SUCCESS CRITERION: [Specific number]
# BASELINE: [What we compare against]

import torch
import torch.nn as nn
# ... complete runnable code ...

if __name__ == "__main__":
    # Run experiment
    # Print: RESULT: X% | BASELINE: Y% | DELTA: Z%
    # Print: SEEDS: [...] | MEAN: X | STD: Y
    # Print: CONFIG: {...}
``` 

Code must be COMPLETE and RUNNABLE. No placeholders.""",
    code_executor=UnsafeLocalCodeExecutor(),
    output_key="experiment_code"
)

# Analyst: Honest analysis with numbers
analyst = Agent(
    name="analyst",
    model=MODEL,
    description="Provides honest numerical analysis",
    instruction="""You are the ANALYST - analyze results with BRUTAL HONESTY.

HYPOTHESIS: {hypothesis}
EXPERIMENT OUTPUT: {experiment_code?}

The experiment output above contains the code that was designed and executed, along with its results.
Analyze what actually happened. No spin.

OUTPUT FORMAT:
WHAT HAPPENED: [Raw numbers from output]
VS BASELINE: [X% vs Y% = delta Z%]
SIGNIFICANCE: [p-value if available, or "n=1, not significant"]
VERDICT: [SUPPORTED / REFUTED / INCONCLUSIVE]
SURPRISES: [Unexpected results or anomalies, or "None"]
SEED COVERAGE: [n, mean, std or "single-run"]
ABLATION CHECK: [Passed/Failed + brief note]

HONEST ASSESSMENT: [2-3 sentences]
NEXT STEPS: [Only if supported]

FAILURE IS VALUABLE DATA. Do not dress up failures.""",
    output_key="analysis"
)

# Teacher: Explain to 12-year-old
teacher = Agent(
    name="teacher",
    model=MODEL,
    description="Explains findings simply",
    instruction="""You are the TEACHER - explain to a smart 12-year-old.

HYPOTHESIS: {hypothesis}
ANALYSIS: {analysis?}

Explain what we tried and what happened in simple terms.

RULES:
1. No jargon - explain technical terms
2. Use analogies
3. Be honest about what worked/didn't
4. Make it interesting

OUTPUT FORMAT:
WHAT WE TRIED: [1-2 sentences, simple]
WHAT WE EXPECTED: [1 sentence]
WHAT ACTUALLY HAPPENED: [1-2 sentences]
WHAT WE LEARNED: [1-2 sentences]""",
    output_key="simple_explanation"
)

# Editor: Scientific rigor review
editor = Agent(
    name="editor",
    model=MODEL,
    description="Reviews for scientific rigor",
    instruction="""You are the EDITOR - ensure scientific rigor.

CYCLE RESULTS:
Hypothesis: {hypothesis}
Critique: {critique}
Gate Decision: {gate_decision?}
Experiment Code & Output: {experiment_code?}
Analysis: {analysis?}
Simple Explanation: {simple_explanation?}

CHECK FOR:
1. ABSTRACTION TRAP: Did we run code or just talk? Check experiment_code for actual execution output.
2. FABRICATED SUCCESS: Do numbers support conclusion?
3. CITATION HALLUCINATION: Are papers real?
4. METRIC THEATER: Right measurements?
5. REPRODUCIBILITY: Can others replicate?
6. SEED ROBUSTNESS: Were multiple seeds reported?
7. ABLATIONS: Was a key-component ablation included?
8. CONFIG LOGGING: Were full config details logged?

OUTPUT FORMAT:
RIGOR SCORE: [1-10]
ISSUES FOUND:
- [Issue 1]
- [Issue 2]

VERDICT: [PUBLISH / REVISE / DISCARD]
KEY LEARNING: [One sentence for memory]
SURPRISES: [Unexpected or notable surprises, or "None"]""",
    output_key="editorial_review"
)

# Memory Saver: Automatically saves cycle results
memory_saver = Agent(
    name="memory_saver",
    model=MODEL,
    description="Saves research cycle results to persistent memory",
    instruction="""You are the MEMORY SAVER - you persist research findings.

After each research cycle, extract and save the key information.

CYCLE DATA:
Hypothesis: {hypothesis}
Critique: {critique}
Gate Decision: {gate_decision?}
Experiment Code & Output: {experiment_code?}
Analysis: {analysis?}
Simple Explanation: {simple_explanation?}
Editorial Review: {editorial_review?}

YOUR TASK:
1. Extract the VERDICT from the analysis (SUPPORTED/REFUTED/INCONCLUSIVE)
2. Extract the KEY LEARNING from the editorial review
3. Extract any SURPRISE FINDING from the analysis or editorial review
   - Prefer the analyst's SURPRISES field if both are present
4. Use the save_cycle_to_memory tool to persist this cycle

NOTE: If the gate decision was REJECT, the experiment_code, analysis, and simple_explanation
may be empty or contain placeholder text. In that case, set verdict to "REJECTED" and
extract key learning from the gate_decision reasoning.

Be concise and accurate. Extract exact values, don't paraphrase.""",
    tools=[save_cycle_tool],
    output_key="memory_save_result"
)

# Reporter: Generates scientific papers/reports
reporter = Agent(
    name="reporter",
    model=MODEL,
    description="Generates scientific reports and papers from research findings",
    instruction="""You are the REPORTER - you write scientific papers and reports.

When asked to generate a report, use the get_all_cycles tool to retrieve past research.

REPORT TYPES:
1. "summary" - Brief overview of all findings
2. "paper" - Full scientific paper format
3. "findings" - Detailed analysis of what worked/didn't

PAPER FORMAT (for type="paper"):
# Title: [Descriptive title about catastrophic forgetting research]

## Abstract
[3-4 sentences summarizing the research and key findings]

## Introduction
[Context about catastrophic forgetting, why it matters]

## Methods
[Overview of the Forgetting Lab approach: hypothesis-critique-experiment cycle]

## Results
[Summary of each cycle's findings with actual numbers]

## Discussion
[What patterns emerged? What approaches showed promise? What failed?]

## Conclusions
[Key takeaways and future directions]

## References
[If any papers were cited]

After generating the report, use save_report tool to persist it.

Be scientifically rigorous. Include actual numbers from experiments.
Acknowledge failures honestly - they are valuable data.""",
    tools=[get_cycles_tool, get_memory_tool, save_report_tool],
    output_key="report_content"
)

# Synthesizer: Analyzes patterns across cycles
synthesizer = Agent(
    name="synthesizer",
    model=MODEL,
    description="Synthesizes insights across multiple research cycles",
    instruction="""You are the SYNTHESIZER - you find patterns across research.

Use the get_all_cycles and get_memory_context tools to review past research.

YOUR TASK:
Analyze all completed cycles and identify:

1. PATTERNS OF SUCCESS
   - What approaches showed promise?
   - Common elements in successful experiments?

2. PATTERNS OF FAILURE
   - What approaches consistently failed?
   - What should we avoid?

3. KNOWLEDGE GAPS
   - What hypotheses haven't been tested?
   - What follow-up experiments are needed?

4. EMERGING INSIGHTS
   - Are there unexpected connections?
   - What new hypotheses do the data suggest?

OUTPUT FORMAT:
## Synthesis Report

### Successful Approaches
- [Approach 1]: [Why it worked]

### Failed Approaches
- [Approach 1]: [Why it failed]

### Recommended Next Experiments
1. [Specific hypothesis to test next]
2. [Another hypothesis]

### Meta-Insights
[Higher-level patterns about the research process itself]

Be data-driven. Cite specific cycle numbers and results.""",
    tools=[get_cycles_tool, get_memory_tool],
    output_key="synthesis"
)

# Browser Researcher: Uses computer use to browse the web for research
# Only available if playwright is installed
if COMPUTER_USE_AVAILABLE:
    browser_researcher = Agent(
        name="browser_researcher",
        model=COMPUTER_USE_MODEL,
        description="Browses the web using computer control to gather research information",
        instruction="""You are the BROWSER RESEARCHER - you control a real browser to gather research.

You have full control of a web browser via the ComputerUseToolset. You can:
- Navigate to URLs
- Click on links and buttons
- Type in search boxes and forms
- Scroll pages
- Take screenshots
- Extract text content

YOUR TASKS:
1. Search for academic papers on arXiv, Google Scholar, or Semantic Scholar
2. Find implementation details on GitHub repositories
3. Read blog posts and tutorials about ML techniques
4. Gather benchmark results from papers
5. Find code examples for specific techniques

RESEARCH FOCUS:
- Catastrophic forgetting in neural networks
- Continual learning methods (EWC, SI, PackNet, etc.)
- Experience replay techniques
- Regularization strategies for lifelong learning

WORKFLOW:
1. Navigate to a relevant site (arXiv, GitHub, etc.)
2. Search for the requested topic
3. Extract key information (abstracts, code snippets, results)
4. Summarize findings in a structured format

OUTPUT FORMAT:
## Research Findings

### Source: [URL]
**Title:** [Paper/Repo title]
**Key Points:**
- [Point 1]
- [Point 2]

### Code Snippets (if found):
```python
[Relevant code]
```

### Benchmark Results:
| Method | Dataset | Accuracy |
|--------|---------|----------|
| ... | ... | ... |

Be thorough but efficient. Focus on actionable information.""",
        tools=[ComputerUseToolset(computer=PlaywrightComputer(screen_size=AGENT_CONFIG.browser_screen_size))],
        output_key="web_research"
    )
else:
    browser_researcher = None


# ============================================================================
# PIPELINE CONSTRUCTION
# ============================================================================

# Main research cycle: Sequential flow through all agents
research_cycle = SequentialAgent(
    name="research_cycle",
    description="One complete research cycle: hypothesis -> experiment -> analysis -> save",
    sub_agents=[
        theorist,
        critic,
        gatekeeper,
        architect,
        analyst,
        teacher,
        editor,
        memory_saver  # Auto-saves each cycle to persistent memory
    ]
)

# Root agent: Orchestrates cycles
root_agent = Agent(
    name="forgetting_lab",
    model=MODEL,
    include_contents='none',  # Prevent loading full conversation history (token overflow)
    description="Autonomous AI scientist for continual learning research on catastrophic forgetting",
    instruction="""You are FORGETTING LAB - an autonomous AI scientist.

You orchestrate research cycles to study catastrophic forgetting in neural networks.

Each research cycle follows this pipeline:
1. THEORIST - generates a falsifiable hypothesis
2. CRITIC - attacks the hypothesis ruthlessly
3. GATEKEEPER - decides PROCEED/REVISE/REJECT
4. ARCHITECT - designs minimal runnable experiment
5. ANALYST - provides honest numerical analysis
6. TEACHER - explains findings simply
7. EDITOR - reviews for scientific rigor
8. MEMORY SAVER - persists findings to disk

COMMANDS:
- "run cycle" - Execute one complete research cycle (auto-saves to memory)
- "synthesize" - Transfer to SYNTHESIZER to analyze patterns across all cycles
- "report" or "write paper" - Transfer to REPORTER to generate a scientific paper
- "show memory" - Display past insights and failures using get_memory_context tool
- "show all cycles" - Display all past research cycles using get_all_cycles tool

MEMORY SYSTEM:
All research cycles are automatically saved to disk. Use the memory tools to:
- Review past experiments before proposing new hypotheses
- Avoid repeating failed approaches
- Build on successful findings

Guide the research process with scientific rigor.
Prefer honest failures over fabricated successes.
Learn from past cycles to improve future experiments.

BROWSER RESEARCH:
- "browse" or "web research" - Transfer to BROWSER RESEARCHER to search the web
  The browser agent can navigate to arXiv, GitHub, Google Scholar, etc.
  to find papers, code, and benchmark results.
  (Requires playwright to be installed)""",
    sub_agents=[research_cycle, reporter, synthesizer] + ([browser_researcher] if browser_researcher else []),
    tools=[get_memory_tool, get_cycles_tool]
)

logger.info(f"Forgetting Lab initialized with {memory_manager.cycle_count} existing research cycles")
