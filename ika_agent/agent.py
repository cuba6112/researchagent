"""
Forgetting Lab - Autonomous Continual Learning Researcher

An autonomous AI scientist that runs iterative research cycles
on catastrophic forgetting in neural networks.
"""

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools import google_search, FunctionTool, ToolContext, transfer_to_agent
from google.adk.code_executors import BuiltInCodeExecutor
import threading
# Computer use imports (optional - requires playwright)
try:
    from google.adk.tools.computer_use.computer_use_toolset import ComputerUseToolset
    from .playwright_computer import PlaywrightComputer
    COMPUTER_USE_AVAILABLE = True
except ImportError:
    COMPUTER_USE_AVAILABLE = False
    ComputerUseToolset = None
    PlaywrightComputer = None
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import json
import os

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_FILE = OUTPUT_DIR / "memory.json"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# MEMORY SYSTEM
# ============================================================================

@dataclass
class Memory:
    """Persistent memory across research cycles."""
    history: List[Dict] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    surprises: List[str] = field(default_factory=list)
    knowledge_base: Dict = field(default_factory=dict)

    def add_cycle(self, cycle_id: int, result: Dict):
        self.history.append({
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            **result
        })

    def add_insight(self, insight: str):
        self.insights.append(insight)

    def add_failure(self, failure: str):
        self.failures.append(failure)

    def add_surprise(self, surprise: str):
        self.surprises.append(surprise)

    def get_context(self) -> str:
        """Build context string for agents."""
        context = []
        if self.insights:
            context.append("KEY INSIGHTS:\n" + "\n".join(f"- {i}" for i in self.insights[-5:]))
        if self.failures:
            context.append("KNOWN FAILURES:\n" + "\n".join(f"- {f}" for f in self.failures[-5:]))
        if self.surprises:
            context.append("SURPRISE FINDINGS:\n" + "\n".join(f"- {s}" for s in self.surprises[-5:]))
        if self.history:
            recent = self.history[-3:]
            cycle_summaries = []
            for h in recent:
                # Safe slicing: handle None values
                hypothesis = h.get('hypothesis') or 'N/A'
                cycle_summaries.append(f"- Cycle {h['cycle_id']}: {hypothesis[:100]}...")
            context.append("RECENT CYCLES:\n" + "\n".join(cycle_summaries))
        return "\n\n".join(context) if context else "No prior context."

    def compact(self, keep_full_cycles: int = 3, max_insights: int = 20, max_failures: int = 10):
        """Compact memory to reduce token usage. Keeps recent cycles in full, summarizes older ones."""
        # Compact history: keep only summary for old cycles
        if len(self.history) > keep_full_cycles:
            old_cycles = self.history[:-keep_full_cycles]
            recent_cycles = self.history[-keep_full_cycles:]

            # Summarize old cycles to minimal format
            compacted = []
            for h in old_cycles:
                compacted.append({
                    "cycle_id": h.get("cycle_id"),
                    "timestamp": h.get("timestamp"),
                    "hypothesis": (h.get("hypothesis") or "")[:200],  # Truncate
                    "verdict": h.get("verdict", "unknown"),
                    "summary": (h.get("key_learning") or h.get("analysis") or "")[:300]
                })
            self.history = compacted + recent_cycles

        # Keep only recent insights/failures, deduplicate
        self.insights = list(dict.fromkeys(self.insights[-max_insights:]))
        self.failures = list(dict.fromkeys(self.failures[-max_failures:]))
        self.surprises = list(dict.fromkeys(self.surprises[-max_insights:]))

        return self

    def save(self, path: str):
        # Auto-compact before saving
        self.compact()
        with open(path, 'w') as f:
            json.dump({
                "history": self.history,
                "insights": self.insights,
                "failures": self.failures,
                "surprises": self.surprises,
                "knowledge_base": self.knowledge_base
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Memory":
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                data.setdefault("history", [])
                data.setdefault("insights", [])
                data.setdefault("failures", [])
                data.setdefault("surprises", [])
                data.setdefault("knowledge_base", {})
                return cls(**data)
        return cls()


# Global memory instance - loaded from disk
memory = Memory.load(str(MEMORY_FILE))
memory_lock = threading.Lock()


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
    global memory

    with memory_lock:
        # Reload from disk to get latest state (avoid race conditions)
        memory = Memory.load(str(MEMORY_FILE))

        cycle_id = len(memory.history) + 1

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
        elif verdict_upper == "REFUTED":
            memory.add_failure(f"Cycle {cycle_id}: {key_learning}")
        # INCONCLUSIVE: neither insight nor failure - valuable data but not conclusive
        if surprise:
            cleaned = surprise.strip()
            if cleaned and cleaned.lower() not in {"none", "n/a", "na", "no", "null"}:
                memory.add_surprise(cleaned)

        # Persist to disk
        memory.save(str(MEMORY_FILE))

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
    global memory

    # Create safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe_title = safe_title.replace(" ", "_")[:50]

    filename = f"{timestamp}_{report_type}_{safe_title}.md"
    filepath = REPORTS_DIR / filename

    # Reload memory to get accurate cycle count
    with memory_lock:
        memory = Memory.load(str(MEMORY_FILE))
        total_cycles = len(memory.history)

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

    return f"Report saved to: {filepath}"


def get_memory_context(tool_context: ToolContext = None) -> str:
    """
    Get the current memory context including past insights, failures, and recent cycles.

    Returns:
        Formatted string with memory context for the agent
    """
    global memory
    with memory_lock:
        # Reload from disk to get latest
        memory = Memory.load(str(MEMORY_FILE))
        return memory.get_context()


def get_all_cycles(tool_context: ToolContext = None) -> str:
    """
    Get detailed information about all past research cycles.

    Returns:
        JSON string with all cycle data
    """
    global memory
    with memory_lock:
        memory = Memory.load(str(MEMORY_FILE))

        if not memory.history:
            return "No research cycles have been recorded yet."

        output = []
        for cycle in memory.history:
            # Safe slicing: handle None values
            hypothesis = cycle.get('hypothesis') or 'N/A'
            output.append(f"""
## Cycle {cycle.get('cycle_id', 'N/A')} - {cycle.get('timestamp', 'N/A')}
**Hypothesis:** {hypothesis[:200]}...
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

MODEL = "gemini-3-flash-preview"
COMPUTER_USE_MODEL = "gemini-2.5-computer-use-preview-10-2025"  # No Gemini 3 computer use model yet

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
```

Code must be COMPLETE and RUNNABLE. No placeholders.""",
    code_executor=BuiltInCodeExecutor(),
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
        tools=[ComputerUseToolset(computer=PlaywrightComputer(screen_size=(1280, 936)))],
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
