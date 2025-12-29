"""
Forgetting Lab - Autonomous Continual Learning Researcher

An autonomous AI scientist that runs iterative research cycles
on catastrophic forgetting in neural networks.
"""

import logging
import hashlib
import platform
import re
import subprocess
import uuid

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
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

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


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for experiment versioning."""
    # Code extraction patterns
    code_block_pattern: str = r"```python\s*(.*?)```"

    # Hyperparameter extraction patterns (common ML params)
    hyperparam_patterns: Tuple[str, ...] = (
        r"lr\s*=\s*([\d.e-]+)",
        r"learning_rate\s*=\s*([\d.e-]+)",
        r"batch_size\s*=\s*(\d+)",
        r"epochs?\s*=\s*(\d+)",
        r"num_tasks?\s*=\s*(\d+)",
        r"hidden_size\s*=\s*(\d+)",
        r"memory_size\s*=\s*(\d+)",
        r"buffer_size\s*=\s*(\d+)",
        r"replay_.*?=\s*(\d+)",
    )

    # Seed extraction patterns
    seed_patterns: Tuple[str, ...] = (
        r"seed\s*=\s*(\d+)",
        r"random_seed\s*=\s*(\d+)",
        r"torch\.manual_seed\((\d+)\)",
        r"np\.random\.seed\((\d+)\)",
        r"seeds?\s*=\s*\[([\d,\s]+)\]",
    )


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration for metrics tracking."""
    # Metric extraction patterns
    accuracy_patterns: Tuple[str, ...] = (
        r"accuracy[:\s]+(\d+\.?\d*)%?",
        r"acc[:\s]+(\d+\.?\d*)%?",
        r"RESULT[:\s]+(\d+\.?\d*)%?",
    )
    loss_patterns: Tuple[str, ...] = (
        r"loss[:\s]+(\d+\.?\d*)",
        r"avg_loss[:\s]+(\d+\.?\d*)",
    )
    forgetting_patterns: Tuple[str, ...] = (
        r"forgetting[:\s]+(\d+\.?\d*)%?",
        r"backward_transfer[:\s]+([-\d.]+)%?",
    )


# Default configuration instances
MEMORY_CONFIG = MemoryConfig()
AGENT_CONFIG = AgentConfig()
EXPERIMENT_CONFIG = ExperimentConfig()
METRICS_CONFIG = MetricsConfig()


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
# EXPERIMENT VERSIONING SYSTEM
# ============================================================================

@dataclass
class ExperimentVersion:
    """
    Complete snapshot of an experiment for reproducibility.

    Captures code, environment, hyperparameters, seeds, and results
    to enable exact reproduction of any past experiment.
    """
    experiment_id: str
    cycle_id: int
    timestamp: str
    hypothesis: str

    # Code snapshot
    code: str
    code_hash: str  # SHA256 of code for quick comparison

    # Reproducibility info
    random_seeds: List[int] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Environment info
    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    platform_info: str = ""
    git_commit: str = ""

    # Results (populated after execution)
    results: Dict[str, Any] = field(default_factory=dict)
    execution_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "code": self.code,
            "code_hash": self.code_hash,
            "random_seeds": self.random_seeds,
            "hyperparameters": self.hyperparameters,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "numpy_version": self.numpy_version,
            "platform_info": self.platform_info,
            "git_commit": self.git_commit,
            "results": self.results,
            "execution_output": self.execution_output,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentVersion":
        """Create ExperimentVersion from dictionary."""
        return cls(
            experiment_id=data.get("experiment_id", ""),
            cycle_id=data.get("cycle_id", 0),
            timestamp=data.get("timestamp", ""),
            hypothesis=data.get("hypothesis", ""),
            code=data.get("code", ""),
            code_hash=data.get("code_hash", ""),
            random_seeds=data.get("random_seeds", []),
            hyperparameters=data.get("hyperparameters", {}),
            python_version=data.get("python_version", ""),
            torch_version=data.get("torch_version", ""),
            numpy_version=data.get("numpy_version", ""),
            platform_info=data.get("platform_info", ""),
            git_commit=data.get("git_commit", ""),
            results=data.get("results", {}),
            execution_output=data.get("execution_output", ""),
        )


class ExperimentVersionManager:
    """
    Manages experiment versions for reproducibility.

    Each experiment is stored in its own directory with:
    - experiment.py: The exact code that was run
    - metadata.json: Environment, seeds, hyperparameters
    - results.json: Execution output and parsed results
    """
    _instance: Optional["ExperimentVersionManager"] = None
    _init_lock = threading.Lock()

    def __init__(self, experiments_dir: Path):
        """Initialize the experiment version manager."""
        self._experiments_dir = experiments_dir
        self._index_path = experiments_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        logger.debug(f"ExperimentVersionManager initialized: {experiments_dir}")

    @classmethod
    def get(cls, experiments_dir: Optional[Path] = None) -> "ExperimentVersionManager":
        """Get the singleton ExperimentVersionManager instance."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    path = experiments_dir or EXPERIMENTS_DIR
                    cls._instance = cls(path)
                    logger.info(f"ExperimentVersionManager created: {path}")
        return cls._instance

    def _load_index(self) -> None:
        """Load the experiment index from disk."""
        if self._index_path.exists():
            with open(self._index_path, 'r') as f:
                self._index = json.load(f)
            logger.debug(f"Loaded {len(self._index)} experiments from index")
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save the experiment index to disk."""
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f, indent=2)

    def _get_environment_info(self) -> Dict[str, str]:
        """Capture current environment information."""
        env_info = {
            "python_version": platform.python_version(),
            "platform_info": f"{platform.system()} {platform.release()}",
        }

        # Get torch version
        try:
            import torch
            env_info["torch_version"] = torch.__version__
        except ImportError:
            env_info["torch_version"] = "not installed"

        # Get numpy version
        try:
            import numpy as np
            env_info["numpy_version"] = np.__version__
        except ImportError:
            env_info["numpy_version"] = "not installed"

        # Get git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                env_info["git_commit"] = result.stdout.strip()[:12]
        except Exception:
            env_info["git_commit"] = "unknown"

        return env_info

    def _extract_code_from_output(self, experiment_output: str) -> str:
        """Extract Python code from experiment output (code block)."""
        config = EXPERIMENT_CONFIG
        matches = re.findall(config.code_block_pattern, experiment_output, re.DOTALL)
        if matches:
            return matches[0].strip()
        # If no code block, try to find code-like content
        if "import " in experiment_output and "def " in experiment_output:
            return experiment_output.strip()
        return ""

    def _extract_seeds(self, code: str) -> List[int]:
        """Extract random seeds from code."""
        config = EXPERIMENT_CONFIG
        seeds = set()

        for pattern in config.seed_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if isinstance(match, str) and ',' in match:
                    # Handle list format: [42, 123, 456]
                    for s in match.split(','):
                        s = s.strip()
                        if s.isdigit():
                            seeds.add(int(s))
                elif match.isdigit():
                    seeds.add(int(match))

        return sorted(list(seeds))

    def _extract_hyperparameters(self, code: str) -> Dict[str, Any]:
        """Extract hyperparameters from code."""
        config = EXPERIMENT_CONFIG
        hyperparams = {}

        for pattern in config.hyperparam_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                # Extract param name from pattern
                param_name = pattern.split(r"\s*=")[0].replace(r"\s*", "").replace("\\", "")
                # Clean up the param name
                param_name = re.sub(r'[^a-zA-Z_]', '', param_name)
                try:
                    value = float(matches[0]) if '.' in matches[0] or 'e' in matches[0].lower() else int(matches[0])
                    hyperparams[param_name] = value
                except ValueError:
                    hyperparams[param_name] = matches[0]

        return hyperparams

    def _compute_code_hash(self, code: str) -> str:
        """Compute SHA256 hash of code for comparison."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def save_experiment(
        self,
        cycle_id: int,
        hypothesis: str,
        experiment_output: str,
        results: Optional[Dict[str, Any]] = None
    ) -> ExperimentVersion:
        """
        Save a new experiment version.

        Args:
            cycle_id: Research cycle ID
            hypothesis: The hypothesis being tested
            experiment_output: Raw output from architect (contains code)
            results: Optional parsed results dict

        Returns:
            The saved ExperimentVersion
        """
        # Generate unique experiment ID
        experiment_id = f"exp_{cycle_id:03d}_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        # Extract code from output
        code = self._extract_code_from_output(experiment_output)
        if not code:
            logger.warning(f"No code found in experiment output for cycle {cycle_id}")
            code = experiment_output  # Save raw output as fallback

        # Create version
        env_info = self._get_environment_info()
        version = ExperimentVersion(
            experiment_id=experiment_id,
            cycle_id=cycle_id,
            timestamp=timestamp,
            hypothesis=hypothesis,
            code=code,
            code_hash=self._compute_code_hash(code),
            random_seeds=self._extract_seeds(code),
            hyperparameters=self._extract_hyperparameters(code),
            python_version=env_info["python_version"],
            torch_version=env_info["torch_version"],
            numpy_version=env_info["numpy_version"],
            platform_info=env_info["platform_info"],
            git_commit=env_info["git_commit"],
            results=results or {},
            execution_output=experiment_output,
        )

        # Create experiment directory
        exp_dir = self._experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save code file
        code_path = exp_dir / "experiment.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        # Save metadata
        metadata_path = exp_dir / "metadata.json"
        metadata = {
            "experiment_id": version.experiment_id,
            "cycle_id": version.cycle_id,
            "timestamp": version.timestamp,
            "hypothesis": version.hypothesis,
            "code_hash": version.code_hash,
            "random_seeds": version.random_seeds,
            "hyperparameters": version.hyperparameters,
            "environment": {
                "python_version": version.python_version,
                "torch_version": version.torch_version,
                "numpy_version": version.numpy_version,
                "platform_info": version.platform_info,
                "git_commit": version.git_commit,
            }
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Save results/output
        results_path = exp_dir / "results.json"
        results_data = {
            "execution_output": version.execution_output,
            "parsed_results": version.results,
        }
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2)

        # Update index
        self._index[experiment_id] = {
            "cycle_id": cycle_id,
            "timestamp": timestamp,
            "hypothesis": hypothesis[:100],
            "code_hash": version.code_hash,
        }
        self._save_index()

        logger.info(f"Saved experiment {experiment_id} (cycle {cycle_id}, hash={version.code_hash})")
        return version

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentVersion]:
        """Load an experiment by ID."""
        exp_dir = self._experiments_dir / experiment_id

        if not exp_dir.exists():
            logger.warning(f"Experiment not found: {experiment_id}")
            return None

        # Load code
        code_path = exp_dir / "experiment.py"
        code = ""
        if code_path.exists():
            with open(code_path, 'r', encoding='utf-8') as f:
                code = f.read()

        # Load metadata
        metadata_path = exp_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Load results
        results_path = exp_dir / "results.json"
        results_data = {}
        if results_path.exists():
            with open(results_path, 'r') as f:
                results_data = json.load(f)

        env = metadata.get("environment", {})
        return ExperimentVersion(
            experiment_id=experiment_id,
            cycle_id=metadata.get("cycle_id", 0),
            timestamp=metadata.get("timestamp", ""),
            hypothesis=metadata.get("hypothesis", ""),
            code=code,
            code_hash=metadata.get("code_hash", ""),
            random_seeds=metadata.get("random_seeds", []),
            hyperparameters=metadata.get("hyperparameters", {}),
            python_version=env.get("python_version", ""),
            torch_version=env.get("torch_version", ""),
            numpy_version=env.get("numpy_version", ""),
            platform_info=env.get("platform_info", ""),
            git_commit=env.get("git_commit", ""),
            results=results_data.get("parsed_results", {}),
            execution_output=results_data.get("execution_output", ""),
        )

    def get_experiment_by_cycle(self, cycle_id: int) -> Optional[ExperimentVersion]:
        """Get the experiment for a specific cycle."""
        for exp_id, info in self._index.items():
            if info.get("cycle_id") == cycle_id:
                return self.get_experiment(exp_id)
        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summary info."""
        experiments = []
        for exp_id, info in sorted(self._index.items(), key=lambda x: x[1].get("cycle_id", 0)):
            experiments.append({
                "experiment_id": exp_id,
                "cycle_id": info.get("cycle_id"),
                "timestamp": info.get("timestamp"),
                "hypothesis": info.get("hypothesis", "")[:80] + "...",
                "code_hash": info.get("code_hash"),
            })
        return experiments

    def compare_experiments(self, exp_id_1: str, exp_id_2: str) -> Dict[str, Any]:
        """Compare two experiments and highlight differences."""
        exp1 = self.get_experiment(exp_id_1)
        exp2 = self.get_experiment(exp_id_2)

        if not exp1 or not exp2:
            return {"error": "One or both experiments not found"}

        comparison = {
            "experiment_1": exp_id_1,
            "experiment_2": exp_id_2,
            "same_code": exp1.code_hash == exp2.code_hash,
            "code_hash_1": exp1.code_hash,
            "code_hash_2": exp2.code_hash,
            "seed_diff": {
                "only_in_1": [s for s in exp1.random_seeds if s not in exp2.random_seeds],
                "only_in_2": [s for s in exp2.random_seeds if s not in exp1.random_seeds],
            },
            "hyperparam_diff": {},
            "environment_diff": {},
        }

        # Compare hyperparameters
        all_params = set(exp1.hyperparameters.keys()) | set(exp2.hyperparameters.keys())
        for param in all_params:
            v1 = exp1.hyperparameters.get(param)
            v2 = exp2.hyperparameters.get(param)
            if v1 != v2:
                comparison["hyperparam_diff"][param] = {"exp1": v1, "exp2": v2}

        # Compare environment
        env_fields = ["python_version", "torch_version", "numpy_version"]
        for field in env_fields:
            v1 = getattr(exp1, field)
            v2 = getattr(exp2, field)
            if v1 != v2:
                comparison["environment_diff"][field] = {"exp1": v1, "exp2": v2}

        return comparison

    def generate_rerun_command(self, experiment_id: str) -> str:
        """Generate a command to re-run an experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            return f"# Experiment {experiment_id} not found"

        exp_dir = self._experiments_dir / experiment_id
        code_path = exp_dir / "experiment.py"

        return f"""# Re-run experiment {experiment_id} (Cycle {exp.cycle_id})
# Original timestamp: {exp.timestamp}
# Git commit: {exp.git_commit}
# Seeds: {exp.random_seeds}
# Hyperparameters: {json.dumps(exp.hyperparameters)}

cd {exp_dir}
python experiment.py
"""

    @property
    def experiment_count(self) -> int:
        """Get total number of saved experiments."""
        return len(self._index)


# Create singleton instance
experiment_manager = ExperimentVersionManager.get()


# ============================================================================
# METRICS TRACKING SYSTEM
# ============================================================================

@dataclass
class MetricPoint:
    """A single metric measurement from an experiment."""
    cycle_id: int
    experiment_id: str
    timestamp: str
    metric_name: str
    value: float
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPoint":
        return cls(**data)


class MetricsTracker:
    """
    Tracks metrics across experiments for trend analysis and visualization.

    Stores metrics in a time-series format for easy querying and plotting.
    """
    _instance: Optional["MetricsTracker"] = None
    _init_lock = threading.Lock()

    def __init__(self, metrics_path: Path):
        self._metrics_path = metrics_path
        self._metrics: List[MetricPoint] = []
        self._tags_index: Dict[str, List[str]] = {}  # experiment_id -> tags
        self._load()

    @classmethod
    def get(cls, metrics_path: Optional[Path] = None) -> "MetricsTracker":
        """Get the singleton MetricsTracker instance."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    path = metrics_path or (OUTPUT_DIR / "metrics.json")
                    cls._instance = cls(path)
                    logger.info(f"MetricsTracker created: {path}")
        return cls._instance

    def _load(self) -> None:
        """Load metrics from disk."""
        if self._metrics_path.exists():
            with open(self._metrics_path, 'r') as f:
                data = json.load(f)
                self._metrics = [MetricPoint.from_dict(m) for m in data.get("metrics", [])]
                self._tags_index = data.get("tags_index", {})
            logger.debug(f"Loaded {len(self._metrics)} metric points")

    def _save(self) -> None:
        """Save metrics to disk."""
        data = {
            "metrics": [m.to_dict() for m in self._metrics],
            "tags_index": self._tags_index,
        }
        with open(self._metrics_path, 'w') as f:
            json.dump(data, f, indent=2)

    def extract_metrics(self, experiment_output: str) -> Dict[str, float]:
        """Extract metrics from experiment output using patterns."""
        config = METRICS_CONFIG
        metrics = {}

        # Extract accuracy
        for pattern in config.accuracy_patterns:
            matches = re.findall(pattern, experiment_output, re.IGNORECASE)
            if matches:
                try:
                    metrics["accuracy"] = float(matches[-1])  # Use last match
                    break
                except ValueError:
                    pass

        # Extract loss
        for pattern in config.loss_patterns:
            matches = re.findall(pattern, experiment_output, re.IGNORECASE)
            if matches:
                try:
                    metrics["loss"] = float(matches[-1])
                    break
                except ValueError:
                    pass

        # Extract forgetting
        for pattern in config.forgetting_patterns:
            matches = re.findall(pattern, experiment_output, re.IGNORECASE)
            if matches:
                try:
                    metrics["forgetting"] = float(matches[-1])
                    break
                except ValueError:
                    pass

        # Look for DELTA values (common in experiment outputs)
        delta_pattern = r"DELTA[:\s]+([-\d.]+)%?"
        matches = re.findall(delta_pattern, experiment_output, re.IGNORECASE)
        if matches:
            try:
                metrics["delta"] = float(matches[-1])
            except ValueError:
                pass

        # Look for BASELINE values
        baseline_pattern = r"BASELINE[:\s]+(\d+\.?\d*)%?"
        matches = re.findall(baseline_pattern, experiment_output, re.IGNORECASE)
        if matches:
            try:
                metrics["baseline"] = float(matches[-1])
            except ValueError:
                pass

        return metrics

    def record_metrics(
        self,
        cycle_id: int,
        experiment_id: str,
        metrics: Dict[str, float],
        tags: Optional[List[str]] = None
    ) -> None:
        """Record metrics for an experiment."""
        timestamp = datetime.now().isoformat()
        tags = tags or []

        for metric_name, value in metrics.items():
            point = MetricPoint(
                cycle_id=cycle_id,
                experiment_id=experiment_id,
                timestamp=timestamp,
                metric_name=metric_name,
                value=value,
                tags=tags,
            )
            self._metrics.append(point)

        # Update tags index
        if tags:
            self._tags_index[experiment_id] = tags

        self._save()
        logger.info(f"Recorded {len(metrics)} metrics for experiment {experiment_id}")

    def add_tags(self, experiment_id: str, tags: List[str]) -> None:
        """Add tags to an experiment."""
        existing = self._tags_index.get(experiment_id, [])
        combined = list(set(existing + tags))
        self._tags_index[experiment_id] = combined

        # Update metric points
        for m in self._metrics:
            if m.experiment_id == experiment_id:
                m.tags = combined

        self._save()

    def get_metric_series(
        self,
        metric_name: str,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get time series of a metric, optionally filtered by tags."""
        series = []
        for m in self._metrics:
            if m.metric_name != metric_name:
                continue
            if tags and not any(t in m.tags for t in tags):
                continue
            series.append({
                "cycle_id": m.cycle_id,
                "value": m.value,
                "timestamp": m.timestamp,
                "experiment_id": m.experiment_id,
            })
        return sorted(series, key=lambda x: x["cycle_id"])

    def get_summary_stats(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        values = [m.value for m in self._metrics if m.metric_name == metric_name]
        if not values:
            return {}

        import statistics
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
        if len(values) >= 2:
            stats["std"] = statistics.stdev(values)
        return stats

    def get_experiments_by_tag(self, tag: str) -> List[str]:
        """Get all experiment IDs with a specific tag."""
        return [exp_id for exp_id, tags in self._tags_index.items() if tag in tags]

    def generate_progress_report(self) -> str:
        """Generate a research progress dashboard."""
        report = ["## Research Progress Dashboard\n"]

        # Overall stats
        total_experiments = len(set(m.experiment_id for m in self._metrics))
        total_cycles = len(set(m.cycle_id for m in self._metrics))
        report.append(f"**Total Experiments:** {total_experiments}")
        report.append(f"**Total Cycles:** {total_cycles}\n")

        # Metrics summary
        metric_names = set(m.metric_name for m in self._metrics)
        if metric_names:
            report.append("### Metric Summaries\n")
            for metric in sorted(metric_names):
                stats = self.get_summary_stats(metric)
                if stats:
                    report.append(f"**{metric.title()}:**")
                    report.append(f"  - Mean: {stats['mean']:.2f}")
                    report.append(f"  - Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                    if 'std' in stats:
                        report.append(f"  - Std: {stats['std']:.2f}")
                    report.append("")

        # Trend analysis (last 5 cycles)
        if self._metrics:
            report.append("### Recent Trends (Last 5 Cycles)\n")
            recent_cycles = sorted(set(m.cycle_id for m in self._metrics))[-5:]
            for metric in sorted(metric_names):
                series = self.get_metric_series(metric)
                recent = [s for s in series if s["cycle_id"] in recent_cycles]
                if len(recent) >= 2:
                    trend = recent[-1]["value"] - recent[0]["value"]
                    direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                    report.append(f"- {metric.title()}: {direction} {abs(trend):.2f}")

        # Tags distribution
        if self._tags_index:
            report.append("\n### Experiment Tags\n")
            tag_counts: Dict[str, int] = {}
            for tags in self._tags_index.values():
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
                report.append(f"- `{tag}`: {count} experiments")

        return "\n".join(report)

    @property
    def metric_count(self) -> int:
        """Get total number of metric points."""
        return len(self._metrics)


# Create singleton instance
metrics_tracker = MetricsTracker.get()


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
# EXPERIMENT VERSIONING TOOLS
# ============================================================================

def save_experiment_version(
    hypothesis: str,
    experiment_output: str,
    tool_context: ToolContext = None
) -> str:
    """
    Save an experiment version with code snapshot and environment info.

    Automatically extracts:
    - Python code from the experiment output
    - Random seeds used
    - Hyperparameters (lr, batch_size, epochs, etc.)
    - Environment info (Python, PyTorch, NumPy versions)
    - Git commit hash

    Args:
        hypothesis: The hypothesis being tested
        experiment_output: Raw output from architect (contains code and results)

    Returns:
        Confirmation with experiment ID and extracted info
    """
    # Get current cycle ID from memory
    memory = memory_manager.load()
    cycle_id = len(memory.history) + 1

    # Save experiment
    version = experiment_manager.save_experiment(
        cycle_id=cycle_id,
        hypothesis=hypothesis,
        experiment_output=experiment_output,
    )

    return f"""Experiment saved:
- ID: {version.experiment_id}
- Cycle: {version.cycle_id}
- Code hash: {version.code_hash}
- Seeds: {version.random_seeds}
- Hyperparameters: {json.dumps(version.hyperparameters)}
- Environment: Python {version.python_version}, PyTorch {version.torch_version}
- Git commit: {version.git_commit}
- Code saved to: outputs/experiments/{version.experiment_id}/experiment.py"""


def get_experiment_version(
    experiment_id: Optional[str] = None,
    cycle_id: Optional[int] = None,
    tool_context: ToolContext = None
) -> str:
    """
    Retrieve an experiment version by ID or cycle number.

    Args:
        experiment_id: Experiment ID (e.g., "exp_001_abc12345")
        cycle_id: Alternatively, retrieve by cycle number

    Returns:
        Experiment details including code, seeds, hyperparameters, and environment
    """
    version = None

    if experiment_id:
        version = experiment_manager.get_experiment(experiment_id)
    elif cycle_id:
        version = experiment_manager.get_experiment_by_cycle(cycle_id)
    else:
        return "Please provide either experiment_id or cycle_id"

    if not version:
        return f"Experiment not found (id={experiment_id}, cycle={cycle_id})"

    return f"""## Experiment: {version.experiment_id}

**Cycle:** {version.cycle_id}
**Timestamp:** {version.timestamp}
**Code Hash:** {version.code_hash}

### Hypothesis
{version.hypothesis[:200]}...

### Reproducibility Info
- **Seeds:** {version.random_seeds}
- **Hyperparameters:** {json.dumps(version.hyperparameters, indent=2)}

### Environment
- Python: {version.python_version}
- PyTorch: {version.torch_version}
- NumPy: {version.numpy_version}
- Platform: {version.platform_info}
- Git Commit: {version.git_commit}

### Code
```python
{version.code[:500]}...
```

### Re-run Command
```bash
cd outputs/experiments/{version.experiment_id}
python experiment.py
```"""


def list_experiment_versions(
    limit: int = 10,
    tool_context: ToolContext = None
) -> str:
    """
    List all saved experiment versions.

    Args:
        limit: Maximum number of experiments to list (default 10)

    Returns:
        Table of experiments with IDs, cycles, timestamps, and code hashes
    """
    experiments = experiment_manager.list_experiments()

    if not experiments:
        return "No experiments have been saved yet."

    output = ["## Saved Experiments\n"]
    output.append("| Cycle | Experiment ID | Timestamp | Code Hash |")
    output.append("|-------|---------------|-----------|-----------|")

    for exp in experiments[-limit:]:
        timestamp = exp["timestamp"][:19] if exp["timestamp"] else "N/A"
        output.append(
            f"| {exp['cycle_id']} | {exp['experiment_id']} | {timestamp} | {exp['code_hash']} |"
        )

    output.append(f"\nTotal experiments: {len(experiments)}")
    return "\n".join(output)


def compare_experiments(
    experiment_id_1: str,
    experiment_id_2: str,
    tool_context: ToolContext = None
) -> str:
    """
    Compare two experiments to identify differences.

    Compares:
    - Code (by hash)
    - Random seeds
    - Hyperparameters
    - Environment versions

    Args:
        experiment_id_1: First experiment ID
        experiment_id_2: Second experiment ID

    Returns:
        Detailed comparison showing what differs between experiments
    """
    comparison = experiment_manager.compare_experiments(experiment_id_1, experiment_id_2)

    if "error" in comparison:
        return comparison["error"]

    output = [f"## Comparison: {experiment_id_1} vs {experiment_id_2}\n"]

    # Code comparison
    if comparison["same_code"]:
        output.append("✅ **Code:** Identical (same hash)")
    else:
        output.append(f"❌ **Code:** Different")
        output.append(f"   - Exp 1 hash: {comparison['code_hash_1']}")
        output.append(f"   - Exp 2 hash: {comparison['code_hash_2']}")

    # Seed comparison
    seed_diff = comparison["seed_diff"]
    if not seed_diff["only_in_1"] and not seed_diff["only_in_2"]:
        output.append("✅ **Seeds:** Identical")
    else:
        output.append("❌ **Seeds:** Different")
        if seed_diff["only_in_1"]:
            output.append(f"   - Only in exp 1: {seed_diff['only_in_1']}")
        if seed_diff["only_in_2"]:
            output.append(f"   - Only in exp 2: {seed_diff['only_in_2']}")

    # Hyperparameter comparison
    hp_diff = comparison["hyperparam_diff"]
    if not hp_diff:
        output.append("✅ **Hyperparameters:** Identical")
    else:
        output.append("❌ **Hyperparameters:** Different")
        for param, values in hp_diff.items():
            output.append(f"   - {param}: {values['exp1']} → {values['exp2']}")

    # Environment comparison
    env_diff = comparison["environment_diff"]
    if not env_diff:
        output.append("✅ **Environment:** Identical")
    else:
        output.append("⚠️ **Environment:** Different")
        for field, values in env_diff.items():
            output.append(f"   - {field}: {values['exp1']} → {values['exp2']}")

    return "\n".join(output)


def get_rerun_command(
    experiment_id: str,
    tool_context: ToolContext = None
) -> str:
    """
    Generate a command to re-run an experiment.

    Args:
        experiment_id: The experiment to re-run

    Returns:
        Shell command with context about the original experiment
    """
    return experiment_manager.generate_rerun_command(experiment_id)


# Create experiment versioning FunctionTool wrappers
save_experiment_tool = FunctionTool(func=save_experiment_version)
get_experiment_tool = FunctionTool(func=get_experiment_version)
list_experiments_tool = FunctionTool(func=list_experiment_versions)
compare_experiments_tool = FunctionTool(func=compare_experiments)
rerun_experiment_tool = FunctionTool(func=get_rerun_command)


# ============================================================================
# METRICS AND TAGGING TOOLS
# ============================================================================

def record_experiment_metrics(
    experiment_id: str,
    experiment_output: str,
    tags: Optional[List[str]] = None,
    tool_context: ToolContext = None
) -> str:
    """
    Extract and record metrics from an experiment's output.

    Automatically extracts accuracy, loss, forgetting, delta, and baseline metrics.

    Args:
        experiment_id: The experiment ID to record metrics for
        experiment_output: Raw output containing metric values
        tags: Optional list of tags (e.g., ["ewc", "replay", "permuted-mnist"])

    Returns:
        Summary of recorded metrics
    """
    # Get cycle ID from experiment
    exp = experiment_manager.get_experiment(experiment_id)
    if not exp:
        return f"Experiment {experiment_id} not found"

    cycle_id = exp.cycle_id

    # Extract metrics
    metrics = metrics_tracker.extract_metrics(experiment_output)
    if not metrics:
        return f"No metrics found in experiment output for {experiment_id}"

    # Record metrics
    metrics_tracker.record_metrics(
        cycle_id=cycle_id,
        experiment_id=experiment_id,
        metrics=metrics,
        tags=tags
    )

    return f"""Recorded metrics for {experiment_id}:
{json.dumps(metrics, indent=2)}
Tags: {tags or 'none'}"""


def tag_experiment(
    experiment_id: str,
    tags: List[str],
    tool_context: ToolContext = None
) -> str:
    """
    Add tags to an experiment for better organization and filtering.

    Common tags: ewc, replay, regularization, sam, permuted-mnist, split-mnist

    Args:
        experiment_id: The experiment to tag
        tags: List of tags to add

    Returns:
        Confirmation of tags added
    """
    # Verify experiment exists
    exp = experiment_manager.get_experiment(experiment_id)
    if not exp:
        return f"Experiment {experiment_id} not found"

    metrics_tracker.add_tags(experiment_id, tags)
    return f"Added tags to {experiment_id}: {tags}"


def get_metric_trends(
    metric_name: str = "accuracy",
    tags: Optional[List[str]] = None,
    tool_context: ToolContext = None
) -> str:
    """
    Get trend data for a specific metric over time.

    Args:
        metric_name: Name of metric (accuracy, loss, forgetting, delta, baseline)
        tags: Optional filter by tags

    Returns:
        Time series data and statistics for the metric
    """
    series = metrics_tracker.get_metric_series(metric_name, tags)
    stats = metrics_tracker.get_summary_stats(metric_name)

    if not series:
        return f"No data found for metric '{metric_name}'"

    output = [f"## {metric_name.title()} Trends\n"]

    if tags:
        output.append(f"*Filtered by tags: {tags}*\n")

    # Stats
    if stats:
        output.append("### Statistics")
        output.append(f"- Count: {stats['count']}")
        output.append(f"- Mean: {stats['mean']:.2f}")
        output.append(f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        if 'std' in stats:
            output.append(f"- Std Dev: {stats['std']:.2f}")
        output.append("")

    # Time series
    output.append("### Time Series")
    output.append("| Cycle | Value | Experiment |")
    output.append("|-------|-------|------------|")
    for point in series[-10:]:  # Last 10 points
        output.append(f"| {point['cycle_id']} | {point['value']:.2f} | {point['experiment_id'][:15]}... |")

    return "\n".join(output)


def get_experiments_by_tag(
    tag: str,
    tool_context: ToolContext = None
) -> str:
    """
    Find all experiments with a specific tag.

    Args:
        tag: Tag to search for

    Returns:
        List of matching experiment IDs
    """
    exp_ids = metrics_tracker.get_experiments_by_tag(tag)
    if not exp_ids:
        return f"No experiments found with tag '{tag}'"

    output = [f"## Experiments tagged '{tag}'\n"]
    for exp_id in exp_ids:
        exp = experiment_manager.get_experiment(exp_id)
        if exp:
            output.append(f"- **{exp_id}** (Cycle {exp.cycle_id})")
            output.append(f"  Hash: {exp.code_hash}")

    return "\n".join(output)


def get_progress_dashboard(tool_context: ToolContext = None) -> str:
    """
    Generate a research progress dashboard with summary statistics and trends.

    Returns:
        Comprehensive progress report with metrics, trends, and tags
    """
    return metrics_tracker.generate_progress_report()


# Create metrics and tagging FunctionTool wrappers
record_metrics_tool = FunctionTool(func=record_experiment_metrics)
tag_experiment_tool = FunctionTool(func=tag_experiment)
get_trends_tool = FunctionTool(func=get_metric_trends)
get_by_tag_tool = FunctionTool(func=get_experiments_by_tag)
progress_dashboard_tool = FunctionTool(func=get_progress_dashboard)


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
    description="Saves research cycle results to persistent memory, experiment versions, and metrics",
    instruction="""You are the MEMORY SAVER - you persist research findings, experiment versions, and metrics.

After each research cycle, extract and save the key information.

CYCLE DATA:
Hypothesis: {hypothesis}
Critique: {critique}
Gate Decision: {gate_decision?}
Experiment Code & Output: {experiment_code?}
Analysis: {analysis?}
Simple Explanation: {simple_explanation?}
Editorial Review: {editorial_review?}

YOUR TASKS:

1. SAVE RESEARCH CYCLE TO MEMORY:
   - Extract the VERDICT from the analysis (SUPPORTED/REFUTED/INCONCLUSIVE)
   - Extract the KEY LEARNING from the editorial review
   - Extract any SURPRISE FINDING from the analysis or editorial review
     (prefer the analyst's SURPRISES field if both are present)
   - Use the save_cycle_to_memory tool to persist this cycle

2. SAVE EXPERIMENT VERSION (if experiment was run):
   - If experiment_code contains actual Python code (not empty/placeholder):
   - Use save_experiment_version tool with:
     - hypothesis: the tested hypothesis
     - experiment_output: the full experiment_code field
   - This saves a reproducible snapshot with code, seeds, hyperparameters, and environment
   - Note the experiment_id returned for step 3

3. RECORD METRICS (if experiment was run):
   - After saving experiment version, use record_experiment_metrics tool with:
     - experiment_id: the ID from step 2
     - experiment_output: the full experiment_code field
     - tags: relevant method tags (e.g., ["ewc"], ["replay"], ["sam"])
   - This tracks accuracy, loss, forgetting metrics over time

NOTE: If the gate decision was REJECT, skip steps 2 and 3.
Set verdict to "REJECTED" and extract key learning from the gate_decision reasoning.

Be concise and accurate. Extract exact values, don't paraphrase.""",
    tools=[save_cycle_tool, save_experiment_tool, record_metrics_tool],
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
8. MEMORY SAVER - persists findings, experiment versions, and metrics to disk

COMMANDS:
- "run cycle" - Execute one complete research cycle (auto-saves to memory)
- "synthesize" - Transfer to SYNTHESIZER to analyze patterns across all cycles
- "report" or "write paper" - Transfer to REPORTER to generate a scientific paper
- "show memory" - Display past insights and failures using get_memory_context tool
- "show all cycles" - Display all past research cycles using get_all_cycles tool

EXPERIMENT VERSIONING:
Each experiment is automatically versioned with code snapshots, random seeds,
hyperparameters, and environment info for full reproducibility.
- "list experiments" - Show all saved experiment versions
- "show experiment <id or cycle>" - Display experiment details including code and environment
- "compare experiments <id1> <id2>" - Compare two experiments to see differences
- "rerun experiment <id>" - Get command to re-run a past experiment

METRICS & PROGRESS TRACKING:
Metrics are automatically extracted and tracked across experiments.
- "dashboard" or "progress" - Show research progress dashboard with trends
- "trends <metric>" - Show trends for a specific metric (accuracy, loss, forgetting)
- "tag experiment <id> <tags>" - Add tags to an experiment for filtering
- "find by tag <tag>" - Find experiments with a specific tag

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
    tools=[
        get_memory_tool,
        get_cycles_tool,
        list_experiments_tool,
        get_experiment_tool,
        compare_experiments_tool,
        rerun_experiment_tool,
        tag_experiment_tool,
        get_trends_tool,
        get_by_tag_tool,
        progress_dashboard_tool,
    ]
)

logger.info(f"Forgetting Lab initialized with {memory_manager.cycle_count} existing research cycles")
logger.info(f"Experiment versions saved: {experiment_manager.experiment_count}")
logger.info(f"Metrics tracked: {metrics_tracker.metric_count} data points")
