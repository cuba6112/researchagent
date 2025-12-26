---
name: prompt-engineering
description: Comprehensive prompt engineering techniques for Claude models. Use this skill when crafting, optimizing, or debugging prompts for Claude API, Claude Code, or any Claude-powered application. Covers system prompts, role prompting, multishot examples, chain of thought, XML structuring, long context handling, extended thinking, prompt chaining, Claude 4.x-specific best practices, and agentic orchestration including subagents, agent loops, skills, MCP integration, and multi-agent workflows.
---

# Prompt Engineering

Prompt engineering techniques derived from Anthropic's official documentation for Claude models.

## Core Principles

### Be Explicit and Direct

Claude responds best to clear, specific instructions. Treat Claude like a brilliant new employee who needs explicit guidance about your norms, styles, and preferences.

**The Golden Rule**: Show your prompt to a colleague with minimal context. If they're confused, Claude will be too.

**Provide Context**:
- What the task results will be used for
- What audience the output is meant for
- Where this task fits in your workflow
- What successful completion looks like

**Be Specific**: If you want only code output, say so. Use numbered steps for sequential tasks.

### Use XML Tags for Structure

XML tags dramatically improve Claude's accuracy by clearly separating prompt components.

```xml
<instructions>
Your task instructions here
</instructions>

<context>
Background information
</context>

<examples>
<example>
Input: X
Output: Y
</example>
</examples>
```

**Best Practices**:
- Be consistent with tag names throughout prompts
- Reference tags explicitly: "Using the data in <context> tags..."
- Nest tags for hierarchical content: `<outer><inner></inner></outer>`
- Combine with examples (`<examples>`) or chain of thought (`<thinking>`, `<answer>`)

### Use Examples (Multishot Prompting)

Examples are the most effective way to guide Claude's output format and style.

**Guidelines**:
- Include 3-5 diverse, relevant examples
- Cover edge cases and challenges
- Wrap in `<example>` tags (nested within `<examples>` if multiple)
- Ensure examples match desired behaviors exactly—Claude pays close attention to details

## System Prompts and Role Prompting

Use the `system` parameter to set Claude's role. This is the most powerful use of system prompts.

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    system="You are a senior data scientist specializing in anomaly detection.",
    messages=[{"role": "user", "content": "Analyze this dataset..."}]
)
```

**Benefits**:
- Enhanced accuracy in complex domains (legal, financial, technical)
- Tailored communication style
- Improved task focus

**Tip**: Be specific with roles. "A data scientist specializing in customer insights for Fortune 500 companies" yields better results than just "a data scientist."

## Chain of Thought (CoT) Prompting

Encourage Claude to break down complex problems step-by-step.

**Three Approaches** (least to most structured):

1. **Basic**: Add "Think step-by-step" to your prompt
2. **Guided**: Outline specific thinking steps
3. **Structured**: Use XML tags to separate reasoning from answer

```xml
<instructions>
Analyze this problem. Put your reasoning in <thinking> tags, then provide your final answer in <answer> tags.
</instructions>
```

**When to Use**: Math, logic, analysis, complex multi-step tasks
**When to Skip**: Simple tasks where latency matters

## Prompt Chaining

Break complex tasks into smaller, sequential subtasks for improved accuracy and traceability.

**Example Workflows**:
- Research → Outline → Draft → Edit → Format
- Extract → Transform → Analyze → Visualize
- Gather info → List options → Analyze → Recommend

**Implementation**:
- Each subtask gets one clear objective
- Use XML tags for clean handoffs between steps
- Debug by isolating problematic steps

**Parallel Execution**: For independent subtasks, run prompts in parallel for speed.

## Long Context Handling

Claude supports up to 200K tokens. For best results with large inputs:

1. **Place documents at the top** of your prompt, above instructions and examples
2. **Structure with XML tags**:
```xml
<document>
  <source>document_name.pdf</source>
  <document_content>...</document_content>
</document>
```
3. **Ground responses in quotes**: Ask Claude to quote relevant sections before answering
4. **Query at the end**: Placing your question after documents can improve quality by up to 30%

## Extended Thinking

Extended thinking enables deep reasoning for complex problems. See `references/extended-thinking.md` for detailed guidance.

**Key Points**:
- Start with minimum budget (1024 tokens), increase as needed
- Use high-level instructions rather than prescriptive step-by-step guidance
- Performs best in English; outputs can be any language
- For thinking below minimum budget, use standard CoT with `<thinking>` tags

**Prompting Tips**:
```
Please think about this thoroughly and in great detail.
Consider multiple approaches and show your complete reasoning.
Try different methods if your first approach doesn't work.
```

## Agentic Orchestration

See `references/agentic-orchestration.md` for comprehensive agent patterns.

### Key Concepts

**Agent Loop**: The core pattern—send message, Claude requests tools, execute tools, return results, repeat until complete.

**Subagents**: Specialized agents with separate context for parallelization and expertise isolation. Define in `.claude/agents/` or programmatically.

**Skills**: Reusable filesystem-based capabilities that load on-demand. Create `SKILL.md` files with instructions, scripts, and references.

### Native Orchestration (Claude 4.5)

Claude 4.5 proactively delegates to subagents when beneficial:
```
Only delegate to subagents when the task clearly benefits from a separate agent with a new context window.
```

### Plan-Validate-Execute Pattern

For high-stakes operations:
1. Analyze requirements
2. Create structured plan (e.g., `changes.json`)
3. Validate plan with script
4. Execute only if validation passes
5. Verify results

### Agentic Research

```xml
<structured_research>
Search in a structured way. Develop competing hypotheses. Track confidence levels in progress notes. Regularly self-critique your approach. Update research notes for transparency.
</structured_research>
```

## Claude 4.x Specific Guidance

See `references/claude-4-best-practices.md` for comprehensive Claude 4.x techniques.

**Key Differences**:
- More precise instruction following—be explicit about desired behaviors
- Add context/motivation for better results
- More concise, direct communication style
- Sensitive to the word "think" when extended thinking is disabled (use "consider", "evaluate" instead)

### Controlling Output Format

Tell Claude what to do (not what to avoid):
- Instead of: "Do not use markdown"
- Try: "Write in flowing prose paragraphs"

Use XML format indicators:
```xml
<smoothly_flowing_prose_paragraphs>
Your content here
</smoothly_flowing_prose_paragraphs>
```

### Tool Usage

Claude 4.x benefits from explicit direction. To encourage action:

```xml
<default_to_action>
Implement changes rather than only suggesting them. If intent is unclear, infer the most useful action and proceed, using tools to discover missing details.
</default_to_action>
```

To encourage caution:

```xml
<do_not_act_before_instructions>
Do not implement changes unless explicitly instructed. Default to providing information and recommendations rather than taking action.
</do_not_act_before_instructions>
```

### Parallel Tool Calling

Claude 4.x excels at parallel execution:

```xml
<use_parallel_tool_calls>
Call multiple independent tools simultaneously. Prioritize parallel calls for actions that can be done concurrently. Only use sequential calls when dependencies exist.
</use_parallel_tool_calls>
```

### Preventing Over-Engineering

Claude Opus 4.5 may over-engineer solutions. Add explicit constraints:

```xml
<avoid_over_engineering>
Only make changes directly requested or clearly necessary. Keep solutions simple. Don't add features, refactoring, or improvements beyond what was asked. Don't create helpers or abstractions for one-time operations.
</avoid_over_engineering>
```

### Encouraging Code Exploration

```xml
<code_exploration>
ALWAYS read and understand relevant files before proposing edits. Do not speculate about code you haven't inspected. Be rigorous in searching code for key facts.
</code_exploration>
```

### Minimizing Hallucinations

```xml
<investigate_before_answering>
Never speculate about code you have not opened. If the user references a file, read it before answering. Give grounded, hallucination-free answers.
</investigate_before_answering>
```

## Multi-Context Window Workflows

For tasks spanning multiple context windows:

1. **First window**: Set up framework (write tests, create setup scripts)
2. **Subsequent windows**: Iterate on todo-list

**Best Practices**:
- Write tests in structured format (e.g., `tests.json`)
- Create setup scripts (e.g., `init.sh`) for graceful restarts
- Use git for state tracking across sessions
- Start fresh rather than compacting when context clears

**Context Management Prompt**:
```
Your context window will be automatically compacted as it approaches its limit. Do not stop tasks early due to token budget concerns. Save progress and state before context refreshes. Complete tasks fully, even near budget limits.
```

## Quick Reference

| Technique | When to Use |
|-----------|-------------|
| Role prompting | Domain-specific tasks, specialized knowledge |
| Multishot examples | Structured outputs, specific formats |
| XML tags | Complex prompts with multiple components |
| Chain of thought | Math, logic, analysis, complex reasoning |
| Prompt chaining | Multi-step tasks, content pipelines |
| Extended thinking | Deep reasoning, verification, complex problems |
| Parallel tool calls | Multiple independent operations |
| Subagents | Context isolation, parallel specialized tasks |
| Agent skills | Reusable domain expertise, complex workflows |
| Plan-validate-execute | High-stakes operations, batch changes |

## Resources

- [Prompt Library](https://docs.anthropic.com/en/prompt-library)
- [Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)
- [Claude Console Prompt Generator](https://console.anthropic.com)
