# Agentic Orchestration

Patterns and techniques for building Claude-powered agents and multi-agent systems.

## Core Agent Capabilities (Claude 4.5)

### Extended Autonomous Operation

Claude 4.5 can work independently for hours while maintaining clarity and focus on incremental progress. The model makes steady advances on a few tasks at a time rather than attempting everything at once.

### Native Subagent Orchestration

Claude 4.5 models can recognize when tasks benefit from delegating to specialized subagents and do so proactively without explicit instruction.

To take advantage of this behavior:
- Ensure well-defined subagent tools with clear descriptions
- Let Claude orchestrate naturally
- Adjust conservativeness if needed:

```
Only delegate to subagents when the task clearly benefits from a separate agent with a new context window.
```

### Context Awareness

Claude tracks its token usage throughout conversations, receiving updates after each tool call. This prevents premature task abandonment and enables effective execution on long-running tasks.

```xml
<context_management>
Your context window will be automatically compacted as it approaches its limit, allowing you to continue working indefinitely. Do not stop tasks early due to token budget concerns. Save progress and state before context refreshes. Always be persistent and autonomous—complete tasks fully even near budget limits.
</context_management>
```

## Agent Patterns

### The Agent Loop

The core pattern for tool-using agents:

1. Send user message to Claude
2. Claude requests tool use (may be sufficient for some workflows)
3. Execute tool and return results
4. Claude analyzes results for response or further tool use
5. Repeat steps 2-4 until task complete

```python
async def agent_loop(model, messages, max_iterations=10):
    """Basic agent loop pattern."""
    for _ in range(max_iterations):
        response = await client.messages.create(
            model=model,
            messages=messages,
            tools=tools
        )
        
        if response.stop_reason == "end_turn":
            return response
            
        # Process tool calls
        for block in response.content:
            if block.type == "tool_use":
                result = await execute_tool(block)
                messages.append({"role": "user", "content": [result]})
```

### Parallel Tool Calling

Claude 4.x excels at parallel execution. Enable aggressive parallelism:

```xml
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between calls, make all independent calls in parallel. Prioritize simultaneous tool calls for actions that can be done concurrently. For example, when reading 3 files, run 3 tool calls in parallel. Never use placeholders or guess missing parameters.
</use_parallel_tool_calls>
```

### Plan-Validate-Execute Pattern

For complex tasks, add validation before execution:

1. **Analyze**: Understand the task requirements
2. **Plan**: Create structured plan (e.g., `changes.json`)
3. **Validate**: Run validation script on plan
4. **Execute**: Apply changes only if validation passes
5. **Verify**: Confirm results

When to use:
- Batch operations
- Destructive changes
- Complex validation rules
- High-stakes operations

## Subagent Architecture

### Benefits of Subagents

**Context Isolation**: Specialized tasks don't pollute the main conversation. A research subagent can explore dozens of files without cluttering the main context—only returning relevant findings.

**Parallelization**: Multiple subagents can run concurrently. During code review, run style-checker, security-scanner, and test-coverage subagents simultaneously.

**Specialized Expertise**: Each subagent has tailored prompts with specific knowledge. A database-migration subagent can have detailed SQL best practices that would be noise in the main agent.

**Tool Restriction**: Subagents can be limited to specific tools. A doc-reviewer might only have Read and Grep, ensuring it can analyze but never modify files.

### Defining Subagents

**Programmatic Definition (Agent SDK)**:

```javascript
const result = query({
  prompt: "Review authentication module for security issues",
  options: {
    agents: {
      'code-reviewer': {
        description: 'Expert code review specialist. Use for quality, security, and maintainability reviews.',
        prompt: `You are a code review specialist with expertise in security, performance, and best practices.

When reviewing code:
- Identify security vulnerabilities
- Check for performance issues
- Verify adherence to coding standards
- Suggest specific improvements

Be thorough but concise in your feedback.`,
        tools: ['Read', 'Grep', 'Glob'],
        model: 'sonnet'
      }
    }
  }
});
```

**Filesystem Definition**:

Create markdown files in `.claude/agents/`:

```markdown
---
name: code-reviewer
description: Expert code review specialist. Use for quality, security, and maintainability reviews.
tools: Read, Grep, Glob, Bash
---

Your subagent's system prompt goes here. This defines the subagent's role, capabilities, and approach to solving problems.
```

Locations:
- **Project-level**: `.claude/agents/*.md` (current project only)
- **User-level**: `~/.claude/agents/*.md` (all projects)

### Automatic Invocation

Claude will:
- Load programmatic agents from options
- Auto-detect filesystem agents from `.claude/agents/`
- Invoke automatically based on task matching and description
- Use specialized prompts and tool restrictions
- Maintain separate context for each invocation

Ensure description fields clearly indicate when to use:

```javascript
'performance-optimizer': {
  description: 'Use PROACTIVELY when code changes might impact performance.'
}
```

## Agent Skills

Skills are reusable, filesystem-based resources that provide domain-specific expertise: workflows, context, and best practices that transform general-purpose agents into specialists.

### Skill Structure

```
skill-name/
├── SKILL.md          # Required: instructions + YAML frontmatter
├── scripts/          # Executable code
├── references/       # Documentation to load as needed
└── assets/           # Templates, images, fonts
```

### Progressive Disclosure

Skills use three-level loading:
1. **Metadata** (name + description): Always in context (~100 words)
2. **SKILL.md body**: When skill triggers (<5k words)
3. **Bundled resources**: As needed by Claude

### Using Skills

**Claude API**:
```python
response = client.beta.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    betas=["code-execution-2025-08-25", "skills-2025-10-02"],
    container={
        "skills": [
            {"type": "anthropic", "skill_id": "xlsx", "version": "latest"},
            {"type": "custom", "skill_id": custom_skill.id, "version": "latest"}
        ]
    },
    messages=[{"role": "user", "content": "Create a financial model"}],
    tools=[{"type": "code_execution_20250825", "name": "code_execution"}]
)
```

**Claude Code**: Create SKILL.md files in `.claude/skills/`—Claude discovers and uses them automatically.

## Hosting Patterns

### Ephemeral Containers

Create new container for each task, destroy when complete.

**Best for**: One-off tasks, user may interact during task but container destroyed after.

### Long-Running Containers

Maintain persistent instances for extended tasks.

**Best for**:
- Email agents (monitor and triage autonomously)
- Site builders (live editing, served through container ports)
- High-frequency chat bots (rapid response times)

### Resumable Sessions

Ephemeral containers hydrated with history and state from database or session resumption.

**Best for**:
- Personal project managers (intermittent check-ins)
- Deep research (multi-hour tasks, save findings, resume later)
- Customer support (ticket history across interactions)

### Multi-Agent Containers

Multiple Claude processes in one container.

**Best for**: Agents that must collaborate closely (simulations, games).

**Caution**: Prevent agents from overwriting each other.

## Agentic Research

Claude 4.5 demonstrates exceptional agentic search capabilities.

### Structured Research Approach

```xml
<structured_research>
Search for this information in a structured way. As you gather data, develop several competing hypotheses. Track your confidence levels in progress notes to improve calibration. Regularly self-critique your approach and plan. Update a hypothesis tree or research notes file to persist information and provide transparency. Break down this complex research task systematically.
</structured_research>
```

### Best Practices

- **Clear success criteria**: Define what constitutes a successful answer
- **Source verification**: Ask Claude to verify across multiple sources
- **Hypothesis tracking**: Develop competing hypotheses as data is gathered
- **Self-critique**: Regularly evaluate approach and plan
- **State persistence**: Update notes files for transparency

## MCP Integration

Extend agents with custom tools through MCP (Model Context Protocol) servers.

```javascript
import { query, tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

const customServer = createSdkMcpServer({
  name: "my-custom-tools",
  version: "1.0.0",
  tools: [
    tool(
      "get_weather",
      "Get current temperature for a location",
      { latitude: z.number(), longitude: z.number() },
      async (args) => {
        const response = await fetch(`https://api.open-meteo.com/...`);
        const data = await response.json();
        return {
          content: [{ type: "text", text: `Temperature: ${data.current.temperature_2m}°F` }]
        };
      }
    )
  ]
});

// Use in query
for await (const message of query({
  prompt: "What's the weather in San Francisco?",
  options: {
    mcpServers: { "my-custom-tools": customServer },
    allowedTools: ["mcp__my-custom-tools__get_weather"]
  }
})) {
  console.log(message);
}
```

## State Management

### Use Structured Formats

For test results, task status—use JSON:

```json
{
  "tests": [
    {"name": "auth_flow", "status": "pass"},
    {"name": "data_validation", "status": "fail", "error": "..."}
  ]
}
```

### Use Unstructured Text for Progress

Freeform notes work well for general progress:

```
## Progress Notes
- Completed auth module review
- Found 3 potential SQL injection points
- Next: review API rate limiting
```

### Use Git for State Tracking

Git provides checkpoints that can be restored across sessions. Claude 4.5 models perform especially well using git to track state.

### Incremental Progress

Explicitly ask Claude to track progress and focus on incremental work:

```
This is a very long task, so plan your work clearly. Spend your entire output context working on the task—just make sure you don't run out of context with significant uncommitted work. Continue working systematically until complete.
```

## Production Considerations

### Error Handling

- Set `maxTurns` to prevent infinite loops
- Implement tool execution timeouts
- Handle API rate limits gracefully

### Cost Management

- Token cost dominates (containers ~$0.05/hour)
- Use efficient models for subagents (Haiku for simple tasks)
- Cache prompt prefixes where possible

### Security

- Run in sandboxed container environments
- Process isolation per session
- Resource limits (CPU, memory, storage)
- Validate skill sources before use

## Prompt Patterns for Agents

### Action vs. Suggestion

**Encourage action**:
```xml
<default_to_action>
Implement changes rather than only suggesting them. If intent is unclear, infer the most useful action and proceed.
</default_to_action>
```

**Encourage caution**:
```xml
<do_not_act_before_instructions>
Do not implement changes unless explicitly instructed. Default to providing information and recommendations.
</do_not_act_before_instructions>
```

### Verification Loops

```
After completing the task, verify your solution:
1. Run the test suite
2. Check for lint errors
3. Confirm the original requirements are met
4. Fix any issues found
```

### Tool Usage Triggers

Claude Opus 4.5 is more responsive to system prompts. Dial back aggressive language:
- Instead of: "CRITICAL: You MUST use this tool when..."
- Try: "Use this tool when..."
